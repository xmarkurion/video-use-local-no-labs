"""Tests for export_timeline.py (EDL -> XMEML / "Final Cut Pro XML").

Real end-to-end runs (actual ffprobe, actual media, actual multi-source EDLs
with overlays and a LUT) were done manually against the real DJI LUT already
in lut/ and real jfk/meeting test clips, and the output was re-parsed with
xml.dom.minidom to confirm strict well-formedness. These tests check the
frame-math and structure-construction logic in isolation, mocking ffprobe the
same way the rest of this suite mocks subprocess.
"""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

# render.py must be importable as a plain sibling module (export_timeline.py
# does `from render import ...` at its own top level) before it's loaded —
# see tests/test_lut_and_censor.py for the ordering bug this avoids.
HELPERS = Path(__file__).parents[1] / "helpers"
sys.path.insert(0, str(HELPERS))

MODULE_PATH = HELPERS / "export_timeline.py"
SPEC = importlib.util.spec_from_file_location("video_use_export_timeline", MODULE_PATH)
assert SPEC and SPEC.loader
export_timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_timeline)


def _ffprobe_stream_result(width=1920, height=1080):
    return subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"streams": [{"width": width, "height": height}]}), stderr="",
    )


def _ffprobe_rate_result(avg="25/1"):
    return subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"streams": [{"avg_frame_rate": avg, "r_frame_rate": avg}]}), stderr="",
    )


class RateToTimebaseTests(unittest.TestCase):
    def test_plain_integer_rates(self):
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(24, 1)), (24, False))
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(30, 1)), (30, False))
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(60, 1)), (60, False))

    def test_ntsc_family_rates(self):
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(30000, 1001)), (30, True))
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(24000, 1001)), (24, True))
        self.assertEqual(export_timeline.rate_to_timebase(Fraction(60000, 1001)), (60, True))

    def test_exotic_non_ntsc_rational_is_rounded_not_flagged_ntsc(self):
        timebase, ntsc = export_timeline.rate_to_timebase(Fraction(48000, 2000))  # = 24, denom != 1001
        self.assertFalse(ntsc)


class SecondsToFramesTests(unittest.TestCase):
    def test_exact_and_rounding(self):
        rate = Fraction(25, 1)
        self.assertEqual(export_timeline.seconds_to_frames(2.47, rate), 62)   # 61.75 -> 62
        self.assertEqual(export_timeline.seconds_to_frames(6.70, rate), 168)  # 167.5 -> 168
        self.assertEqual(export_timeline.seconds_to_frames(0.0, rate), 0)

    def test_never_negative(self):
        self.assertEqual(export_timeline.seconds_to_frames(-1.0, Fraction(25, 1)), 0)


class PathToPathurlTests(unittest.TestCase):
    def test_contains_file_localhost_scheme(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clip.mp4"
            p.write_bytes(b"\x00")
            url = export_timeline.path_to_pathurl(p)
            self.assertTrue(url.startswith("file://localhost/"))

    def test_backslashes_never_survive(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clip.mp4"
            p.write_bytes(b"\x00")
            self.assertNotIn("\\", export_timeline.path_to_pathurl(p))

    def test_spaces_are_percent_encoded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "DJI OSMO Action 4 D-Log M to Rec.709 V1.cube"
            p.write_text("LUT_3D_SIZE 2\n")
            url = export_timeline.path_to_pathurl(p)
            self.assertNotIn(" ", url)
            self.assertIn("%20", url)


class BuildTimelineFrameMathTests(unittest.TestCase):
    """The core correctness surface: output-timeline frame positions must be
    contiguous across ranges (no gaps/overlaps from accumulated rounding),
    and in/out must be computed against each source's OWN rate."""

    def _edl(self, ranges, **extra):
        return {
            "sources": {"A": "A.mp4", "B": "B.mp4"},
            "ranges": ranges,
            **extra,
        }

    def _run(self, edl, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            (edit_dir / "B.mp4").write_bytes(b"\x00")
            with (
                patch.object(export_timeline.subprocess, "run", return_value=_ffprobe_rate_result()),
                patch.object(export_timeline, "probe_dimensions", return_value=(1920, 1080)),
            ):
                root, notes = export_timeline.build_timeline(edl, edit_dir)
            return root, notes

    def test_two_ranges_are_contiguous_on_the_timeline(self):
        edl = self._edl([
            {"source": "A", "start": 0.0, "end": 2.0},
            {"source": "B", "start": 5.0, "end": 6.5},
        ])
        root, _ = self._run(edl)
        v1 = root.find("./sequence/media/video/track")
        clips = v1.findall("clipitem")
        self.assertEqual(len(clips), 2)
        first_end = int(clips[0].find("end").text)
        second_start = int(clips[1].find("start").text)
        self.assertEqual(first_end, second_start)  # no gap, no overlap

    def test_same_source_twice_reuses_the_file_definition(self):
        edl = self._edl([
            {"source": "A", "start": 0.0, "end": 1.0},
            {"source": "B", "start": 0.0, "end": 1.0},
            {"source": "A", "start": 3.0, "end": 4.0},
        ])
        root, _ = self._run(edl)
        v1 = root.find("./sequence/media/video/track")
        clips = v1.findall("clipitem")
        # First A clip: full <file> definition with children.
        first_a_file = clips[0].find("file")
        self.assertIsNotNone(first_a_file.find("name"))
        # Second A clip (the 3rd range overall): bare self-closing reference,
        # same id, no children — the xmeml "same media" convention.
        second_a_file = clips[2].find("file")
        self.assertEqual(second_a_file.get("id"), first_a_file.get("id"))
        self.assertEqual(len(list(second_a_file)), 0)

    def test_in_out_use_source_rate_start_end_use_sequence_rate(self):
        # Two sources at different native rates; sequence rate forced to 24.
        def fake_run(cmd, **kwargs):
            # cmd is an argv list; the probed path is one element of it, not
            # the whole list — membership needs a substring check per element.
            if any("A.mp4" in str(arg) for arg in cmd):
                return _ffprobe_rate_result("24/1")
            return _ffprobe_rate_result("60/1")

        edl = self._edl(
            [{"source": "A", "start": 1.0, "end": 2.0}],
        )
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            with (
                patch.object(export_timeline.subprocess, "run", side_effect=lambda cmd, **kw: fake_run(cmd, **kw)),
                patch.object(export_timeline, "probe_dimensions", return_value=(1920, 1080)),
            ):
                root, _ = export_timeline.build_timeline(edl, edit_dir, fps="30")
        clip = root.find("./sequence/media/video/track/clipitem")
        # in/out: where in the SOURCE file this range sits — source's own rate.
        self.assertEqual(clip.find("in").text, "24")   # 1.0s * 24fps
        self.assertEqual(clip.find("out").text, "48")  # 2.0s * 24fps
        # start/end: where on the OUTPUT TIMELINE this clip sits — sequence
        # rate, and since this is the only (first) range, that's 0 -> duration,
        # not a rate-converted copy of the source's own 1.0-2.0 window.
        self.assertEqual(clip.find("start").text, "0")   # output position 0
        self.assertEqual(clip.find("end").text, "30")    # + 1.0s duration * 30fps

    def test_audio_track_mirrors_video_positions_without_duplicate_file_defs(self):
        edl = self._edl([{"source": "A", "start": 0.0, "end": 1.0}])
        root, _ = self._run(edl)
        v_clip = root.find("./sequence/media/video/track/clipitem")
        a_clip = root.find("./sequence/media/audio/track/clipitem")
        self.assertEqual(v_clip.find("start").text, a_clip.find("start").text)
        self.assertEqual(v_clip.find("end").text, a_clip.find("end").text)
        # The audio clipitem must NOT re-emit the full <file> — it was already
        # written once, on the video clipitem just above it.
        self.assertEqual(len(list(a_clip.find("file"))), 0)

    def test_no_ranges_produces_an_empty_valid_sequence(self):
        root, notes = self._run(self._edl([]))
        self.assertEqual(root.find("./sequence/duration").text, "0")


class BuildTimelineNotesTests(unittest.TestCase):
    def _run(self, edl, lut_dir=None):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            patches = [
                patch.object(export_timeline.subprocess, "run", return_value=_ffprobe_rate_result()),
                patch.object(export_timeline, "probe_dimensions", return_value=(1920, 1080)),
            ]
            if lut_dir is not None:
                import render as render_module  # the real sibling module
                patches.append(patch.object(render_module, "LUT_DIR", lut_dir))
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                return export_timeline.build_timeline(edl, edit_dir)

    def test_grade_is_noted_not_applied(self):
        edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}],
               "grade": "warm_cinematic"}
        root, notes = self._run(edl)
        self.assertEqual(notes["grade"], "warm_cinematic")
        comments = root.find("./sequence/comments").text
        self.assertIn("warm_cinematic", comments)
        self.assertIn("not applied", comments)
        # No filter/effect element was fabricated anywhere in the document.
        self.assertIsNone(root.find(".//filter"))

    def test_lut_resolves_to_a_real_path_when_found(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            lut_dir = Path(lut_dir_str)
            (lut_dir / "warm.cube").write_text("LUT_3D_SIZE 2\n")
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}],
                   "lut": "warm"}
            root, notes = self._run(edl, lut_dir=lut_dir)
        self.assertEqual(notes["lut_path"], lut_dir / "warm.cube")
        self.assertIn(str(lut_dir / "warm.cube"), root.find("./sequence/comments").text)

    def test_unresolvable_lut_name_falls_back_to_the_bare_name_in_notes(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}],
                   "lut": "nonexistent_lut_name"}
            root, notes = self._run(edl, lut_dir=Path(lut_dir_str))
        self.assertIsNone(notes["lut_path"])
        self.assertIn("nonexistent_lut_name", root.find("./sequence/comments").text)

    def test_subtitles_are_noted_as_a_sidecar_not_embedded(self):
        edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}],
               "subtitles": "edit/master.srt"}
        root, notes = self._run(edl)
        self.assertEqual(notes["subtitles"], "edit/master.srt")
        self.assertIsNone(root.find(".//caption"))  # no fabricated caption element
        comments = root.find("./sequence/comments").text
        self.assertIn("master.srt", comments)

    def test_no_grade_lut_or_subtitles_means_no_comments_element_at_all(self):
        edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
        root, notes = self._run(edl)
        self.assertIsNone(root.find("./sequence/comments"))


class BuildTimelineOverlayPathTests(unittest.TestCase):
    """Regression coverage for the real bug this feature surfaced: an
    overlay's EDL "file" field written as "edit/animations/.../render.mp4"
    must resolve correctly even though edit_dir already ends in "edit" —
    plain resolve_path() alone would double that up and silently miss it."""

    def test_overlay_path_written_with_edit_prefix_still_resolves(self):
        with tempfile.TemporaryDirectory() as d:
            root_dir = Path(d)
            edit_dir = root_dir / "edit"
            (edit_dir / "animations" / "slot_1").mkdir(parents=True)
            overlay_file = edit_dir / "animations" / "slot_1" / "render.mp4"
            overlay_file.write_bytes(b"\x00")
            (edit_dir / "A.mp4").write_bytes(b"\x00")

            edl = {
                "sources": {"A": "A.mp4"},
                "ranges": [{"source": "A", "start": 0.0, "end": 2.0}],
                "overlays": [{
                    "file": "edit/animations/slot_1/render.mp4",  # the SKILL.md-shaped path
                    "start_in_output": 0.5, "duration": 1.0,
                }],
            }
            import os
            cwd = os.getcwd()
            try:
                os.chdir(root_dir)  # matches real usage: agent runs from <videos_dir>
                with (
                    patch.object(export_timeline.subprocess, "run", return_value=_ffprobe_rate_result()),
                    patch.object(export_timeline, "probe_dimensions", return_value=(1920, 1080)),
                ):
                    root, notes = export_timeline.build_timeline(edl, edit_dir)
            finally:
                os.chdir(cwd)

        self.assertEqual(notes["sources_missing_media"], [])
        overlay_clip = root.find("./sequence/media/video/track[2]/clipitem")
        self.assertIsNotNone(overlay_clip)
        pathurl = overlay_clip.find("file/pathurl").text
        self.assertIn("render.mp4", pathurl)
        self.assertNotIn("edit/edit", pathurl.replace("\\", "/"))


class WriteXmemlTests(unittest.TestCase):
    def test_output_is_well_formed_and_has_doctype(self):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            with (
                patch.object(export_timeline.subprocess, "run", return_value=_ffprobe_rate_result()),
                patch.object(export_timeline, "probe_dimensions", return_value=(1920, 1080)),
            ):
                root, _ = export_timeline.build_timeline(
                    {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]},
                    edit_dir,
                )
            out = edit_dir / "timeline.xml"
            export_timeline.write_xmeml(root, out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE xmeml>", text)
            self.assertIn('<xmeml version="5">', text)
            # Re-parse to confirm strict well-formedness, not just string presence.
            ET.fromstring(text.split("\n", 2)[-1])


if __name__ == "__main__":
    unittest.main()
