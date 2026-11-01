"""Tests for the bug-report fixes in render.py:

  - clamp_range_to_source / clamp_edl_ranges  (range exceeds source duration)
  - extract_segment's frame-exact quantization  (A/V drift)
  - per-range / per-source LUT resolution       (LUT was global-only)
  - --res/--width                               (resolution was hardcoded)
  - natural subtitle style + text fixes         (no sidecar/readable option)
  - measure_true_peak / verify_and_limit_true_peak  (loudnorm TP overshoot)

Real end-to-end verification (actual ffmpeg, actual mixed-rate 47.952/30fps
sources, actual loud audio pushed through the real limiter, actual per-source
LUT producing visibly different frames) was done manually; see the session
notes. These are the fast, offline, mocked-subprocess unit tests matching
the rest of this suite's style.
"""

import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "render.py"
SPEC = importlib.util.spec_from_file_location("video_use_render", MODULE_PATH)
assert SPEC and SPEC.loader
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


def w(text, start, end):
    return {"type": "word", "text": text, "start": start, "end": end}


def _duration_result(seconds: float) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=f"{seconds}\n", stderr="")


def _rate_result(rate: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"streams": [{"avg_frame_rate": rate, "r_frame_rate": rate}]}), stderr="",
    )


# -------- clamp_range_to_source / clamp_edl_ranges (bug #2) ----------------


class ClampRangeToSourceTests(unittest.TestCase):
    def test_range_within_duration_is_untouched(self):
        with patch.object(render, "probe_duration", return_value=10.0):
            start, end = render.clamp_range_to_source(Path("a.mp4"), 1.0, 5.0, fps="24")
        self.assertEqual((start, end), (1.0, 5.0))

    def test_range_past_duration_is_clamped_to_two_frames_before_the_end(self):
        with patch.object(render, "probe_duration", return_value=10.0):
            start, end = render.clamp_range_to_source(Path("a.mp4"), 8.0, 10.5, fps="24")
        self.assertEqual(start, 8.0)
        self.assertAlmostEqual(end, 10.0 - 2 / 24, places=6)

    def test_undetectable_duration_leaves_range_unchanged(self):
        with patch.object(render, "probe_duration", return_value=None):
            start, end = render.clamp_range_to_source(Path("a.mp4"), 1.0, 999.0, fps="24")
        self.assertEqual((start, end), (1.0, 999.0))

    def test_omitted_fps_probes_the_sources_own_rate(self):
        # Mixed-rate EDL: "2 frames of margin" means something different per
        # source file, so the function must probe when not told explicitly.
        with (
            patch.object(render, "probe_duration", return_value=10.0),
            patch.object(render, "probe_source_fps", return_value="48000/1001"),
        ):
            start, end = render.clamp_range_to_source(Path("dji.mp4"), 9.9, 10.2)
        self.assertAlmostEqual(end, 10.0 - 2 / (48000 / 1001), places=6)

    def test_clamp_never_goes_below_start(self):
        with patch.object(render, "probe_duration", return_value=1.0):
            start, end = render.clamp_range_to_source(Path("a.mp4"), 5.0, 20.0, fps="24")
        self.assertGreaterEqual(end, start)


class ClampEdlRangesTests(unittest.TestCase):
    def test_mutates_ranges_in_place_and_returns_clamp_count(self):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            edl = {
                "sources": {"A": "A.mp4"},
                "ranges": [
                    {"source": "A", "start": 0.0, "end": 5.0},
                    {"source": "A", "start": 8.0, "end": 10.5},  # exceeds
                ],
            }
            with (
                patch.object(render, "probe_duration", return_value=10.0),
                patch.object(render, "probe_source_fps", return_value="24/1"),
            ):
                count = render.clamp_edl_ranges(edl, edit_dir)
        self.assertEqual(count, 1)
        self.assertEqual(edl["ranges"][0]["end"], 5.0)          # untouched
        self.assertLess(edl["ranges"][1]["end"], 10.5)          # clamped

    def test_idempotent_second_call_clamps_nothing_more(self):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "A.mp4").write_bytes(b"\x00")
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 8.0, "end": 10.5}]}
            with (
                patch.object(render, "probe_duration", return_value=10.0),
                patch.object(render, "probe_source_fps", return_value="24/1"),
            ):
                first = render.clamp_edl_ranges(edl, edit_dir)
                second = render.clamp_edl_ranges(edl, edit_dir)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)


# -------- Resolution defaults to the source's own max, not a hardcoded value -


def _dims_result(width: int, height: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"streams": [{"width": width, "height": height}]}), stderr="",
    )


class ProbeDisplayDimensionsTests(unittest.TestCase):
    def test_plain_dimensions_no_rotation(self):
        with patch.object(render.subprocess, "run", return_value=_dims_result(1920, 1080)):
            self.assertEqual(render.probe_display_dimensions(Path("a.mp4")), (1920, 1080))

    def test_rotation_swaps_axes(self):
        result = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"streams": [
                {"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]}
            ]}), stderr="",
        )
        with patch.object(render.subprocess, "run", return_value=result):
            self.assertEqual(render.probe_display_dimensions(Path("a.mp4")), (1080, 1920))

    def test_unprobeable_returns_none(self):
        bad = subprocess.CompletedProcess([], 0, stdout="not json", stderr="")
        with patch.object(render.subprocess, "run", return_value=bad):
            self.assertIsNone(render.probe_display_dimensions(Path("a.mp4")))


class ProbeSourceLongEdgeTests(unittest.TestCase):
    def test_landscape_long_edge_is_width(self):
        with patch.object(render.subprocess, "run", return_value=_dims_result(3840, 2160)):
            self.assertEqual(render.probe_source_long_edge(Path("a.mp4")), 3840)

    def test_portrait_long_edge_is_height(self):
        with patch.object(render.subprocess, "run", return_value=_dims_result(1080, 1920)):
            self.assertEqual(render.probe_source_long_edge(Path("a.mp4")), 1920)

    def test_unprobeable_returns_none(self):
        bad = subprocess.CompletedProcess([], 0, stdout="not json", stderr="")
        with patch.object(render.subprocess, "run", return_value=bad):
            self.assertIsNone(render.probe_source_long_edge(Path("a.mp4")))


class ProbeMaxResolutionTests(unittest.TestCase):
    def test_picks_the_highest_resolution_source_actually_used(self):
        def fake_run(cmd, **kwargs):
            path = str(cmd[-1])
            if "4k" in path:
                return _dims_result(3840, 2160)
            return _dims_result(1920, 1080)

        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "4k.mp4").write_bytes(b"\x00")
            (edit_dir / "hd.mp4").write_bytes(b"\x00")
            edl = {
                "sources": {"A": "4k.mp4", "B": "hd.mp4"},
                "ranges": [
                    {"source": "A", "start": 0.0, "end": 1.0},
                    {"source": "B", "start": 0.0, "end": 1.0},
                ],
            }
            with patch.object(render.subprocess, "run", side_effect=fake_run):
                edge, per_source = render.probe_max_resolution(edl, edit_dir)
        self.assertEqual(edge, 3840)
        self.assertEqual(per_source, {"A": 3840, "B": 1920})

    def test_ignores_sources_not_actually_used_in_ranges(self):
        # A 4K source sits in "sources" but no range references it — must not
        # pull the default up to a resolution nothing in the cut actually uses.
        def fake_run(cmd, **kwargs):
            path = str(cmd[-1])
            if "unused_4k" in path:
                return _dims_result(3840, 2160)
            return _dims_result(1280, 720)

        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "unused_4k.mp4").write_bytes(b"\x00")
            (edit_dir / "used.mp4").write_bytes(b"\x00")
            edl = {
                "sources": {"A": "unused_4k.mp4", "B": "used.mp4"},
                "ranges": [{"source": "B", "start": 0.0, "end": 1.0}],
            }
            with patch.object(render.subprocess, "run", side_effect=fake_run):
                edge, per_source = render.probe_max_resolution(edl, edit_dir)
        self.assertEqual(edge, 1280)  # long edge of a 1280x720 (landscape) source
        self.assertEqual(per_source, {"B": 1280})

    def test_empty_ranges_falls_back_to_default_width(self):
        edl = {"sources": {}, "ranges": []}
        with tempfile.TemporaryDirectory() as d:
            edge, per_source = render.probe_max_resolution(edl, Path(d))
        self.assertEqual(edge, render.DEFAULT_WIDTH)
        self.assertEqual(per_source, {})

    def test_unprobeable_sources_fall_back_to_default_width(self):
        bad = subprocess.CompletedProcess([], 0, stdout="not json", stderr="")
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            (edit_dir / "a.mp4").write_bytes(b"\x00")
            edl = {"sources": {"A": "a.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            with patch.object(render.subprocess, "run", return_value=bad):
                edge, per_source = render.probe_max_resolution(edl, edit_dir)
        self.assertEqual(edge, render.DEFAULT_WIDTH)
        self.assertEqual(per_source, {})


# -------- extract_segment frame-exact quantization (bug #3, #6) ------------


class ExtractSegmentQuantizationTests(unittest.TestCase):
    def _run(self, duration, rate="24/1", **kwargs):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.mp4"
            src.write_bytes(b"\x00")
            out = Path(d) / "out.mp4"
            captured = {}

            def fake_run(cmd, check, stdout, stderr):
                captured["cmd"] = cmd

            with (
                patch.object(render, "is_portrait_source", return_value=False),
                patch.object(render, "is_hdr_source", return_value=False),
                patch.object(render.subprocess, "run", side_effect=fake_run),
            ):
                render.extract_segment(src, 0.0, duration, "", out, rate=rate, **kwargs)
            return captured["cmd"]

    def test_video_frame_count_is_a_whole_number(self):
        # 1.013s at 24fps = 24.312 frames -> must round to a whole number,
        # not get truncated/ceil'd implicitly by ffmpeg's own -t handling.
        cmd = self._run(duration=1.013)
        nfr = int(cmd[cmd.index("-frames:v") + 1])
        self.assertEqual(nfr, round(1.013 * 24))

    def test_audio_is_forced_to_the_same_quantized_duration(self):
        cmd = self._run(duration=1.013)
        af = cmd[cmd.index("-af") + 1]
        nfr = round(1.013 * 24)
        quantized = nfr / 24
        self.assertIn(f"atrim=0:{quantized:.6f}", af)

    def test_input_read_overshoots_the_quantized_duration(self):
        # -t on the INPUT read must exceed the quantized duration (so
        # -frames:v has enough decoded source to reach that count), not
        # equal it — equal would risk an early EOF truncating the video short
        # again, the exact failure mode this fix exists to prevent.
        cmd = self._run(duration=1.013)
        input_t = float(cmd[cmd.index("-t") + 1])
        nfr = round(1.013 * 24)
        self.assertGreater(input_t, nfr / 24)

    def test_explicit_stream_mapping_drops_stray_streams(self):
        cmd = self._run(duration=1.0)
        self.assertIn("0:v:0", cmd)
        self.assertIn("0:a:0", cmd)
        self.assertIn("-write_tmcd", cmd)
        self.assertIn("-map_metadata", cmd)

    def test_color_tags_are_explicit_bt709(self):
        cmd = self._run(duration=1.0)
        self.assertIn("-color_primaries", cmd)
        self.assertIn("bt709", cmd)

    def test_lanczos_scaling_flag_present(self):
        cmd = self._run(duration=1.0)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("flags=lanczos", vf)

    def test_width_parameter_controls_scale_target(self):
        cmd = self._run(duration=1.0, width=3840)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("3840", vf)

    def test_draft_mode_ignores_width_uses_fixed_1280(self):
        cmd = self._run(duration=1.0, width=3840, draft=True)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("1280", vf)
        self.assertNotIn("3840", vf)


# -------- Per-range / per-source LUT resolution (bug #4) -------------------


class PerSegmentLutTests(unittest.TestCase):
    def _run_extract_all(self, edl, lut_dir):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d)
            for name in edl["sources"].values():
                (edit_dir / name).write_bytes(b"\x00")
            calls = []

            def fake_extract(src, start, duration, grade, out, **kw):
                calls.append({"src": str(src), "lut_filter": kw.get("lut_filter", "")})

            with (
                patch.object(render, "LUT_DIR", lut_dir),
                patch.object(render, "extract_segment", side_effect=fake_extract),
                patch.object(render, "probe_source_fps", return_value="24/1"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                render.extract_all_segments(edl, edit_dir, preview=False)
            return calls

    def test_range_level_lut_overrides_global(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            lut_dir = Path(lut_dir_str)
            (lut_dir / "global.cube").write_text("LUT_3D_SIZE 2\n")
            (lut_dir / "special.cube").write_text("LUT_3D_SIZE 2\n")
            edl = {
                "sources": {"A": "A.mp4"},
                "ranges": [{"source": "A", "start": 0.0, "end": 1.0, "lut": "special"}],
                "lut": "global",
            }
            calls = self._run_extract_all(edl, lut_dir)
        self.assertIn("special.cube", calls[0]["lut_filter"].replace("\\", "/"))

    def test_per_source_luts_map_used_when_no_range_override(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            lut_dir = Path(lut_dir_str)
            (lut_dir / "camera_look.cube").write_text("LUT_3D_SIZE 2\n")
            edl = {
                "sources": {"DJI": "DJI.mp4"},
                "ranges": [{"source": "DJI", "start": 0.0, "end": 1.0}],
                "luts": {"DJI": "camera_look"},
            }
            calls = self._run_extract_all(edl, lut_dir)
        self.assertIn("camera_look.cube", calls[0]["lut_filter"].replace("\\", "/"))

    def test_different_sources_get_different_luts_in_one_render(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            lut_dir = Path(lut_dir_str)
            (lut_dir / "a_look.cube").write_text("LUT_3D_SIZE 2\n")
            (lut_dir / "b_look.cube").write_text("LUT_3D_SIZE 2\n")
            edl = {
                "sources": {"A": "A.mp4", "B": "B.mp4"},
                "ranges": [
                    {"source": "A", "start": 0.0, "end": 1.0},
                    {"source": "B", "start": 0.0, "end": 1.0},
                ],
                "luts": {"A": "a_look", "B": "b_look"},
            }
            calls = self._run_extract_all(edl, lut_dir)
        self.assertIn("a_look.cube", calls[0]["lut_filter"].replace("\\", "/"))
        self.assertIn("b_look.cube", calls[1]["lut_filter"].replace("\\", "/"))

    def test_no_lut_anywhere_is_a_noop(self):
        with tempfile.TemporaryDirectory() as lut_dir_str:
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            calls = self._run_extract_all(edl, Path(lut_dir_str))
        self.assertEqual(calls[0]["lut_filter"], "")


# -------- natural subtitle style / text fixes (bug #8) ---------------------


class ChunkWordsNaturalTests(unittest.TestCase):
    def test_breaks_on_sentence_punctuation(self):
        words = [w("Hi", 0, 0.3), w("there.", 0.3, 0.6), w("Bye", 0.7, 0.9)]
        chunks = render.chunk_words_natural(words)
        texts = [" ".join(x["text"] for x in c) for c in chunks]
        self.assertEqual(texts, ["Hi there.", "Bye"])

    def test_breaks_on_pause(self):
        words = [w("one", 0, 0.3), w("two", 1.0, 1.3)]  # 0.7s gap >= 0.45 default
        chunks = render.chunk_words_natural(words)
        self.assertEqual(len(chunks), 2)

    def test_respects_char_limit(self):
        words = [w("a" * 20, 0, 1), w("b" * 20, 1.01, 2), w("c", 2.02, 2.1)]
        chunks = render.chunk_words_natural(words, max_chars=42)
        # First two words alone are already 41 chars + 1 space = 41, adding a
        # 3rd word must not be allowed to exceed 42.
        self.assertEqual(len(chunks[0]), 2)

    def test_respects_word_cap(self):
        words = [w(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(20)]
        chunks = render.chunk_words_natural(words, max_words=8, pause_s=100)
        self.assertTrue(all(len(c) <= 8 for c in chunks))


class LoadTextFixesTests(unittest.TestCase):
    def test_parses_tab_separated_pairs_skips_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixes.txt"
            p.write_text("# header\n\nfoo\tbar\n  baz \t qux  \n# trailing\n")
            fixes = render.load_text_fixes(p)
        self.assertEqual(fixes, [("foo", "bar"), ("baz", "qux")])

    def test_none_path_is_empty_list(self):
        self.assertEqual(render.load_text_fixes(None), [])

    def test_missing_file_is_a_clean_error(self):
        with self.assertRaises(SystemExit):
            render.load_text_fixes(Path("/nope/nope.txt"))

    def test_line_without_tab_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixes.txt"
            p.write_text("no tab here\n")
            with self.assertRaises(SystemExit):
                render.load_text_fixes(p)

    def test_apply_text_fixes_runs_regex_substitutions_in_order(self):
        fixes = [(r"\bfoo\b", "bar"), (r"\bbar\b", "baz")]
        self.assertEqual(render._apply_text_fixes("foo", fixes), "baz")


class BuildMasterSrtNaturalStyleTests(unittest.TestCase):
    def test_natural_style_is_sentence_case_not_upper(self):
        words = [w("hello", 0.0, 0.3), w("world.", 0.32, 0.6), w("next", 0.8, 1.0)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            out = edit / "master.srt"
            render.build_master_srt(edl, edit, out, style="natural")
            text = out.read_text()
        self.assertIn("Hello world.", text)
        self.assertNotIn("HELLO", text)

    def test_bold_overlay_style_is_unchanged_default(self):
        words = [w("hello", 0.0, 0.3), w("world.", 0.32, 0.6)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            out = edit / "master.srt"
            render.build_master_srt(edl, edit, out)  # no style kwarg -> default
            self.assertIn("HELLO", out.read_text())

    def test_natural_cues_never_overlap(self):
        words = [w("a", 0.0, 0.2), w("b", 0.25, 0.4), w("c", 2.0, 2.2), w("d", 2.25, 2.4)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 3.0}]}
            out = edit / "master.srt"
            render.build_master_srt(edl, edit, out, style="natural")
            blocks = [b for b in out.read_text().strip().split("\n\n") if b]
        times = []
        for b in blocks:
            line = b.splitlines()[1]
            a_str, _, b_str = line.partition(" --> ")
            times.append((a_str, b_str))
        # Each cue's end timestamp string must sort before the next cue's start.
        for (_, end), (start, _) in zip(times, times[1:]):
            self.assertLessEqual(end, start)

    def test_text_fixes_applied_in_natural_style(self):
        words = [w("that's", 0.0, 0.3), w("our", 0.32, 0.5), w("effect.", 0.52, 0.9)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            out = edit / "master.srt"
            fixes = [(r"that's our effect\.", "that's our copy.")]
            render.build_master_srt(edl, edit, out, style="natural", text_fixes=fixes)
            text = out.read_text()
        self.assertIn("copy", text)
        self.assertNotIn("effect", text)


# -------- True-peak verification / limiter fallback (bug #7) ---------------


class MeasureTruePeakTests(unittest.TestCase):
    def test_parses_the_last_peak_line_in_ebur128_output(self):
        stderr = (
            "  Integrated loudness:\n    I: -20.0 LUFS\n"
            "  True peak:\n    Peak:        -3.2 dBFS\n"
        )
        with patch.object(
            render.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=stderr),
        ):
            tp = render.measure_true_peak(Path("x.mp4"))
        self.assertEqual(tp, -3.2)

    def test_no_match_returns_none(self):
        with patch.object(
            render.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="nothing here"),
        ):
            self.assertIsNone(render.measure_true_peak(Path("x.mp4")))


class VerifyAndLimitTruePeakTests(unittest.TestCase):
    def test_compliant_peak_does_nothing(self):
        with (
            patch.object(render, "measure_true_peak", return_value=-2.0),
            patch.object(render.subprocess, "run") as run_mock,
        ):
            render.verify_and_limit_true_peak(Path("x.mp4"))
        run_mock.assert_not_called()

    def test_overshooting_peak_triggers_a_limiter_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.mp4"
            path.write_bytes(b"\x00")
            captured = {}

            def fake_run(cmd, check, stdout, stderr):
                captured["cmd"] = cmd
                # The limiter pass writes to `path` (renamed-back target);
                # create it so the post-limit re-measure's Path exists.
                Path(cmd[-1]).write_bytes(b"\x00")

            with (
                patch.object(render, "measure_true_peak", side_effect=[0.6, -1.9]),
                patch.object(render.subprocess, "run", side_effect=fake_run),
            ):
                render.verify_and_limit_true_peak(path)
            self.assertIn("alimiter", captured["cmd"][captured["cmd"].index("-af") + 1])

    def test_unmeasurable_peak_does_nothing(self):
        with (
            patch.object(render, "measure_true_peak", return_value=None),
            patch.object(render.subprocess, "run") as run_mock,
        ):
            render.verify_and_limit_true_peak(Path("x.mp4"))
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
