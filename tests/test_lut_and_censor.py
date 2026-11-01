"""Tests for --lut and --censor-profanity.

Fast and offline, matching the rest of this suite — real ffmpeg/numpy
verification (actual pixel diffs, actual FFT tone detection on a rendered
file) was done manually against the real DJI LUT already sitting in lut/ and
is not repeated here; these tests check the command/string construction that
feeds ffmpeg, mocking subprocess the same way test_render_fps.py does.
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# helpers/ must be on sys.path BEFORE render.py is exec'd, not after: render.py
# does `from profanity_filter import (...)` at its own module level (wrapped in
# a try/except whose fallback is an always-False/no-op stub, same pattern as
# its grade.py import). Insert this first so render's *real* implementation
# gets bound — otherwise every test below would silently exercise the stub
# instead of helpers/profanity_filter.py, and still pass its own asserts
# against a correctly-imported `profanity` while render.find_censor_ranges /
# render.build_master_srt quietly used the no-op one. (This is exactly what
# happened during development here — worth keeping the order correct.)
sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))
import profanity_filter as profanity  # noqa: E402

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "render.py"
SPEC = importlib.util.spec_from_file_location("video_use_render", MODULE_PATH)
assert SPEC and SPEC.loader
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


def w(text, start, end, type_="word"):
    return {"type": type_, "text": text, "start": start, "end": end}


# -------- LUT resolution ------------------------------------------------


class FindLutFileTests(unittest.TestCase):
    def test_bare_name_resolves_against_lut_dir(self):
        with tempfile.TemporaryDirectory() as d:
            lut_dir = Path(d)
            (lut_dir / "warm.cube").write_text("LUT_3D_SIZE 2\n")
            with patch.object(render, "LUT_DIR", lut_dir):
                self.assertEqual(render.find_lut_file("warm"), lut_dir / "warm.cube")

    def test_name_containing_its_own_dots_still_resolves(self):
        # Regression: a real camera/colorist LUT name routinely contains dots
        # ("...to Rec.709 V1.cube"), which made Path(name).suffix misread the
        # bare search term as "already extensioned" and skip the .cube guess.
        with tempfile.TemporaryDirectory() as d:
            lut_dir = Path(d)
            name = "DJI OSMO Action 4 D-Log M to Rec.709 V1"
            (lut_dir / f"{name}.cube").write_text("LUT_3D_SIZE 2\n")
            with patch.object(render, "LUT_DIR", lut_dir):
                found = render.find_lut_file(name)
                self.assertEqual(found, lut_dir / f"{name}.cube")

    def test_name_with_extension_already_given(self):
        with tempfile.TemporaryDirectory() as d:
            lut_dir = Path(d)
            (lut_dir / "warm.cube").write_text("LUT_3D_SIZE 2\n")
            with patch.object(render, "LUT_DIR", lut_dir):
                self.assertEqual(render.find_lut_file("warm.cube"), lut_dir / "warm.cube")

    def test_absolute_path_bypasses_lut_dir(self):
        with tempfile.TemporaryDirectory() as d:
            somewhere = Path(d) / "elsewhere.cube"
            somewhere.write_text("LUT_3D_SIZE 2\n")
            with tempfile.TemporaryDirectory() as empty_lut_dir:
                with patch.object(render, "LUT_DIR", Path(empty_lut_dir)):
                    self.assertEqual(render.find_lut_file(str(somewhere)), somewhere)

    def test_path_relative_to_edit_dir(self):
        with tempfile.TemporaryDirectory() as d:
            edit_dir = Path(d) / "edit"
            edit_dir.mkdir()
            (edit_dir / "local.cube").write_text("LUT_3D_SIZE 2\n")
            with tempfile.TemporaryDirectory() as empty_lut_dir:
                with patch.object(render, "LUT_DIR", Path(empty_lut_dir)):
                    found = render.find_lut_file("local.cube", edit_dir=edit_dir)
                    self.assertEqual(found, (edit_dir / "local.cube").resolve())

    def test_missing_lut_lists_whats_available(self):
        with tempfile.TemporaryDirectory() as d:
            lut_dir = Path(d)
            (lut_dir / "kodak_2383.cube").write_text("LUT_3D_SIZE 2\n")
            with patch.object(render, "LUT_DIR", lut_dir):
                with self.assertRaises(SystemExit) as e:
                    render.find_lut_file("nope")
                self.assertIn("kodak_2383", str(e.exception))

    def test_missing_lut_with_empty_folder_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(render, "LUT_DIR", Path(d)):
                with self.assertRaises(SystemExit) as e:
                    render.find_lut_file("nope")
                self.assertIn("empty", str(e.exception))


class LutFilterStringTests(unittest.TestCase):
    def test_produces_lut3d_filter(self):
        with tempfile.TemporaryDirectory() as d:
            lut = Path(d) / "warm.cube"
            lut.write_text("LUT_3D_SIZE 2\n")
            s = render.lut_filter_string(lut)
            self.assertTrue(s.startswith("lut3d=file='"))
            self.assertTrue(s.endswith("'"))

    def test_backslashes_become_forward_slashes(self):
        # Same reasoning as the proven subtitles-path escaping: a backslash is
        # a filtergraph escape char, so a native Windows path breaks lut3d=
        # the same way it breaks subtitles= unless normalized first. The one
        # expected backslash is the drive-colon escape itself (`C\:/...`) —
        # strip exactly that escape sequence and nothing else should remain.
        with tempfile.TemporaryDirectory() as d:
            lut = Path(d) / "warm.cube"
            lut.write_text("LUT_3D_SIZE 2\n")
            s = render.lut_filter_string(lut)
            inner = s[len("lut3d=file='"):-1]
            self.assertNotIn("\\", inner.replace(r"\:", ""))
            # And no raw path separators survived as backslashes either.
            self.assertNotRegex(inner, r"\\[^:]")

    def test_drive_colon_is_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            lut = Path(d) / "warm.cube"
            lut.write_text("LUT_3D_SIZE 2\n")
            s = render.lut_filter_string(lut)
            resolved = str(lut.resolve())
            if ":" in resolved:  # only meaningful on a platform with drive letters
                self.assertIn(r"\:", s)


class ResolveLutFilterTests(unittest.TestCase):
    def test_none_or_empty_is_a_noop(self):
        self.assertEqual(render.resolve_lut_filter(None), "")
        self.assertEqual(render.resolve_lut_filter(""), "")

    def test_resolves_through_to_a_filter_string(self):
        with tempfile.TemporaryDirectory() as d:
            lut_dir = Path(d)
            (lut_dir / "warm.cube").write_text("LUT_3D_SIZE 2\n")
            with patch.object(render, "LUT_DIR", lut_dir):
                self.assertIn("lut3d=file=", render.resolve_lut_filter("warm"))


class ExtractSegmentLutOrderingTests(unittest.TestCase):
    def test_lut_runs_after_grade_in_the_vf_chain(self):
        # Corrective grade first, creative look last — matches how a colorist
        # layers a corrective node under a creative one.
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.mp4"
            src.write_bytes(b"\x00")
            out = Path(d) / "out.mp4"
            captured = {}

            def fake_run(cmd, check, stdout, stderr):
                captured["cmd"] = cmd
                return None

            with (
                patch.object(render, "is_portrait_source", return_value=False),
                patch.object(render, "is_hdr_source", return_value=False),
                patch.object(render.subprocess, "run", side_effect=fake_run),
            ):
                render.extract_segment(
                    src, 0.0, 1.0, "eq=contrast=1.1", out, rate="24",
                    lut_filter="lut3d=file='warm.cube'",
                )
            vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
            self.assertLess(vf.index("eq=contrast"), vf.index("lut3d="))


# -------- Profanity module -----------------------------------------------


class LoadProfanityStemsTests(unittest.TestCase):
    def test_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "list.txt"
            p.write_text("# header\n\nfuck\n  shit  \n# trailing comment\nass # inline too\n")
            self.assertEqual(profanity.load_profanity_stems(p), ["fuck", "shit", "ass"])

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(profanity.load_profanity_stems(Path("/nope/nope.txt")), [])


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.matcher = profanity.build_profanity_matcher(["fuck", "shit", "ass"])

    def test_empty_stems_gives_no_matcher(self):
        self.assertIsNone(profanity.build_profanity_matcher([]))

    def test_exact_stem_matches(self):
        for word in ("fuck", "Fuck", "FUCK", "shit", "ass"):
            self.assertTrue(profanity.is_profane(word, self.matcher), word)

    def test_common_inflections_match(self):
        for word in ("fucking", "fucked", "fucker", "fuckers", "fucks"):
            self.assertTrue(profanity.is_profane(word, self.matcher), word)

    def test_punctuation_is_stripped_before_matching(self):
        for word in ("fuck,", "Fuck!", "“shit”", "(ass)"):
            self.assertTrue(profanity.is_profane(word, self.matcher), word)

    def test_innocent_words_are_not_false_positives(self):
        # Stem + inflection matching, not substring: these all contain a stem
        # as a literal substring but are not that stem plus a known suffix.
        for word in ("assume", "class", "passenger", "assassin", "bypass"):
            self.assertFalse(profanity.is_profane(word, self.matcher), word)

    def test_none_matcher_matches_nothing(self):
        self.assertFalse(profanity.is_profane("fuck", None))

    def test_empty_text_is_not_profane(self):
        self.assertFalse(profanity.is_profane("", self.matcher))


class CensorWordTests(unittest.TestCase):
    def setUp(self):
        self.matcher = profanity.build_profanity_matcher(["fuck", "shit"])

    def test_replaces_core_keeps_punctuation(self):
        self.assertEqual(profanity.censor_word("fuck,", self.matcher), "***,")
        self.assertEqual(profanity.censor_word("Fucking!", self.matcher), "***!")
        self.assertEqual(profanity.censor_word("shit.", self.matcher), "***.")

    def test_non_matching_word_is_untouched(self):
        self.assertEqual(profanity.censor_word("duck", self.matcher), "duck")
        self.assertEqual(profanity.censor_word("shift", self.matcher), "shift")

    def test_none_matcher_is_a_noop(self):
        self.assertEqual(profanity.censor_word("fuck", None), "fuck")


# -------- find_censor_ranges / build_censor_enable_expr -------------------


class FindCensorRangesTests(unittest.TestCase):
    def setUp(self):
        self.matcher = profanity.build_profanity_matcher(["shit"])

    def test_output_timeline_math_across_two_segments(self):
        words_a = [w("hello", 0.0, 0.5), w("shit", 1.0, 1.3), w("there", 1.4, 1.6)]
        words_b = [w("another", 0.0, 0.4), w("shit", 0.5, 0.8)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words_a}))
            (edit / "transcripts" / "B.json").write_text(json.dumps({"words": words_b}))
            edl = {
                "sources": {"A": "A.mp4", "B": "B.mp4"},
                "ranges": [
                    {"source": "A", "start": 0.0, "end": 2.0},   # output 0.0-2.0
                    {"source": "B", "start": 0.0, "end": 1.0},   # output 2.0-3.0
                ],
            }
            ranges = render.find_censor_ranges(edl, edit, self.matcher, pad=0.0)
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0][0], 1.0)   # A's "shit": no offset yet
        self.assertAlmostEqual(ranges[0][1], 1.3)
        self.assertAlmostEqual(ranges[1][0], 2.5)   # B's "shit": +2.0 seg_offset
        self.assertAlmostEqual(ranges[1][1], 2.8)

    def test_padding_is_applied_and_clamped_at_zero(self):
        words = [w("shit", 0.02, 0.10)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            ranges = render.find_censor_ranges(edl, edit, self.matcher, pad=0.06)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0][0], 0.0)  # 0.02 - 0.06 would go negative
        self.assertAlmostEqual(ranges[0][1], 0.16)

    def test_clean_words_are_not_flagged(self):
        words = [w("hello", 0.0, 0.5), w("there", 0.6, 0.9)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            self.assertEqual(render.find_censor_ranges(edl, edit, self.matcher), [])

    def test_missing_transcript_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            self.assertEqual(render.find_censor_ranges(edl, edit, self.matcher), [])


class BuildCensorEnableExprTests(unittest.TestCase):
    def test_empty_ranges_is_always_off(self):
        self.assertEqual(render.build_censor_enable_expr([]), "0")

    def test_single_range(self):
        expr = render.build_censor_enable_expr([(1.0, 2.0, "shit")])
        self.assertEqual(expr, "between(t,1.000,2.000)")

    def test_multiple_ranges_are_summed(self):
        expr = render.build_censor_enable_expr([(1.0, 2.0, "shit"), (5.0, 5.5, "fuck")])
        self.assertEqual(expr, "between(t,1.000,2.000)+between(t,5.000,5.500)")


# -------- build_master_srt + profanity integration -------------------------


class BuildMasterSrtCensorTests(unittest.TestCase):
    def test_flagged_word_becomes_asterisks_in_the_caption(self):
        matcher = profanity.build_profanity_matcher(["shit"])
        words = [w("this", 0.0, 0.3), w("is", 0.32, 0.5), w("shit.", 0.52, 0.9)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            out = edit / "master.srt"
            render.build_master_srt(edl, edit, out, profanity_matcher=matcher)
            text = out.read_text()
        self.assertIn("***", text)
        self.assertNotIn("SHIT", text.upper().replace("***", ""))
        self.assertNotIn("shit", text.lower())

    def test_no_matcher_leaves_captions_unchanged(self):
        words = [w("this", 0.0, 0.3), w("is", 0.32, 0.5), w("shit.", 0.52, 0.9)]
        with tempfile.TemporaryDirectory() as d:
            edit = Path(d)
            (edit / "transcripts").mkdir()
            (edit / "transcripts" / "A.json").write_text(json.dumps({"words": words}))
            edl = {"sources": {"A": "A.mp4"}, "ranges": [{"source": "A", "start": 0.0, "end": 1.0}]}
            out = edit / "master.srt"
            render.build_master_srt(edl, edit, out)  # no profanity_matcher kwarg at all
            self.assertIn("SHIT", out.read_text())


# -------- build_final_composite: video/audio map correctness --------------


class BuildFinalCompositeCensorTests(unittest.TestCase):
    """Mocks subprocess.run and inspects the constructed ffmpeg command —
    exactly the class of bug a real render would surface slowly (a failed
    ffmpeg call after minutes of segment extraction). This is the regression
    test for the "[0:v] used as a bare stream map" bug: censoring alone, with
    no overlays and no subtitles, must map the unfiltered video with the RAW
    stream specifier "0:v", never the bracketed filtergraph-pad form "[0:v]"
    (there is no such pad when no video filter chain exists).
    """

    def _run(self, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.mp4"
            base.write_bytes(b"\x00")
            out = Path(d) / "out.mp4"
            captured = {}

            def fake_run(cmd, check, stdout, stderr):
                captured["cmd"] = cmd
                return None

            with (
                patch.object(render.subprocess, "run", side_effect=fake_run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                render.build_final_composite(base, [], None, out, Path(d), **kwargs)
            return captured["cmd"]

    def test_censor_only_maps_raw_unbracketed_video_stream(self):
        cmd = self._run(censor_ranges=[(1.0, 2.0, "shit")], censor_mode="mute")
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertIn("0:v", maps)
        self.assertNotIn("[0:v]", maps)

    def test_censor_only_output_audio_is_a_filtergraph_pad(self):
        cmd = self._run(censor_ranges=[(1.0, 2.0, "shit")], censor_mode="mute")
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertIn("[outa]", maps)
        self.assertNotIn("0:a", maps)

    def test_mute_mode_has_no_extra_sine_input(self):
        cmd = self._run(censor_ranges=[(1.0, 2.0, "shit")], censor_mode="mute")
        self.assertNotIn("lavfi", cmd)

    def test_bleep_mode_adds_one_sine_input(self):
        cmd = self._run(censor_ranges=[(1.0, 2.0, "shit")], censor_mode="bleep")
        self.assertIn("lavfi", cmd)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("amix=inputs=2", fc)
        self.assertIn("normalize=0", fc)

    def test_censor_forces_audio_reencode_not_copy(self):
        cmd = self._run(censor_ranges=[(1.0, 2.0, "shit")], censor_mode="mute")
        self.assertIn("aac", cmd)
        self.assertNotIn("copy", cmd)

    def test_no_censor_keeps_audio_copy_unchanged(self):
        # Regression: nothing about the censor feature should alter the
        # pre-existing no-overlays/no-subs/no-censor behavior at all — that
        # path is the early-return "just copy" branch, never reaching
        # -filter_complex, so there's no cmd to inspect here beyond the call
        # actually succeeding via the plain copy path.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.mp4"
            base.write_bytes(b"\x00")
            out = Path(d) / "out.mp4"
            with (
                patch.object(render, "run") as run_mock,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                render.build_final_composite(base, [], None, out, Path(d))
            run_mock.assert_called_once()
            self.assertIn("copy", run_mock.call_args.args[0])

    def test_censor_with_subtitles_keeps_both_filter_chains(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.mp4"
            base.write_bytes(b"\x00")
            subs = Path(d) / "master.srt"
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nHI\n\n")
            out = Path(d) / "out.mp4"
            captured = {}

            def fake_run(cmd, check, stdout, stderr):
                captured["cmd"] = cmd

            with (
                patch.object(render.subprocess, "run", side_effect=fake_run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                render.build_final_composite(
                    base, [], subs, out, Path(d),
                    censor_ranges=[(1.0, 2.0, "shit")], censor_mode="mute",
                )
            fc = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
            self.assertIn("subtitles=", fc)
            self.assertIn("volume=enable=", fc)
            maps = [captured["cmd"][i + 1] for i, a in enumerate(captured["cmd"]) if a == "-map"]
            self.assertIn("[outv]", maps)
            self.assertIn("[outa]", maps)


if __name__ == "__main__":
    unittest.main()
