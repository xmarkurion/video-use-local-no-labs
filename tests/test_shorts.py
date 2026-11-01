"""Tests for shorts.py. Real end-to-end generation (actual local transcription,
actual window scoring+selection, actual blur-bg/crop portrait reformatting,
actual caption burn-in, actual loudnorm+limiter) was verified manually against
a real 100s video with real AMI meeting speech and a genuinely varying visual
pattern — see session notes, including a real bug this surfaced (snap_window's
fallback could return a clip a few seconds long against a 20s target, when a
32s gap in real speech meant the naive "nearest word" fallback picked one far
short of the minimum). These are the fast, offline unit tests for the pure
logic: boundary snapping, speech-density scoring, and candidate selection.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

# helpers/ must be on sys.path BEFORE shorts.py is exec'd — it does
# `from render import ...` at its own module level. See
# tests/test_lut_and_censor.py for the ordering bug this avoids: get this
# backwards and every test below silently exercises render.py's no-op
# import-failure stubs instead of the real functions.
HELPERS = Path(__file__).parents[1] / "helpers"
sys.path.insert(0, str(HELPERS))

MODULE_PATH = HELPERS / "shorts.py"
SPEC = importlib.util.spec_from_file_location("video_use_shorts", MODULE_PATH)
assert SPEC and SPEC.loader
shorts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shorts)


def w(text, start, end):
    return {"type": "word", "text": text, "start": start, "end": end}


class SnapWindowTests(unittest.TestCase):
    def test_start_snaps_to_first_word_at_or_after_raw_start(self):
        words = [w("a", 2.0, 2.5), w("b", 5.0, 5.5), w("c", 10.0, 10.5)]
        start, _ = shorts.snap_window(words, raw_start=4.0, raw_end=20.0, target_duration=10.0)
        self.assertEqual(start, 5.0)

    def test_prefers_a_sentence_end_within_range(self):
        words = [
            w("hello", 0.0, 0.5),
            w("world,", 1.0, 1.5),
            w("continuing.", 9.5, 10.0),   # sentence end, close to target
            w("more", 10.2, 10.6),          # non-sentence, also in range
            w("stuff", 15.0, 15.5),
        ]
        start, end = shorts.snap_window(words, raw_start=0.0, raw_end=10.0, target_duration=10.0)
        self.assertEqual(start, 0.0)
        self.assertEqual(end, 10.0)  # the sentence-ending word, not the later non-sentence one

    def test_falls_back_to_nearest_word_end_when_no_sentence_end_in_range(self):
        words = [w("a", 0.0, 0.5), w("b", 8.0, 8.5), w("c", 9.5, 10.0), w("d", 20.0, 20.5)]
        # target 10s, range [6, 16] (factors 0.6-1.6) — "c" at 10.0 is closest
        start, end = shorts.snap_window(words, raw_start=0.0, raw_end=10.0, target_duration=10.0)
        self.assertEqual(end, 10.0)

    def test_never_returns_a_clip_shorter_than_the_minimum_factor(self):
        # Regression: a real 32s gap in speech made the old fallback pick the
        # last word before raw_end, producing a ~3s clip from a 20s target.
        words = [
            w("opening", 5.0, 7.9),             # the only early speech
            w("later", 40.0, 40.3),              # next word, well past the gap
        ]
        start, end = shorts.snap_window(words, raw_start=5.0, raw_end=25.0, target_duration=20.0)
        min_allowed = 20.0 * shorts.MIN_DURATION_FACTOR
        self.assertGreaterEqual(end - start, min_allowed)
        self.assertEqual(end, 40.3)  # the fallback, not the too-short in-range word

    def test_sanity_cap_prevents_an_absurdly_long_clip(self):
        words = [w("opening", 5.0, 5.5), w("much_later", 500.0, 500.5)]
        start, end = shorts.snap_window(words, raw_start=5.0, raw_end=25.0, target_duration=20.0)
        self.assertLessEqual(end - start, 20.0 * 2.0 + 1e-6)

    def test_empty_words_returns_raw_window_unchanged(self):
        self.assertEqual(shorts.snap_window([], 5.0, 25.0, 20.0), (5.0, 25.0))

    def test_no_words_at_all_past_minimum_falls_back_to_last_available(self):
        words = [w("only", 1.0, 1.5)]
        start, end = shorts.snap_window(words, raw_start=0.0, raw_end=2.0, target_duration=20.0)
        self.assertEqual(end, 1.5)  # nothing better exists than the one word there is


class SpeechDensityTests(unittest.TestCase):
    def test_full_bucket_coverage_is_one(self):
        words = [w("a", 0.0, 1.0)]
        density = shorts.compute_speech_density(words, duration=2.0, bucket_s=1.0)
        self.assertAlmostEqual(density[0], 1.0)
        self.assertAlmostEqual(density[1], 0.0)

    def test_partial_overlap_is_fractional(self):
        words = [w("a", 0.5, 1.0)]  # half of bucket 0
        density = shorts.compute_speech_density(words, duration=1.0, bucket_s=1.0)
        self.assertAlmostEqual(density[0], 0.5)

    def test_word_spanning_multiple_buckets(self):
        words = [w("a", 0.5, 2.5)]  # spans buckets 0, 1, 2 partially
        density = shorts.compute_speech_density(words, duration=3.0, bucket_s=1.0)
        self.assertAlmostEqual(density[0], 0.5)
        self.assertAlmostEqual(density[1], 1.0)
        self.assertAlmostEqual(density[2], 0.5)

    def test_silence_is_zero(self):
        density = shorts.compute_speech_density([], duration=5.0, bucket_s=1.0)
        self.assertTrue((density == 0.0).all())

    def test_never_exceeds_one_even_with_overlapping_words(self):
        words = [w("a", 0.0, 1.0), w("b", 0.0, 1.0)]  # pathological overlap
        density = shorts.compute_speech_density(words, duration=1.0, bucket_s=1.0)
        self.assertLessEqual(density[0], 1.0)


class ScoreWindowsTests(unittest.TestCase):
    def test_picks_the_highest_speech_density_window(self):
        # Dense speech in [20,30), silence elsewhere, over a 60s timeline.
        words = [w(f"w{i}", 20.0 + i, 20.0 + i + 0.5) for i in range(10)]
        starts = shorts.score_windows(
            duration=60.0, words=words, target_duration=10.0, count=1, step_s=5.0,
        )
        self.assertEqual(len(starts), 1)
        self.assertTrue(15.0 <= starts[0] <= 25.0)  # window containing the dense speech

    def test_respects_minimum_gap_between_candidates(self):
        words = [w(f"w{i}", float(i), float(i) + 0.5) for i in range(0, 60, 2)]  # even speech
        starts = shorts.score_windows(
            duration=60.0, words=words, target_duration=10.0, count=3, step_s=5.0,
        )
        min_gap = 10.0 * shorts.MIN_GAP_FACTOR
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, min_gap - 1e-6)

    def test_returns_sorted_by_time_not_by_score(self):
        words = [w(f"w{i}", float(i), float(i) + 0.5) for i in range(0, 60, 2)]
        starts = shorts.score_windows(
            duration=60.0, words=words, target_duration=10.0, count=3, step_s=5.0,
        )
        self.assertEqual(starts, sorted(starts))

    def test_short_duration_returns_at_most_one_window(self):
        starts = shorts.score_windows(
            duration=8.0, words=[], target_duration=10.0, count=3, step_s=5.0,
        )
        self.assertLessEqual(len(starts), 1)

    def test_visual_samples_contribute_to_the_score(self):
        words = []  # no speech anywhere — pure visual-interest comparison
        # High visual interest concentrated in [30, 40), low everywhere else.
        visual = [(float(t), 50.0 if 30 <= t < 40 else 0.0) for t in range(0, 60, 3)]
        starts = shorts.score_windows(
            duration=60.0, words=words, target_duration=10.0, count=1,
            visual_samples=visual, step_s=5.0,
        )
        self.assertEqual(len(starts), 1)
        self.assertTrue(25.0 <= starts[0] <= 35.0)


class BuildShortSrtTests(unittest.TestCase):
    def test_offsets_are_local_to_the_window_not_the_source_timeline(self):
        words = [w("hello", 50.0, 50.3), w("world.", 50.4, 50.8)]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cue.srt"
            n = shorts.build_short_srt(words, window_start=50.0, out_path=out)
            text = out.read_text()
        self.assertEqual(n, 1)
        self.assertIn("00:00:00,000", text)  # 50.0 - 50.0 = 0, not 00:00:50,000
        self.assertIn("HELLO WORLD.", text)

    def test_empty_words_produces_zero_cues_and_empty_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cue.srt"
            n = shorts.build_short_srt([], window_start=0.0, out_path=out)
            text = out.read_text()
        self.assertEqual(n, 0)
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
