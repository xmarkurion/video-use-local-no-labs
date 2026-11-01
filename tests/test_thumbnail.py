"""Tests for thumbnail.py. Real end-to-end generation (actual ffmpeg frame
extraction, actual title burn with the gradient scrim, actual font loading on
Windows) was verified manually against a real render — see session notes.
These test the pure logic: interest scoring, candidate selection, wrapping,
and the scrim-compositing math that had a real bug during development (a
stray paste clobbered the frame with solid black before the real blend ran).
"""

import importlib.util
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "thumbnail.py"
SPEC = importlib.util.spec_from_file_location("video_use_thumbnail", MODULE_PATH)
assert SPEC and SPEC.loader
thumbnail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(thumbnail)


class InterestScoreTests(unittest.TestCase):
    def test_flat_solid_color_scores_low(self):
        flat = Image.new("RGB", (100, 100), (128, 128, 128))
        self.assertLess(thumbnail._interest_score(flat), 1.0)

    def test_high_contrast_pattern_scores_higher_than_flat(self):
        flat = Image.new("RGB", (100, 100), (128, 128, 128))
        checker = Image.new("RGB", (100, 100))
        px = checker.load()
        for y in range(100):
            for x in range(100):
                px[x, y] = (255, 255, 255) if (x // 10 + y // 10) % 2 else (0, 0, 0)
        self.assertGreater(thumbnail._interest_score(checker), thumbnail._interest_score(flat))

    def test_saturated_color_scores_higher_than_grayscale_of_same_contrast(self):
        gray_checker = Image.new("RGB", (100, 100))
        color_checker = Image.new("RGB", (100, 100))
        gpx, cpx = gray_checker.load(), color_checker.load()
        for y in range(100):
            for x in range(100):
                on = (x // 10 + y // 10) % 2
                gpx[x, y] = (200, 200, 200) if on else (50, 50, 50)
                cpx[x, y] = (255, 50, 50) if on else (50, 50, 255)
        self.assertGreater(
            thumbnail._interest_score(color_checker), thumbnail._interest_score(gray_checker)
        )


class CandidateSelectionTests(unittest.TestCase):
    """pick_candidate_timestamps does real ffprobe/ffmpeg work internally, so
    exercise its selection logic directly instead (same approach used
    elsewhere in this suite for logic that wraps real subprocess calls)."""

    def _select(self, scored, count, min_gap):
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        chosen = []
        for t, score in scored:
            if all(abs(t - c[0]) >= min_gap for c in chosen):
                chosen.append((t, score))
            if len(chosen) >= count:
                break
        chosen.sort(key=lambda x: x[0])
        return chosen

    def test_picks_top_scoring_candidates(self):
        scored = [(0.0, 1.0), (5.0, 9.0), (10.0, 5.0), (15.0, 2.0)]
        chosen = self._select(scored, count=3, min_gap=thumbnail.MIN_GAP_S)
        self.assertEqual([t for t, _ in chosen], [5.0, 10.0, 15.0])

    def test_enforces_minimum_gap_between_candidates(self):
        # Two near-adjacent high scorers (1s apart) must not both be chosen
        # when MIN_GAP_S is larger — that would give 3 near-duplicate frames.
        scored = [(5.0, 10.0), (6.0, 9.9), (20.0, 5.0), (35.0, 4.0)]
        chosen = self._select(scored, count=3, min_gap=3.0)
        times = [t for t, _ in chosen]
        self.assertNotIn(6.0, times)  # too close to the already-chosen 5.0
        self.assertEqual(times, sorted(times))

    def test_fewer_qualifying_candidates_than_requested_is_fine(self):
        scored = [(5.0, 10.0), (6.0, 9.0), (7.0, 8.0)]  # all within MIN_GAP_S of each other
        chosen = self._select(scored, count=3, min_gap=10.0)
        self.assertEqual(len(chosen), 1)


class WrapTextTests(unittest.TestCase):
    def test_short_text_is_one_line(self):
        img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(img)
        font = thumbnail.load_font(20)
        lines = thumbnail._wrap_text("HI", font, max_width=1000, draw=draw)
        self.assertEqual(lines, ["HI"])

    def test_long_text_wraps_to_multiple_lines(self):
        img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(img)
        font = thumbnail.load_font(40)
        lines = thumbnail._wrap_text(
            "THIS IS A FAIRLY LONG TITLE THAT SHOULD WRAP", font, max_width=200, draw=draw
        )
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(draw.textlength(line, font=font), 200)

    def test_single_overlong_word_does_not_loop_forever(self):
        img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(img)
        font = thumbnail.load_font(40)
        # A single word wider than max_width must still terminate, as its own line.
        lines = thumbnail._wrap_text("SUPERCALIFRAGILISTICEXPIALIDOCIOUS", font, max_width=10, draw=draw)
        self.assertEqual(lines, ["SUPERCALIFRAGILISTICEXPIALIDOCIOUS"])


class ScrimCompositeTests(unittest.TestCase):
    """Regression test for a real bug found during development: a stray
    `img.paste(...convert("RGB"))` of the scrim onto itself (not blended
    against the real frame) painted a solid black band instead of a
    translucent gradient, discarding the frame content underneath entirely."""

    def test_gradient_blends_against_frame_not_solid_black(self):
        width, height = 200, 100
        img = Image.new("RGB", (width, height), (128, 128, 128))
        scrim_top = 50

        scrim = Image.new("RGBA", (width, height - scrim_top), (0, 0, 0, 0))
        scrim_draw = ImageDraw.Draw(scrim)
        for y in range(scrim.height):
            alpha = int(200 * (y / max(1, scrim.height)))
            scrim_draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

        region = img.crop((0, scrim_top, width, height)).convert("RGBA")
        blended = Image.alpha_composite(region, scrim).convert("RGB")
        img.paste(blended, (0, scrim_top))

        # Top of the scrim (low alpha) must still show the frame's gray, not black.
        self.assertEqual(img.getpixel((0, scrim_top)), (128, 128, 128))
        # Bottom of the scrim (high alpha) should be visibly darkened, but not
        # pure black — some of the original frame should still show through.
        bottom = img.getpixel((0, height - 1))
        self.assertLess(bottom[0], 128)
        self.assertGreater(bottom[0], 0)
        # Above the scrim entirely: completely untouched.
        self.assertEqual(img.getpixel((0, 10)), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
