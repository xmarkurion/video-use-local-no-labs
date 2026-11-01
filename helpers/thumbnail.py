"""Generate thumbnail candidates from a rendered video.

Produces 3 distinct frame candidates by default — the thing worth testing is
almost always "which MOMENT works as a thumbnail", not "text vs no text" (an
obvious call once you can see the frame). All 3 get the same treatment
(title overlay if given, same crop/resize), so the comparison is apples to
apples: same style, different underlying frame.

Frame selection, no ML dependency: sample evenly across the video, score each
sample for visual "interest" (edge energy via a Sobel-ish gradient + overall
saturation — avoids picking a flat, low-detail moment), then take the top 3
local maxima at least MIN_GAP_S apart so the three candidates are genuinely
different moments, not three near-duplicate frames off the same peak.

Usage:
    python helpers/thumbnail.py final.mp4                       # 3 auto-picked candidates
    python helpers/thumbnail.py final.mp4 --title "LAUNCH VIDEO"
    python helpers/thumbnail.py final.mp4 --at 2.5 --at 14.0 --at 30.0  # explicit timestamps
    python helpers/thumbnail.py final.mp4 -o edit/thumbnails --count 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Windows' legacy console codepage (cp1252) can't encode some characters that
# show up in these prints (an em dash is fine; a unicode arrow like U+2192
# raised UnicodeEncodeError and crashed the run after the real work was
# already done and the output file already written). Force UTF-8 stdout/
# stderr so prose text never crashes a run regardless of what a future edit
# adds. Wrapped in try/except: reconfigure can itself fail against some
# redirected/piped streams, and a cosmetic safety net must never be fatal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from timeline_view import load_font  # same directory — reuses the real font search
except Exception:  # pragma: no cover — same reason as render.py's own grade/profanity imports
    def load_font(size: int):  # type: ignore
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

THUMB_SIZES = {
    "youtube": (1280, 720),
    "square": (1080, 1080),
    "vertical": (1080, 1920),
}
SAMPLE_INTERVAL_S = 1.0
MIN_GAP_S = 3.0  # candidates must be at least this far apart
EDGE_SKIP_S = 0.5  # ignore the very first/last fraction-of-a-second (fade in/out, often black)


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _extract_frame(video: Path, t: float, dest: Path, width: int = 480) -> None:
    """Small frame for scoring — full-res extraction happens once, later, only
    for the 3 chosen candidates. Scoring 30-60 small frames is fast; scoring
    that many full-res frames would not be."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)],
        check=True,
    )


def _interest_score(img: Image.Image) -> float:
    """Higher = more visually interesting as a thumbnail: detailed (edge
    energy) and colorful (saturation), not a flat/empty/monochrome frame."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_energy = gx + gy

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = ((mx - mn) / (mx + 1e-6)).mean()

    return float(edge_energy + saturation * 40)


def pick_candidate_timestamps(video: Path, count: int = 3) -> list[tuple[float, float]]:
    """Returns up to `count` (timestamp, score) pairs, sorted by timestamp,
    each at least MIN_GAP_S apart, chosen from the highest-scoring samples."""
    duration = probe_duration(video)
    t0 = min(EDGE_SKIP_S, duration * 0.02)
    t1 = max(t0, duration - min(EDGE_SKIP_S, duration * 0.02))
    if t1 <= t0:
        return [(duration / 2, 0.0)]

    sample_times = list(np.arange(t0, t1, SAMPLE_INTERVAL_S))
    if not sample_times:
        sample_times = [t0]

    scored: list[tuple[float, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in sample_times:
            frame_path = Path(tmp) / "s.jpg"
            try:
                _extract_frame(video, float(t), frame_path)
                score = _interest_score(Image.open(frame_path))
            except Exception:
                score = -1.0
            scored.append((float(t), score))

    scored.sort(key=lambda x: x[1], reverse=True)

    chosen: list[tuple[float, float]] = []
    for t, score in scored:
        if all(abs(t - c[0]) >= MIN_GAP_S for c in chosen):
            chosen.append((t, score))
        if len(chosen) >= count:
            break

    chosen.sort(key=lambda x: x[0])
    return chosen


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_thumbnail(
    video: Path,
    timestamp: float,
    out_path: Path,
    title: str | None = None,
    size: tuple[int, int] = THUMB_SIZES["youtube"],
    accent: tuple[int, int, int] = (255, 90, 0),
) -> None:
    """Extract one full-resolution frame, crop/scale to `size`, optionally
    burn a bold title in the lower third with a gradient scrim behind it —
    the standard thumbnail-readable-at-small-size treatment."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.png"
        # Scale to cover, then center-crop to the exact target aspect — avoids
        # letterboxing when the source aspect doesn't match the thumbnail's.
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", vf, str(raw)],
            check=True,
        )
        img = Image.open(raw).convert("RGB")

    if title:
        img = img.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        font_size = max(28, height // 11)
        font = load_font(font_size)
        margin = width // 20
        max_text_width = width - 2 * margin
        lines = _wrap_text(title.upper(), font, max_text_width, draw)

        line_height = int(font_size * 1.25)
        block_height = line_height * len(lines) + margin
        scrim_top = height - block_height - margin // 2
        # Gradient scrim (not a flat band) so the frame under the text stays
        # visible, just progressively darkened — not a harsh pasted-on bar.
        scrim = Image.new("RGBA", (width, height - scrim_top), (0, 0, 0, 0))
        scrim_draw = ImageDraw.Draw(scrim)
        for y in range(scrim.height):
            alpha = int(200 * (y / max(1, scrim.height)))
            scrim_draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
        # alpha_composite needs the scrim blended against the REAL frame
        # content in that region, not pasted over it — pasting an
        # already-RGB-converted (alpha-discarded) scrim would just paint a
        # solid black band instead of a translucent gradient.
        region = img.crop((0, scrim_top, width, height)).convert("RGBA")
        blended = Image.alpha_composite(region, scrim).convert("RGB")
        img.paste(blended, (0, scrim_top))
        draw = ImageDraw.Draw(img)

        y = height - margin - line_height * len(lines)
        for line in lines:
            # 3px black outline for legibility over any background, accent-
            # colored fill — the launch-video palette this project already uses.
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
                draw.text((margin + dx, y + dy), line, font=font, fill=(0, 0, 0))
            draw.text((margin, y), line, font=font, fill=accent)
            y += line_height

    img.save(out_path, quality=92)


def generate_thumbnails(
    video: Path,
    out_dir: Path,
    title: str | None = None,
    timestamps: list[float] | None = None,
    count: int = 3,
    size_name: str = "youtube",
) -> list[Path]:
    size = THUMB_SIZES.get(size_name, THUMB_SIZES["youtube"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if timestamps:
        chosen = [(t, None) for t in timestamps]
        print(f"  using {len(chosen)} explicit timestamp(s)")
    else:
        print(f"  scanning for {count} diverse candidate frame(s)...")
        chosen = pick_candidate_timestamps(video, count=count)
        for t, score in chosen:
            print(f"    candidate at {t:6.2f}s  (interest score {score:.1f})")

    paths: list[Path] = []
    for i, (t, _score) in enumerate(chosen, start=1):
        out_path = out_dir / f"thumbnail_{i}_{t:.1f}s.jpg"
        render_thumbnail(video, t, out_path, title=title, size=size)
        paths.append(out_path)
        print(f"  [{i}/{len(chosen)}] {t:6.2f}s -> {out_path.name}")

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thumbnail candidates from a rendered video")
    ap.add_argument("video", type=Path, help="Rendered video (preview.mp4 or final.mp4)")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Output directory (default: <video_parent>/thumbnails)")
    ap.add_argument("--title", type=str, default=None,
                    help="Title text to burn into the lower third of every candidate. "
                         "Omit for clean frames with no text.")
    ap.add_argument("--at", type=float, action="append", default=None,
                    help="Explicit timestamp (seconds) for a candidate. Repeat for "
                         "multiple. Overrides auto-selection.")
    ap.add_argument("--count", type=int, default=3,
                    help="How many candidates to auto-select when --at isn't given. Default: 3.")
    ap.add_argument("--size", choices=sorted(THUMB_SIZES), default="youtube",
                    help="Output dimensions: youtube (1280x720, default), square (1080x1080), "
                         "vertical (1080x1920).")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    out_dir = (args.output_dir or (video.parent / "thumbnails")).resolve()
    print(f"generating thumbnails from {video.name}")
    paths = generate_thumbnails(
        video, out_dir, title=args.title, timestamps=args.at,
        count=args.count, size_name=args.size,
    )
    print(f"\n{len(paths)} thumbnail(s) -> {out_dir}")


if __name__ == "__main__":
    main()
