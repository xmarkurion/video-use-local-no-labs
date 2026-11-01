"""Generate portrait short-form clips (YouTube Shorts / Reels / TikTok) from
a finished render.

Hand it any rendered video — typically `edit/final.mp4` from this skill's own
pipeline, but any video works, no EDL or prior transcript required — and it:

  1. Transcribes it locally (cached — same backend as the rest of this skill,
     free, no quota) if no transcript exists yet for it.
  2. Finds `--count` (default 3) non-overlapping high-interest windows,
     scored by speech density (favor continuous talking, not dead air) and
     visual interest (reused from thumbnail.py — edge energy + saturation).
  3. Snaps each window's edges to real word/sentence boundaries (Hard Rule 6:
     never cut inside a word) instead of an arbitrary timestamp.
  4. Reformats each to 1080x1920 portrait — blurred-background fill by
     default (the standard "landscape repurposed as a Short" look: nothing
     from the original frame is ever cropped away), or a tight center-crop
     with `--style crop`.
  5. Burns captions (bold-overlay, same proven style as render.py's own —
     its MarginV was tuned to clear the Shorts/Reels/TikTok UI chrome on
     vertical video specifically) unless `--no-captions`.
  6. Loudness-normalizes each clip individually — shorts are watched at
     wildly inconsistent volumes, worth the same -14 LUFS / -1 dBTP target
     as a full render.

Usage:
    python helpers/shorts.py edit/final.mp4                  # 3 portrait shorts, captions on
    python helpers/shorts.py edit/final.mp4 --count 5
    python helpers/shorts.py edit/final.mp4 --duration 45
    python helpers/shorts.py edit/final.mp4 --style crop
    python helpers/shorts.py edit/final.mp4 --no-captions
    python helpers/shorts.py edit/final.mp4 --at 12.0 --at 90.0   # explicit starts, skip scoring
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
from PIL import Image

from render import (  # same directory
    SUB_FORCE_STYLE,
    apply_loudnorm_two_pass,
    chunk_words,
    is_nonspeech_token,
    probe_duration,
)

try:
    from thumbnail import _interest_score  # same directory — reuse frame scoring
except Exception:  # pragma: no cover — same reason as render.py's own grade/profanity imports
    def _interest_score(img):  # type: ignore
        return 0.0

PORTRAIT_SIZE = (1080, 1920)
DEFAULT_DURATION_S = 36.0
MIN_DURATION_FACTOR = 0.6   # a snapped clip must stay within
MAX_DURATION_FACTOR = 1.6   # [factor_min, factor_max] * target duration
DEFAULT_COUNT = 3
MIN_GAP_FACTOR = 1.1        # candidates must be at least this many target-durations apart
SCAN_STEP_S = 5.0           # candidate window scan stride
SPEECH_BUCKET_S = 1.0
VISUAL_SAMPLE_EVERY_S = 3.0
VISUAL_WEIGHT = 30.0        # visual score (0-ish to ~1) vs. speech density (0-1) — see score_windows


# -------- Transcription (reuses the skill's own local backend) -------------


def transcribe_source(video: Path, edit_dir: Path | None = None, verbose: bool = True) -> dict:
    """Transcript JSON (Scribe shape) for `video`, local + cached — transcribes
    once, reused on every later call for the same file. `edit_dir` defaults to
    a sibling "edit/" folder, or `video`'s own parent if it's already named
    "edit" (the common case: this usually runs on edit/final.mp4 directly).
    """
    from transcribe import transcribe_one  # same directory
    from transcribe_local import LocalTranscriber  # same directory

    if edit_dir is None:
        edit_dir = video.parent if video.parent.name == "edit" else video.parent / "edit"

    from transcribe import transcript_path  # same directory
    existing = transcript_path(edit_dir, video)
    transcriber = None if existing.exists() else LocalTranscriber(verbose=verbose)
    try:
        out_path = transcribe_one(
            video=video, edit_dir=edit_dir, verbose=verbose,
            backend="local", transcriber=transcriber,
        )
    finally:
        if transcriber is not None:
            transcriber.close()
    return json.loads(out_path.read_text(encoding="utf-8"))


# -------- Word/sentence boundary snapping (Hard Rule 6) --------------------


def _spoken_words(transcript: dict) -> list[dict]:
    return [
        w for w in transcript.get("words", [])
        if w.get("type") == "word" and w.get("start") is not None and w.get("end") is not None
        and not is_nonspeech_token(w.get("text") or "")
    ]


def snap_window(
    words: list[dict],
    raw_start: float,
    raw_end: float,
    target_duration: float,
) -> tuple[float, float]:
    """Nudge a raw [start, end) window onto real word boundaries.

    Start: the first word beginning at or after raw_start — never opens
    mid-word. End: prefer a SENTENCE-ending word (text ends in . ? !) whose
    position keeps the clip's duration within
    [target_duration*MIN_DURATION_FACTOR, target_duration*MAX_DURATION_FACTOR]
    — a clip that ends mid-sentence reads as cut off, so a few seconds of
    slack to land on a clean sentence end is worth it. Falls back to the
    nearest word boundary if no sentence end exists in range.
    """
    if not words:
        return raw_start, raw_end

    after_start = [w for w in words if w["start"] >= raw_start]
    start = after_start[0]["start"] if after_start else words[0]["start"]

    lo = start + target_duration * MIN_DURATION_FACTOR
    hi = start + target_duration * MAX_DURATION_FACTOR
    in_range = [w for w in words if lo <= w["end"] <= hi]

    sentence_enders = [w for w in in_range if (w.get("text") or "").rstrip()[-1:] in ".?!"]
    if sentence_enders:
        best = min(sentence_enders, key=lambda w: abs(w["end"] - (start + target_duration)))
        return start, best["end"]

    if in_range:
        best = min(in_range, key=lambda w: abs(w["end"] - (start + target_duration)))
        return start, best["end"]

    # Nothing landed in [lo, hi] — speech is sparse here (found on real
    # material: a 32s gap between two lines meant the naive "nearest word to
    # raw_end" fallback picked a word 20s short of the minimum, producing a
    # ~3s "short" from a 20s target). Never return a clip shorter than the
    # minimum: prefer the FIRST word ending at or after `lo`, wherever it
    # actually falls, even well past `hi` — a short that runs long still
    # works; one that's a few seconds long does not.
    at_or_after_lo = [w for w in words if w["end"] >= lo]
    if at_or_after_lo:
        candidate_end = at_or_after_lo[0]["end"]
        # Sanity cap: if even the NEAREST qualifying word is absurdly far
        # past the target (a long silent/visual-only stretch with no next
        # line until minutes later), don't let the clip balloon — truncate
        # to 2x the target instead. This can land off a word boundary in
        # this one specific edge case; everywhere else in this function
        # never does.
        if candidate_end - start > target_duration * 2.0:
            return start, start + target_duration * 2.0
        return start, candidate_end

    # No word anywhere reaches the minimum (extremely sparse speech, or this
    # window starts near the very end of the transcript) — nothing better
    # exists than the last word available, short as that leaves the clip.
    at_or_before_end = [w for w in words if w["end"] <= raw_end]
    if at_or_before_end:
        return start, at_or_before_end[-1]["end"]
    return start, words[-1]["end"]


# -------- Candidate scoring --------------------------------------------------


def compute_speech_density(words: list[dict], duration: float, bucket_s: float = SPEECH_BUCKET_S) -> np.ndarray:
    """Fraction of each `bucket_s`-second bucket covered by speech — a window
    of mostly silence scores low regardless of how visually interesting it
    looks, and vice versa."""
    n_buckets = max(1, int(np.ceil(duration / bucket_s)))
    covered = np.zeros(n_buckets, dtype=np.float64)
    for w in words:
        s, e = max(0.0, w["start"]), min(duration, w["end"])
        if e <= s:
            continue
        b0, b1 = int(s // bucket_s), int(e // bucket_s)
        for b in range(b0, min(b1 + 1, n_buckets)):
            bucket_start, bucket_end = b * bucket_s, (b + 1) * bucket_s
            overlap = max(0.0, min(e, bucket_end) - max(s, bucket_start))
            covered[b] += overlap / bucket_s
    return np.clip(covered, 0.0, 1.0)


def sample_visual_interest(video: Path, duration: float, every_s: float = VISUAL_SAMPLE_EVERY_S) -> list[tuple[float, float]]:
    """(timestamp, interest_score) pairs sampled across the video — reuses
    thumbnail.py's frame-interest heuristic (edge energy + saturation)."""
    times = list(np.arange(0.5, max(0.5, duration - 0.5), every_s))
    if not times:
        return [(duration / 2, 0.0)]
    samples: list[tuple[float, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            frame_path = Path(tmp) / "s.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                     "-frames:v", "1", "-vf", "scale=320:-2", str(frame_path)],
                    check=True,
                )
                score = _interest_score(Image.open(frame_path))
            except Exception:
                score = 0.0
            samples.append((float(t), score))
    return samples


def _visual_score_at(samples: list[tuple[float, float]], t0: float, t1: float) -> float:
    in_window = [s for ts, s in samples if t0 <= ts < t1]
    return float(np.mean(in_window)) if in_window else 0.0


def score_windows(
    duration: float,
    words: list[dict],
    target_duration: float,
    count: int,
    visual_samples: list[tuple[float, float]] | None = None,
    step_s: float = SCAN_STEP_S,
) -> list[float]:
    """Raw (pre-snap) start times of the `count` highest-scoring non-
    overlapping windows, at least MIN_GAP_FACTOR*target_duration apart."""
    speech = compute_speech_density(words, duration)
    bucket_s = SPEECH_BUCKET_S
    min_gap = target_duration * MIN_GAP_FACTOR

    starts = list(np.arange(0.0, max(0.0, duration - target_duration), step_s))
    if not starts:
        return [0.0] if duration > 0 else []

    scored: list[tuple[float, float]] = []
    for t in starts:
        b0 = int(t // bucket_s)
        b1 = int((t + target_duration) // bucket_s)
        window_speech = speech[b0:min(b1 + 1, len(speech))]
        speech_score = float(np.mean(window_speech)) if len(window_speech) else 0.0
        visual_score = (
            _visual_score_at(visual_samples, t, t + target_duration) if visual_samples else 0.0
        )
        scored.append((float(t), speech_score + visual_score / max(VISUAL_WEIGHT, 1e-6)))

    scored.sort(key=lambda x: x[1], reverse=True)
    chosen: list[tuple[float, float]] = []
    for t, score in scored:
        if all(abs(t - c[0]) >= min_gap for c in chosen):
            chosen.append((t, score))
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda x: x[0])
    return [t for t, _ in chosen]


# -------- Captions (local to the extracted clip, not the source timeline) --


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_short_srt(words_in_window: list[dict], window_start: float, out_path: Path) -> int:
    """Same bold-overlay chunking as render.py's build_master_srt, but for a
    single self-contained window — offsets are LOCAL to the clip (word.start
    - window_start), not an output-timeline accumulation across many ranges."""
    lines: list[str] = []
    cue_count = 0
    for chunk in chunk_words(words_in_window):
        a = max(0.0, chunk[0]["start"] - window_start)
        b = max(0.0, chunk[-1]["end"] - window_start)
        if b <= a:
            b = a + 0.4
        text = " ".join((w.get("text") or "").strip() for w in chunk)
        text = text.rstrip(",;:").upper()
        cue_count += 1
        lines += [str(cue_count), f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}", text, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return cue_count


# -------- Rendering -----------------------------------------------------------


def render_short(
    video: Path,
    start: float,
    end: float,
    out_path: Path,
    words: list[dict] | None = None,
    style: str = "blur-bg",
    captions: bool = True,
    loudnorm: bool = True,
) -> None:
    """Extract [start, end) from `video`, reformat to 1080x1920 portrait,
    optionally burn captions, optionally loudnorm. One self-contained ffmpeg
    pass for extract+reformat(+captions), then loudnorm as its own pass
    (same two-stage shape as render.py's own pipeline, for the same reason:
    loudnorm needs to measure the already-composited output)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    width, height = PORTRAIT_SIZE

    if style == "crop":
        vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    else:  # blur-bg (default) — nothing from the original frame is ever lost
        vf = (
            f"split=2[bg][fg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=25[bgblur];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgfit];"
            f"[bgblur][fgfit]overlay=(W-w)/2:(H-h)/2"
        )

    srt_path: Path | None = None
    if captions and words:
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = Path(tmp) / "cue.srt"
            words_in_window = [w for w in words if w["start"] < end and w["end"] > start]
            n_cues = build_short_srt(words_in_window, start, srt_path)
            if n_cues == 0:
                srt_path = None
            _render_short_ffmpeg(video, start, duration, vf, srt_path, out_path, style)
    else:
        _render_short_ffmpeg(video, start, duration, vf, None, out_path, style)

    if loudnorm:
        tmp_out = out_path.with_suffix(".prenorm.mp4")
        out_path.rename(tmp_out)
        apply_loudnorm_two_pass(tmp_out, out_path)
        tmp_out.unlink(missing_ok=True)


def _render_short_ffmpeg(
    video: Path, start: float, duration: float, vf: str,
    srt_path: Path | None, out_path: Path, style: str,
) -> None:
    if srt_path is not None:
        # Forward slashes even on Windows: a backslash is an escape character
        # inside a filtergraph — same reasoning as render.py's own subtitle
        # and LUT path handling.
        srt_abs = (
            str(srt_path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        )
        vf_full = f"[0:v]{vf}[preform];[preform]subtitles='{srt_abs}':force_style='{SUB_FORCE_STYLE}'[outv]"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video),
            "-t", f"{duration:.3f}",
            "-filter_complex", vf_full,
            "-map", "[outv]", "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Orchestration -------------------------------------------------------


def generate_shorts(
    video: Path,
    out_dir: Path,
    count: int = DEFAULT_COUNT,
    duration: float = DEFAULT_DURATION_S,
    style: str = "blur-bg",
    captions: bool = True,
    loudnorm: bool = True,
    starts: list[float] | None = None,
    verbose: bool = True,
) -> list[Path]:
    video_duration = probe_duration(video)
    if video_duration is None:
        sys.exit(f"could not probe duration of {video}")
    if video_duration < duration * MIN_DURATION_FACTOR:
        sys.exit(f"{video.name} is only {video_duration:.1f}s — too short for a "
                 f"{duration:.0f}s short. Pass --duration to shrink the target.")

    transcript = None
    words: list[dict] = []
    if captions or starts is None:
        if verbose:
            print("transcribing source (cached if already done)...")
        transcript = transcribe_source(video, verbose=verbose)
        words = _spoken_words(transcript)

    if starts is not None:
        raw_starts = starts
        if verbose:
            print(f"  using {len(raw_starts)} explicit start time(s)")
    else:
        if verbose:
            print("scanning for high-interest windows...")
        visual_samples = sample_visual_interest(video, video_duration)
        raw_starts = score_windows(video_duration, words, duration, count, visual_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, raw_start in enumerate(raw_starts, start=1):
        raw_end = min(video_duration, raw_start + duration)
        if words:
            snapped_start, snapped_end = snap_window(words, raw_start, raw_end, duration)
        else:
            snapped_start, snapped_end = raw_start, raw_end
        snapped_end = min(snapped_end, video_duration)
        if snapped_end <= snapped_start:
            continue

        out_path = out_dir / f"short_{i}_{snapped_start:.1f}s.mp4"
        if verbose:
            print(f"  [{i}/{len(raw_starts)}] {snapped_start:.1f}s-{snapped_end:.1f}s "
                  f"({snapped_end - snapped_start:.1f}s) -> {out_path.name}")
        render_short(
            video, snapped_start, snapped_end, out_path,
            words=words if captions else None, style=style,
            captions=captions, loudnorm=loudnorm,
        )
        paths.append(out_path)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate portrait short-form clips from a rendered video"
    )
    ap.add_argument("video", type=Path, help="Rendered video (e.g. edit/final.mp4)")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="Output directory (default: <video_parent>/shorts)")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT,
                    help=f"How many shorts to generate. Default: {DEFAULT_COUNT}.")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                    help=f"Target clip length in seconds, before snapping to "
                         f"real sentence boundaries. Default: {DEFAULT_DURATION_S:.0f}.")
    ap.add_argument("--style", choices=("blur-bg", "crop"), default="blur-bg",
                    help="blur-bg (default): blurred fill, nothing cropped away. "
                         "crop: tight center-crop to 9:16, fills the frame but can "
                         "lose off-center content.")
    ap.add_argument("--no-captions", dest="captions", action="store_false",
                    help="Skip burning captions.")
    ap.add_argument("--no-loudnorm", dest="loudnorm", action="store_false",
                    help="Skip per-clip loudness normalization.")
    ap.add_argument("--at", type=float, action="append", default=None,
                    help="Explicit start timestamp (seconds). Repeat for multiple. "
                         "Skips the auto-scoring step; end is start+duration, "
                         "still snapped to a sentence boundary.")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    out_dir = (args.output_dir or (video.parent / "shorts")).resolve()
    print(f"generating {'explicit' if args.at else args.count} short(s) from {video.name} "
          f"({args.style}, captions {'on' if args.captions else 'off'})")
    paths = generate_shorts(
        video, out_dir, count=args.count, duration=args.duration, style=args.style,
        captions=args.captions, loudnorm=args.loudnorm, starts=args.at,
    )
    print(f"\n{len(paths)} short(s) -> {out_dir}")


if __name__ == "__main__":
    main()
