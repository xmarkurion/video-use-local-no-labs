"""Batch-transcribe every video in a directory.

Walks <videos_dir> for common video extensions and writes transcripts to
<videos_dir>/edit/transcripts/<name>.json.

Concurrency differs by backend, because the bottleneck does:

  local       one worker, one set of models loaded for the whole run. The GPU is
              already saturated by a single stream, and four copies of
              CrisperWhisper would not fit in 12GB anyway. The win here is
              loading the models once instead of once per file.
  elevenlabs  four parallel uploads - the wait is network, not compute.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --diarize --num-speakers 2
    python helpers/transcribe_batch.py <videos_dir> --audio-events
    python helpers/transcribe_batch.py <videos_dir> --backend elevenlabs --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import (
    add_backend_args,
    build_transcriber,
    load_api_key,
    transcribe_one,
    transcript_path,
)

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch transcription of a videos directory")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers. Default: 1 for the local backend (GPU-bound), "
             "4 for elevenlabs (network-bound). Forcing >1 locally will contend "
             "for VRAM.",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers. Improves diarization when known.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe (OBS: 0 = game, 1 = mic).",
    )
    add_backend_args(ap)
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    cached = {v for v in videos
              if transcript_path(edit_dir, v, args.audio_track).exists()}
    pending = [v for v in videos if v not in cached]

    print(f"found {len(videos)} videos ({len(cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    workers = args.workers if args.workers else (1 if args.backend == "local" else 4)
    if args.backend == "local" and workers > 1:
        print(f"warning: {workers} workers share one GPU; expect VRAM pressure, "
              "not speedup")

    api_key = load_api_key() if args.backend == "elevenlabs" else None
    transcriber = build_transcriber(args)

    label = args.model if args.backend == "local" else "elevenlabs scribe"
    print(f"transcribing {len(pending)} files with {label} ({workers} worker"
          f"{'s' if workers != 1 else ''})")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []

    def run(v: Path) -> Path:
        return transcribe_one(
            video=v,
            edit_dir=edit_dir,
            api_key=api_key,
            language=args.language,
            num_speakers=args.num_speakers,
            verbose=False,
            audio_track=args.audio_track,
            backend=args.backend,
            transcriber=transcriber,
        )

    if workers == 1:
        # Sequential, so a long run reports progress as it goes rather than in a
        # burst at the end.
        for i, v in enumerate(pending, 1):
            try:
                out = run(v)
                print(f"  [{i}/{len(pending)}] + {v.stem}  ->  {out.name}", flush=True)
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  [{i}/{len(pending)}] x {v.stem}  FAILED: {e}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, v): v for v in pending}
            for fut in as_completed(futures):
                v = futures[fut]
                try:
                    out = fut.result()
                    print(f"  + {v.stem}  ->  {out.name}")
                except Exception as e:
                    errors.append((v, str(e)))
                    print(f"  x {v.stem}  FAILED: {e}")

    if transcriber is not None:
        transcriber.close()

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
