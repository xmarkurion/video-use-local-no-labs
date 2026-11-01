"""Transcribe a video to word-level JSON.

Extracts mono 16kHz audio via ffmpeg, transcribes it, writes the full response
to <edit_dir>/transcripts/<video_stem>.json.

Two backends, same output shape:

  local (default)  WhisperX + CrisperWhisper on the GPU. Verbatim - keeps the
                   fillers, stutters and false starts a cut might want to
                   remove. Word timestamps from wav2vec2 forced alignment.
                   Free, offline, no quota. See transcribe_local.py.

  elevenlabs       Hosted Scribe. Costs credits. Kept for A/B comparison and
                   as a fallback; needs ELEVENLABS_API_KEY.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --diarize --num-speakers 2
    python helpers/transcribe.py <video_path> --audio-events
    python helpers/transcribe.py <video_path> --model large-v3    # MIT, not verbatim
    python helpers/transcribe.py <video_path> --backend elevenlabs
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit --language en
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
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
import time
import wave
from pathlib import Path

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"

BACKENDS = ("local", "elevenlabs")


def _dotenv_value(key: str) -> str | None:
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
    val = os.environ.get(key, "").strip()
    return val or None


def load_api_key() -> str:
    """ElevenLabs key, for --backend elevenlabs only."""
    v = _dotenv_value("ELEVENLABS_API_KEY")
    if not v:
        sys.exit(
            "ELEVENLABS_API_KEY not found in .env or environment.\n"
            "The local backend needs no key - drop --backend elevenlabs."
        )
    return v


def load_hf_token() -> str | None:
    """HuggingFace token, for --diarize only. Everything else runs token-free."""
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = _dotenv_value(key)
        if v:
            return v
    return None


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    import requests  # noqa: PLC0415  - only the hosted backend needs it

    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    backend: str = "local",
    transcriber=None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    `transcriber` is a live LocalTranscriber to reuse; batch mode passes one so
    the models load once for the whole run.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        # A silent track transcribes to nothing, or worse to hallucinated text.
        # Catch the wrong-track case before spending GPU time or Scribe credits.
        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - not transcribing. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        size_mb = audio.stat().st_size / (1024 * 1024)

        if backend == "elevenlabs":
            if verbose:
                print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            payload = call_scribe(audio, api_key, language, num_speakers)
        else:
            if verbose:
                print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            if transcriber is None:
                raise RuntimeError("local backend needs a LocalTranscriber instance")
            payload = transcriber.transcribe(audio, language=language,
                                             num_speakers=num_speakers)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            spoken = [w for w in payload["words"] if w.get("type") == "word"]
            events = [w for w in payload["words"] if w.get("type") == "audio_event"]
            print(f"    words: {len(spoken)}"
                  + (f", audio events: {len(events)}" if events else ""))

    return out_path


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by transcribe.py and transcribe_batch.py."""
    from transcribe_local import DEFAULT_MODEL  # noqa: PLC0415

    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="local",
        help="local (WhisperX + CrisperWhisper, free, verbatim) or elevenlabs "
             "(hosted Scribe, costs credits). Default: local.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Local ASR model. Default {DEFAULT_MODEL} (verbatim, CC-BY-NC-4.0). "
             "Use 'large-v3' for an MIT-licensed model that normalizes fillers away.",
    )
    ap.add_argument(
        "--device", type=str, default="auto", help="cuda, cpu, or auto. Default: auto."
    )
    ap.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        help="CTranslate2 precision: float16 (default, ~3GB VRAM), int8_float16, "
             "float32, int8 for CPU.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=8,
        help="WhisperX batch size. 8 fits 12GB alongside the other models; raise it "
             "on a bigger card for more throughput. Default: 8.",
    )
    ap.add_argument(
        "--diarize",
        action="store_true",
        help="Label words by speaker with pyannote. Needs HF_TOKEN in .env and the "
             "terms accepted at huggingface.co/pyannote/speaker-diarization-community-1.",
    )
    ap.add_argument(
        "--audio-events",
        action="store_true",
        help="Tag (laughter)/(applause)/(sigh) with an AudioSet classifier. ~1s "
             "resolution - editorial signal, not a cut point.",
    )
    ap.add_argument(
        "--vad-method", type=str, default="pyannote", choices=["pyannote", "silero"],
        help="Voice activity detection used to chunk long audio. Default: pyannote.",
    )
    ap.add_argument(
        "--filler-prompt",
        action="store_true",
        help="Prime the decoder with filler words. Only useful with --model large-v3; "
             "CrisperWhisper is verbatim already.",
    )


def build_transcriber(args, verbose: bool = True):
    """Construct a LocalTranscriber from parsed args, or None for the hosted backend."""
    if args.backend != "local":
        return None
    from transcribe_local import LocalTranscriber  # noqa: PLC0415

    return LocalTranscriber(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        diarize=args.diarize,
        hf_token=load_hf_token() if args.diarize else None,
        audio_events=args.audio_events,
        vad_method=args.vad_method,
        filler_prompt=args.filler_prompt,
        verbose=verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcribe a video to word-level JSON (local GPU by default)"
    )
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    add_backend_args(ap)
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    # Cache check before model load: a cached transcript should not cost 30s of
    # CUDA warm-up.
    if transcript_path(edit_dir, video, args.audio_track).exists():
        print(f"cached: {transcript_path(edit_dir, video, args.audio_track).name}")
        return

    api_key = load_api_key() if args.backend == "elevenlabs" else None
    transcriber = build_transcriber(args)

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
        backend=args.backend,
        transcriber=transcriber,
    )


if __name__ == "__main__":
    main()
