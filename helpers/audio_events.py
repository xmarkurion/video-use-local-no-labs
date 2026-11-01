"""Tag non-speech audio events - (laughter), (applause), (sigh).

No local ASR emits these; Scribe did, via `tag_audio_events`, and
`pack_transcripts.py` already knows how to render an `audio_event` entry. This
restores them with an AudioSet classifier (AST, ~350MB, BSD-3) run over a
sliding window, merging adjacent positive windows into spans.

Resolution is the window length, ~1s - good enough for the editorial signal
these tags carry ("there's a laugh at 0:42"). Do not cut to an event boundary;
cut to a word boundary, which is forced-aligned to ~30ms.

Usage:
    python helpers/audio_events.py <audio_or_video_path>
    python helpers/audio_events.py <path> --threshold 0.4 --labels Laughter,Applause
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000

# AudioSet has 527 classes, most of them noise for an editing decision. These are
# the ones that change a cut. Values are the tag text that lands in the transcript.
EVENT_LABELS: dict[str, str] = {
    "Laughter": "laughter",
    "Giggle": "laughter",
    "Chuckle, chortle": "laughter",
    "Snicker": "laughter",
    "Belly laugh": "laughter",
    "Applause": "applause",
    "Clapping": "applause",
    "Cheering": "cheering",
    "Crowd": "crowd",
    "Sigh": "sigh",
    "Cough": "cough",
    "Throat clearing": "throat clearing",
    "Sneeze": "sneeze",
    "Gasp": "gasp",
    "Crying, sobbing": "crying",
    "Whistling": "whistling",
    "Finger snapping": "snap",
    "Music": "music",
    "Singing": "singing",
    "Telephone bell ringing": "phone ringing",
    "Ringtone": "phone ringing",
    "Knock": "knock",
    "Door": "door",
    "Typing": "typing",
    "Computer keyboard": "typing",
    "Mouse": "click",
}

WINDOW_S = 2.0
HOP_S = 1.0
THRESHOLD = 0.35
MIN_DURATION_S = 0.4


class AudioEventTagger:
    """Sliding-window AudioSet tagger. Loads once, reused across a batch."""

    def __init__(
        self,
        device: str = "auto",
        threshold: float = THRESHOLD,
        window_s: float = WINDOW_S,
        hop_s: float = HOP_S,
        labels: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import ASTFeatureExtractor, ASTForAudioClassification  # noqa: PLC0415

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.window_s = window_s
        self.hop_s = hop_s
        self.labels = labels or EVENT_LABELS

        if verbose:
            print(f"  loading {MODEL_ID} on {device}", flush=True)
        self.extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()

        id2label = self.model.config.id2label
        # Map the labels we care about to their logit indices once, so the hot
        # loop only ever touches the columns that can produce a tag.
        self.wanted: dict[int, str] = {
            int(i): self.labels[name]
            for i, name in id2label.items()
            if name in self.labels
        }
        if not self.wanted:
            raise RuntimeError(
                f"none of the requested labels exist in {MODEL_ID}: "
                f"{sorted(self.labels)}"
            )

    def tag(self, audio) -> list[dict]:
        """Score a 16kHz mono float32 array. Returns [{text, start, end, score}]."""
        import numpy as np  # noqa: PLC0415

        torch = self._torch
        audio = np.asarray(audio, dtype=np.float32)
        win = int(self.window_s * SAMPLE_RATE)
        hop = int(self.hop_s * SAMPLE_RATE)
        if audio.size < win:
            audio = np.pad(audio, (0, win - audio.size))

        starts = list(range(0, max(1, audio.size - win + 1), hop))
        # AST was trained on 10.24s inputs and the feature extractor pads to that
        # length, so a 2s window arrives padded. That is what buys localization
        # finer than the model's native clip length.
        batches = [audio[s:s + win] for s in starts]

        hits: list[tuple[float, float, str, float]] = []
        BATCH = 16
        for b0 in range(0, len(batches), BATCH):
            chunk = batches[b0:b0 + BATCH]
            inputs = self.extractor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()

            for k, row in enumerate(probs):
                t0 = starts[b0 + k] / SAMPLE_RATE
                for idx, tag in self.wanted.items():
                    p = float(row[idx])
                    if p >= self.threshold:
                        hits.append((t0, t0 + self.window_s, tag, p))

        return _merge_spans(hits)


def _merge_spans(hits: list[tuple[float, float, str, float]]) -> list[dict]:
    """Collapse overlapping windows of the same tag into one span."""
    by_tag: dict[str, list[tuple[float, float, float]]] = {}
    for start, end, tag, score in hits:
        by_tag.setdefault(tag, []).append((start, end, score))

    events: list[dict] = []
    for tag, spans in by_tag.items():
        spans.sort()
        cur_start, cur_end, cur_score = spans[0]
        for start, end, score in spans[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                cur_score = max(cur_score, score)
            else:
                events.append({"text": tag, "start": cur_start, "end": cur_end,
                               "score": round(cur_score, 3)})
                cur_start, cur_end, cur_score = start, end, score
        events.append({"text": tag, "start": cur_start, "end": cur_end,
                       "score": round(cur_score, 3)})

    events = [e for e in events if e["end"] - e["start"] >= MIN_DURATION_S]
    events.sort(key=lambda e: e["start"])
    return events


def _load_audio(path: Path):
    """Decode anything ffmpeg reads into 16kHz mono float32."""
    import numpy as np  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import wave  # noqa: PLC0415

        with wave.open(str(wav), "rb") as w:
            raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag non-speech audio events with AudioSet/AST")
    ap.add_argument("path", type=Path, help="Audio or video file")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"Sigmoid confidence to accept a window (default {THRESHOLD})")
    ap.add_argument("--window", type=float, default=WINDOW_S,
                    help=f"Window length in seconds (default {WINDOW_S})")
    ap.add_argument("--hop", type=float, default=HOP_S,
                    help=f"Hop in seconds; this is the timing resolution (default {HOP_S})")
    ap.add_argument("--labels", type=str, default=None,
                    help="Comma-separated AudioSet class names to keep. "
                         "Omit for the curated editorial set.")
    ap.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write JSON here instead of stdout")
    args = ap.parse_args()

    if not args.path.exists():
        sys.exit(f"not found: {args.path}")

    labels = None
    if args.labels:
        labels = {name.strip(): name.strip().lower()
                  for name in args.labels.split(",") if name.strip()}

    tagger = AudioEventTagger(
        device=args.device, threshold=args.threshold,
        window_s=args.window, hop_s=args.hop, labels=labels,
    )
    events = tagger.tag(_load_audio(args.path))

    payload = json.dumps(events, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"{len(events)} events -> {args.output}")
    else:
        print(payload)
        print(f"\n{len(events)} events", file=sys.stderr)


if __name__ == "__main__":
    main()
