"""Local GPU transcription that emits Scribe-shaped JSON.

WhisperX pipeline with CrisperWhisper weights:

  CrisperWhisper (CTranslate2)  ->  verbatim text, every filler and false start
  wav2vec2 forced alignment     ->  word timestamps (~30ms, not Whisper's DTW drift)
  pyannote community-1          ->  speaker_id per word          [--diarize]
  AST AudioSet tagger           ->  (laughter) / (applause) spans [--audio-events]

The two halves matter together. CrisperWhisper keeps the disfluencies that stock
Whisper deletes, but its CTranslate2 build does not carry the timestamp accuracy
of the reference implementation; WhisperX throws those timestamps away and
forced-aligns instead, so the pipeline cancels that weakness out.

Output is the same shape `pack_transcripts.py` and `render.py` already read from
ElevenLabs Scribe - a flat `words` list of `word` / `spacing` / `audio_event`
entries - so nothing downstream knows which backend produced a transcript.

Model weights note: CrisperWhisper is CC-BY-NC-4.0 (research / personal use).
For commercial work either license it from nyra labs or pass
`--model large-v3`, which is MIT but normalizes fillers away.
"""

from __future__ import annotations

import re
from pathlib import Path

# CrisperWhisper v1 converted to CTranslate2 - the build faster-whisper can load,
# and the only CrisperWhisper that runs on Windows today (the `crisperwhisper[ct2]`
# extra is Linux-only, and its PyTorch backend does not plug into WhisperX).
DEFAULT_MODEL = "nyralabs/faster_CrisperWhisper"

# pyannote's gated pipeline. Accept the terms once at
# https://huggingface.co/pyannote/speaker-diarization-community-1
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

# Nudges a stock Whisper toward verbatim. A no-op for CrisperWhisper, which is
# trained for it - this only earns its keep with --model large-v3.
FILLER_PROMPT = "Um, uh, so, like, you know, I mean, er, hmm, uhh, ahh."

_SPEAKER_RE = re.compile(r"^speaker[_-]?(\d+)$", re.IGNORECASE)


def normalize_speaker(speaker: str | None) -> str | None:
    """pyannote's "SPEAKER_00" -> Scribe's "speaker_0".

    `pack_transcripts.py` renders a speaker tag by stripping a literal
    "speaker_" prefix, so a raw pyannote label would print as "SSPEAKER_00".
    """
    if speaker is None:
        return None
    m = _SPEAKER_RE.match(str(speaker).strip())
    if m:
        return f"speaker_{int(m.group(1))}"
    return str(speaker)


def _fill_missing_times(words: list[dict]) -> list[dict]:
    """Give every word a start and an end.

    WhisperX omits both keys for a word it could not align (no character in the
    wav2vec2 dictionary - a stray emoji, a bare "..."), even after its own
    interpolation pass. Downstream skips a word with no timestamp, which would
    silently drop it from the editor's reading view, so pin it to the gap
    between its neighbours instead.
    """
    out = [dict(w) for w in words]
    n = len(out)

    for i, w in enumerate(out):
        if w.get("start") is not None and w.get("end") is not None:
            continue

        prev_end = next(
            (out[j].get("end") for j in range(i - 1, -1, -1) if out[j].get("end") is not None),
            None,
        )
        next_start = next(
            (out[j].get("start") for j in range(i + 1, n) if out[j].get("start") is not None),
            None,
        )

        if prev_end is None and next_start is None:
            start = end = 0.0
        elif prev_end is None:
            start = end = float(next_start)
        elif next_start is None:
            start = end = float(prev_end)
        else:
            # Share the gap rather than claiming all of it: a long pause around an
            # unalignable token would otherwise read as a long spoken word.
            span = max(0.0, float(next_start) - float(prev_end))
            start = float(prev_end)
            end = start + min(span, 0.2)

        w["start"] = w.get("start") if w.get("start") is not None else start
        w["end"] = w.get("end") if w.get("end") is not None else end
        if w["end"] < w["start"]:
            w["end"] = w["start"]

    return out


def _nearest_speaker(words: list[dict], t: float) -> str | None:
    """Speaker of the word closest in time to `t`, Scribe-normalized."""
    best: str | None = None
    best_gap = float("inf")
    for w in words:
        if w.get("speaker") is None:
            continue
        gap = 0.0 if w["start"] <= t <= w["end"] else min(
            abs(t - w["start"]), abs(t - w["end"])
        )
        if gap < best_gap:
            best_gap = gap
            best = w["speaker"]
    return normalize_speaker(best)


def to_scribe_schema(
    segments: list[dict],
    language_code: str,
    events: list[dict] | None = None,
    model_id: str | None = None,
    verbatim: bool | None = None,
) -> dict:
    """WhisperX aligned segments -> the Scribe response shape.

    Scribe returns one flat `words` list carrying three entry types:
      `word`         - a spoken token, punctuation attached
      `spacing`      - the gap between two words; its start/end is how
                       `pack_transcripts.py` finds phrase breaks
      `audio_event`  - "(laughter)", "(applause)"

    Entries are time-ordered, with audio events merged in by start time.
    """
    flat: list[dict] = []
    for seg in segments:
        for w in seg.get("words", []):
            text = (w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            flat.append({
                "text": text,
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker", seg.get("speaker")),
                "score": w.get("score"),
            })

    flat = _fill_missing_times(flat)
    flat.sort(key=lambda w: (w["start"], w["end"]))

    words: list[dict] = []
    for i, w in enumerate(flat):
        speaker = normalize_speaker(w.get("speaker"))
        entry: dict = {
            "text": w["text"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
            "type": "word",
        }
        if speaker is not None:
            entry["speaker_id"] = speaker
        if w.get("score") is not None:
            entry["logprob"] = round(float(w["score"]), 4)
        words.append(entry)

        # Spacing entry for the gap to the next word. Scribe emits one for every
        # inter-word gap including a zero-length one; matching that keeps the
        # phrase-breaking logic in pack_transcripts.py on familiar ground.
        if i + 1 < len(flat):
            gap_start = round(float(w["end"]), 3)
            gap_end = round(float(flat[i + 1]["start"]), 3)
            if gap_end < gap_start:
                gap_end = gap_start
            spacing: dict = {
                "text": " ",
                "start": gap_start,
                "end": gap_end,
                "type": "spacing",
            }
            if speaker is not None:
                spacing["speaker_id"] = speaker
            words.append(spacing)

    for ev in events or []:
        label = ev["text"]
        if not label.startswith("("):
            label = f"({label})"
        entry = {
            "text": label,
            "start": round(float(ev["start"]), 3),
            "end": round(float(ev["end"]), 3),
            "type": "audio_event",
        }
        # Scribe tags events with a speaker, and so must we: pack_transcripts
        # takes a phrase's speaker from its first entry, so an untagged laugh
        # sitting just before a line would strip that line's speaker label.
        speaker = _nearest_speaker(flat, float(ev["start"]))
        if speaker is not None:
            entry["speaker_id"] = speaker
        words.append(entry)

    words.sort(key=lambda w: (w["start"], 0 if w["type"] == "word" else 1))

    text = " ".join(w["text"] for w in words if w["type"] == "word")
    text = (
        text.replace(" ,", ",").replace(" .", ".")
        .replace(" ?", "?").replace(" !", "!")
    )

    out = {
        "language_code": language_code,
        "text": text,
        "words": words,
    }
    if model_id:
        out["_video_use"] = {
            "backend": "local",
            "model": model_id,
            "verbatim": verbatim,
        }
    return out


class LocalTranscriber:
    """Holds the loaded models so batch mode pays the load cost once.

    Cold start is 20-40s across the four models; a 10-minute take then runs in
    well under a minute on a 12GB card. Batch mode keeps one instance for the
    whole run - see transcribe_batch.py.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "float16",
        batch_size: int = 8,
        diarize: bool = False,
        hf_token: str | None = None,
        audio_events: bool = False,
        vad_method: str = "pyannote",
        filler_prompt: bool = False,
        verbose: bool = True,
    ) -> None:
        import torch  # noqa: PLC0415  - heavy; only the local backend pays for it
        import whisperx  # noqa: PLC0415

        self._whisperx = whisperx
        self._torch = torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and compute_type == "float16":
            # CTranslate2 has no float16 CPU kernels and fails late and cryptically.
            compute_type = "int8"

        self.model_id = model
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.diarize = diarize
        self.audio_events = audio_events
        self.verbose = verbose
        # CrisperWhisper is verbatim by training; stock Whisper never is.
        self.verbatim = "crisper" in model.lower()

        asr_options: dict = {}
        if filler_prompt:
            asr_options["initial_prompt"] = FILLER_PROMPT

        if verbose:
            print(f"  loading {model} on {device} ({compute_type})", flush=True)
        self.asr = whisperx.load_model(
            model,
            device=device,
            compute_type=compute_type,
            vad_method=vad_method,
            asr_options=asr_options or None,
        )

        self._align_cache: dict[str, tuple] = {}

        self.diarizer = None
        if diarize:
            if not hf_token:
                raise RuntimeError(
                    "--diarize needs a HuggingFace token. Accept the terms at "
                    "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                    "then put HF_TOKEN=hf_... in .env or the environment."
                )
            if verbose:
                print(f"  loading {DEFAULT_DIARIZE_MODEL}", flush=True)
            from whisperx.diarize import DiarizationPipeline  # noqa: PLC0415

            self.diarizer = DiarizationPipeline(
                model_name=DEFAULT_DIARIZE_MODEL, token=hf_token, device=device
            )

        self.tagger = None
        if audio_events:
            from audio_events import AudioEventTagger  # noqa: PLC0415

            self.tagger = AudioEventTagger(device=device, verbose=verbose)

    def _align_model(self, language_code: str):
        if language_code not in self._align_cache:
            if self.verbose:
                print(f"  loading alignment model for '{language_code}'", flush=True)
            self._align_cache[language_code] = self._whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self._align_cache[language_code]

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        num_speakers: int | None = None,
    ) -> dict:
        """Transcribe one 16kHz mono wav. Returns a Scribe-shaped dict."""
        whisperx = self._whisperx
        # Decode once, here, and pass the array to every stage below.
        #
        # This is load-bearing, not a micro-optimization. whisperx.load_audio
        # shells out to the ffmpeg CLI, whereas pyannote and torchaudio decode
        # through torchcodec, which only links FFmpeg 4-7 and fails outright
        # against a modern FFmpeg 8 install. Hand any stage a *path* and it may
        # reach for that broken decoder; hand it an array and none of them can.
        audio = whisperx.load_audio(str(audio_path))

        result = self.asr.transcribe(
            audio, batch_size=self.batch_size, language=language
        )
        language_code = result.get("language") or language or "en"

        try:
            align_model, metadata = self._align_model(language_code)
        except Exception as e:
            # No wav2vec2 model for this language. Segment-level timing is worse
            # than Scribe's, so say so loudly rather than quietly shipping it.
            print(
                f"  warning: no alignment model for '{language_code}' ({e}); "
                "falling back to segment-level timestamps - cut edges will need "
                "wider padding",
                flush=True,
            )
            aligned = {"segments": result["segments"]}
            for seg in aligned["segments"]:
                seg.setdefault("words", [{
                    "word": seg.get("text", "").strip(),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                }])
        else:
            aligned = whisperx.align(
                result["segments"], align_model, metadata, audio, self.device,
                return_char_alignments=False,
            )

        if self.diarizer is not None:
            diarize_df = self.diarizer(audio, num_speakers=num_speakers)
            aligned = whisperx.assign_word_speakers(diarize_df, aligned)

        events = self.tagger.tag(audio) if self.tagger is not None else None

        return to_scribe_schema(
            aligned["segments"],
            language_code=language_code,
            events=events,
            model_id=self.model_id,
            verbatim=self.verbatim,
        )

    def close(self) -> None:
        """Release VRAM. Worth calling between a batch run and a render."""
        self.asr = None
        self._align_cache.clear()
        self.diarizer = None
        self.tagger = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
