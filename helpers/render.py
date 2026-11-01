"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}

try:
    from profanity_filter import (  # same directory
        build_profanity_matcher,
        censor_word,
        is_profane,
        load_profanity_stems,
    )
except Exception:
    # Same reason as the grade.py fallback above: a caller that loads this
    # file via importlib.util.spec_from_file_location (every test in tests/
    # does) doesn't put helpers/ on sys.path, so a plain sibling import 404s.
    # These stubs are also exactly correct behavior for "no matcher" — a
    # matcher of None is already a documented no-op in helpers/profanity_filter.py.
    def load_profanity_stems(path=None):  # type: ignore
        return []

    def build_profanity_matcher(stems):  # type: ignore
        return None

    def is_profane(text, matcher):  # type: ignore
        return False

    def censor_word(text, matcher):  # type: ignore
        return text


# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
#
# MarginV is NOT taste — it is a platform safe-zone rule.
# TikTok / IG Reels / Shorts UI (caption, username, music, right-rail actions)
# covers roughly the bottom ~25–30% of a 1080×1920 frame. Captions placed near
# the bottom edge get clipped or obscured by the UI. libass auto-scales the
# render canvas relative to PlayResY=288, so MarginV=90 lands the caption
# baseline roughly 30% up from the bottom on any aspect — clear of the UI on
# every major vertical-video platform. Do not drop this below ~75 without a
# specific reason.
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=90"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


# -------- 3D LUTs (creative look, applied per-segment after grade) ---------
#
# A LUT is a separate concern from `grade`: grade is corrective (exposure,
# contrast, saturation — fix what's wrong), a LUT is a creative "look" baked
# by a colorist or a camera manufacturer (log-to-Rec.709 conversion, a film
# emulation). Both can be present on the same segment; grade runs first.

LUT_DIR = Path(__file__).resolve().parent.parent / "lut"
LUT_EXTENSIONS = (".cube", ".3dl", ".dat", ".m3d", ".csp")


def find_lut_file(name: str, edit_dir: Path | None = None) -> Path:
    """Resolve a LUT by bare name against lut/, or by an explicit path.

    `name` can be:
      - a bare name ("kodak_2383")            -> lut/kodak_2383.cube (or any
                                                  other known extension)
      - a bare name with extension already on -> lut/<name> as given
      - an absolute or relative path to a LUT file anywhere on disk
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if edit_dir is not None:
        relative = resolve_path(name, edit_dir)
        if relative.exists():
            return relative
    if p.exists():
        return p.resolve()

    # Don't trust Path.suffix here: a real camera/colorist LUT name routinely
    # contains its own dots ("...to Rec.709 V1.cube" — the file already in
    # lut/), so `Path(name).suffix` reads back ".709 V1" for the bare name
    # and wrongly looks "already extensioned". Check the actual extension set.
    candidates = [LUT_DIR / name]
    if not name.lower().endswith(LUT_EXTENSIONS):
        candidates += [LUT_DIR / f"{name}{ext}" for ext in LUT_EXTENSIONS]
    for c in candidates:
        if c.exists():
            return c

    available = (
        sorted(f.stem for f in LUT_DIR.glob("*") if f.suffix.lower() in LUT_EXTENSIONS)
        if LUT_DIR.is_dir() else []
    )
    hint = (
        f" Available in lut/: {', '.join(available)}" if available
        else f" lut/ is empty — drop a .cube file there (see lut/README.md)."
    )
    sys.exit(f"LUT not found: '{name}'.{hint}")


def lut_filter_string(lut_path: Path) -> str:
    """Build the `lut3d=file='...'` filter segment, Windows-path-safe.

    Same escaping as the subtitles filter below: forward slashes (a backslash
    is an escape character inside a filtergraph, so a native Windows path
    breaks this the same way it breaks `subtitles=`), then escape the drive
    colon and any single quotes.
    """
    abs_path = (
        str(lut_path.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )
    return f"lut3d=file='{abs_path}'"


def resolve_lut_filter(lut_field: str | None, edit_dir: Path | None = None) -> str:
    """CLI --lut or the EDL's 'lut' field -> the ffmpeg filter string, or ""."""
    if not lut_field:
        return ""
    return lut_filter_string(find_lut_file(lut_field, edit_dir))


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def resolve_edl_asset_path(maybe_path: str, edit_dir: Path) -> Path | None:
    """Resolve a path the EDL references (overlay file, subtitles) that may be
    written two ways: relative to the EDL's own directory, or relative to the
    project root with a literal "edit/" prefix — what it looks like from
    <videos_dir>, which is naturally how an agent writes it (the EDL format
    example in SKILL.md itself does this: "edit/animations/slot_1/render.mp4").
    Since `edit_dir` IS already <videos_dir>/edit, resolving that string
    edit_dir-relative alone would double up to ".../edit/edit/...". Try both,
    return whichever exists; None if neither does.
    """
    candidates = [resolve_path(maybe_path, edit_dir)]
    if not Path(maybe_path).is_absolute():
        candidates.append(Path(maybe_path).resolve())
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_subtitles_path(maybe_path: str, edit_dir: Path) -> Path:
    """Resolve the EDL's subtitles path. A missing file is an error: rendering
    on without it silently ships a video with no captions."""
    found = resolve_edl_asset_path(maybe_path, edit_dir)
    if found is not None:
        return found
    tried = ", ".join(
        str(c) for c in (resolve_path(maybe_path, edit_dir), Path(maybe_path).resolve())
    )
    sys.exit(f"subtitles file in EDL not found (tried {tried}). Fix the path or pass --no-subtitles.")


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------
#
# iPhone defaults to HLG HDR in Rec.2020 (and many mirrorless cameras ship PQ).
# If the source is HDR and we only downconvert bit depth (yuv420p10le → yuv420p)
# without tone-mapping, the output is 8-bit but still carries HLG/PQ transfer
# metadata. Players that honor the metadata (screen recorders, most social
# upload re-encodes) interpret 8-bit values in an HDR container and the result
# looks oversaturated / blown out. QuickTime on macOS can hide this locally —
# screen recording and uploaded renders cannot.
#
# Fix: detect HDR via color_transfer and prepend a zscale+tonemap chain to the
# vf graph so the output is clean Rec.709 SDR.

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    """Return True if the source uses a PQ or HLG transfer function."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() in HDR_TRANSFERS
    except subprocess.CalledProcessError:
        return False


def is_portrait_source(video: Path) -> bool:
    """Return True if the displayed video is portrait, including rotation."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=width,height:stream_side_data=rotation",
             "-of", "json", str(video)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(out.stdout).get("streams") or []
        if not streams:
            return False
        stream = streams[0]
        w, h = int(stream["width"]), int(stream["height"])

        # ffmpeg autorotates display-matrix side data before applying filters.
        # Swap coded dimensions for quarter-turns so the scale axis is selected
        # from the dimensions the filter actually sees. A plain metadata tag is
        # intentionally ignored because it does not guarantee autorotation.
        rotation = 0
        for side_data in stream.get("side_data_list") or []:
            if side_data.get("rotation") is not None:
                rotation = side_data["rotation"]
                break
        if int(round(float(rotation))) % 360 in (90, 270):
            w, h = h, w
        return h > w
    except (
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        OSError,
        OverflowError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False


def parse_fps(value: str) -> str:
    """Validate and canonicalize an ffmpeg frame rate."""
    text = value.strip()
    if len(text) > 32 or not re.fullmatch(
        r"(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+)", text
    ):
        raise argparse.ArgumentTypeError(
            "FPS must be a positive number or rational, e.g. 30 or 30000/1001"
        )
    try:
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            "FPS must be a positive number or rational, e.g. 30 or 30000/1001"
        ) from exc
    if rate <= 0:
        raise argparse.ArgumentTypeError("FPS must be greater than zero")
    # FFmpeg stores video rates as AVRational (signed 32-bit components).
    # Bounding the reduced fraction keeps every accepted canonical value safe
    # for ffmpeg and makes parse_fps(parse_fps(value)) idempotent.
    max_component = 2_147_483_647
    if rate.numerator > max_component or rate.denominator > max_component:
        raise argparse.ArgumentTypeError("FPS precision or magnitude is too large")
    return f"{rate.numerator}/{rate.denominator}"


def probe_source_fps(video: Path) -> str | None:
    """Return an ffmpeg-ready source rate, preferring the average frame rate.

    ``avg_frame_rate`` represents the observed average and is the better default
    for variable-frame-rate inputs. ``r_frame_rate`` remains a fallback for
    streams where the average is unavailable. Values are normalized to an exact
    rational so rates such as ``30000/1001`` survive without rounding.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-of", "json", str(video)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(out.stdout).get("streams") or []
        if not streams:
            return None
        for field in ("avg_frame_rate", "r_frame_rate"):
            value = streams[0].get(field)
            if value and value != "0/0":
                try:
                    return parse_fps(value)
                except argparse.ArgumentTypeError:
                    continue
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None
    return None


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    rate: str | None = None,
    lut_filter: str = "",
) -> None:
    """Extract a cut range as its own MP4 with grade + LUT + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale to 1080p from 4K.
    Portrait sources (height > width) are scaled by height to preserve orientation.

    Quality ladder:
      - final (default): 1080p libx264 fast CRF 20
      - preview:         1080p libx264 medium CRF 22 (evaluable for QC)
      - draft:           720p libx264 ultrafast CRF 28 (cut-point check only)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    portrait = is_portrait_source(source)
    if draft:
        scale = "scale=-2:1280" if portrait else "scale=1280:-2"
    else:
        scale = "scale=-2:1920" if portrait else "scale=1920:-2"

    vf_parts: list[str] = []
    if is_hdr_source(source):
        vf_parts.append(TONEMAP_CHAIN)
    vf_parts.append(scale)
    if grade_filter:
        vf_parts.append(grade_filter)
    # LUT last: a corrective grade runs first, the creative "look" on top of it —
    # same order a colorist layers a corrective node under a creative node.
    if lut_filter:
        vf_parts.append(lut_filter)
    vf = ",".join(vf_parts)

    # 30ms audio fades at both edges (Rule 3) — prevent pops
    fade_out_start = max(0.0, duration - 0.03)
    af = f"afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_start:.3f}:d=0.03"

    if draft:
        preset, crf = "ultrafast", "28"
    elif preview:
        preset, crf = "medium", "22"
    else:
        preset, crf = "fast", "20"

    # Frame rate: use the rate the caller resolved once for the whole render
    # (every segment must share it — concat -c copy in Rule 2 requires a uniform
    # frame rate). When called standalone with no rate, preserve this source's
    # own rate; fall back to 24 only if it can't be probed.
    out_rate = rate if rate is not None else (probe_source_fps(source) or "24")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-pix_fmt", "yuv420p", "-r", out_rate,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
    fps: str | None = None,
    lut: str | None = None,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.

    `lut`: CLI --lut value, or None to fall back to the EDL's own "lut" field.
    A LUT is global to the render (one "look"), unlike grade which can be
    "auto" per segment — so it's resolved once here, not per-range.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    lut_filter = resolve_lut_filter(lut or edl.get("lut"), edit_dir)
    if lut_filter:
        print(f"  lut: {(lut or edl.get('lut'))}")
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]

    # Resolve ONE output frame rate for the entire render and apply it to every
    # segment. The lossless concat (Rule 2, `-c copy`) requires all segments to
    # share a frame rate; probing per-segment would diverge for multi-source
    # EDLs that mix rates (e.g. a 30fps and a 60fps source) and break the concat.
    # Explicit --fps wins; otherwise preserve the first source's rate.
    if fps is not None:
        out_rate = parse_fps(str(fps))
    elif ranges:
        first_src = resolve_path(sources[ranges[0]["source"]], edit_dir)
        out_rate = probe_source_fps(first_src) or "24"
    else:
        out_rate = "24"

    seg_paths: list[Path] = []
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/  @ {out_rate} fps"
          f"{' (forced)' if fps is not None else ' (from source)'}")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        note = r.get("beat") or r.get("note") or ""
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  ({duration:5.2f}s)  {note}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(src_path, start, duration, seg_filter, out_path, preview=preview, draft=draft, rate=out_rate, lut_filter=lut_filter)
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments(segment_paths: list[Path], out_path: Path, edit_dir: Path) -> None:
    """Lossless concat via the concat demuxer. No re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_NONSPEECH_TOKEN = re.compile(r"^[\[\(<].*[\]\)>][.,;:!?]*$")


def is_nonspeech_token(text: str) -> bool:
    """True for a bracketed transcription artifact: [UH], [UM], (laughter).

    A verbatim ASR writes fillers as bracket tags, which is exactly what makes
    them findable when deciding a cut — but they are notation, not speech, and
    burning "[UH]" onto the picture is a bug. Subtitling convention omits
    fillers anyway.
    """
    return bool(_NONSPEECH_TOKEN.match(text.strip()))


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    """Caption-bearing words overlapping [t_start, t_end).

    Drops bracketed non-speech tokens, so this is the caption view of the
    transcript — not the cut view. Read `transcript["words"]` directly when
    picking cut edges, where a filler's timing is the whole point.
    """
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        if is_nonspeech_token(w.get("text") or ""):
            continue
        out.append(w)
    return out


CHUNK_WORDS = 2         # target words per cue
CHUNK_MAX_WORDS = 3     # a too-short chunk may grow to this many words
CHUNK_MIN_S = 0.35      # a cue shorter than this reads as a flash
CHUNK_PAUSE_S = 0.3     # a gap this long between words ends the cue


def chunk_words(words: list[dict]) -> list[list[dict]]:
    """Group transcript words into caption cues.

    A cue closes on trailing punctuation or on a pause before the next word.
    Otherwise it closes at CHUNK_WORDS words, unless it would be on screen for
    less than CHUNK_MIN_S; then it takes up to CHUNK_MAX_WORDS words. So
    "does | is" across a pause stays split, and fast "what a" does not flash.
    """
    words = [w for w in words if (w.get("text") or "").strip()]
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        text = w["text"].strip()
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        dur = w["end"] - current[0]["start"]
        if (
            nxt is None
            or text[-1] in PUNCT_BREAK
            or gap >= CHUNK_PAUSE_S
            or len(current) >= CHUNK_MAX_WORDS
            or (len(current) >= CHUNK_WORDS and dur >= CHUNK_MIN_S)
        ):
            chunks.append(current)
            current = []
    return chunks


def build_master_srt(
    edl: dict,
    edit_dir: Path,
    out_path: Path,
    profanity_matcher: re.Pattern | None = None,
) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - phrase-aware ~2-word chunks (see chunk_words)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    - If `profanity_matcher` is given, each matched word is replaced with
      literal "***" before joining — see helpers/profanity.py. This is the
      caption side of --censor-profanity; find_censor_ranges() below is the
      audio side, computed independently so audio stays censored even when
      the EDL points at a hand-written subtitles file instead of this one.
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping captions for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text())
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        for chunk in chunk_words(words_in_seg):
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            words_text = [(w.get("text") or "").strip() for w in chunk]
            if profanity_matcher is not None:
                words_text = [censor_word(t, profanity_matcher) for t in words_text]
            text = " ".join(words_text)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Profanity audio censoring (--censor-profanity) --------------------
#
# Pad is deliberately independent of Hard Rule 7's cut-edge padding (30-200ms):
# that padding absorbs timestamp *drift* at a cut boundary; this padding exists
# so a mute/bleep fully covers the word's actual audio even if the aligned
# start/end is a touch tight, without eating into the neighboring word.
CENSOR_PAD_S = 0.06
CENSOR_BLEEP_HZ = 1000
# ffmpeg's lavfi `sine` source has no amplitude option in this build and emits
# a fixed, quiet ~0.125 peak — NOT full-scale. volume=4.0 on top of that lands
# the tone around 0.5 peak / ~0.35 RMS, clearly audible over dialogue (RMS
# ~0.15-0.2 in typical speech) without clipping. Measured empirically — see
# tests/test_lut_and_censor.py — because ffmpeg's docs don't state the default.
CENSOR_BLEEP_GAIN = 4.0


def find_censor_ranges(
    edl: dict,
    edit_dir: Path,
    matcher: re.Pattern,
    pad: float = CENSOR_PAD_S,
) -> list[tuple[float, float, str]]:
    """Output-timeline (start, end, word) for every profane word, padded.

    Walks the same per-range transcript lookup as build_master_srt, but at
    word granularity rather than caption-chunk granularity — a censor
    mute/bleep must cover exactly the word's audio, not the whole caption cue
    it happens to share the screen with. Independent of whether captions are
    being built at all, so audio stays censored even with --no-subtitles or
    an EDL-provided subtitles file this function never reads.
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]

    ranges_out: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if tr_path.exists():
            transcript = json.loads(tr_path.read_text())
            for w in _words_in_range(transcript, seg_start, seg_end):
                text = (w.get("text") or "").strip()
                if not is_profane(text, matcher):
                    continue
                local_start = max(seg_start, w.get("start", seg_start))
                local_end = min(seg_end, w.get("end", seg_end))
                out_start = max(0.0, local_start - seg_start) + seg_offset - pad
                out_end = max(0.0, local_end - seg_start) + seg_offset + pad
                ranges_out.append((max(0.0, out_start), out_end, text))

        seg_offset += seg_duration

    ranges_out.sort(key=lambda t: t[0])
    return ranges_out


def build_censor_enable_expr(ranges: list[tuple[float, float, str]]) -> str:
    """ffmpeg `enable` boolean expression (unquoted): true while `t` is in ANY range.

    `between(t,a,b)` is 0/1 per range; summed with `+` so ffmpeg's `enable`
    (any nonzero = on) treats overlap or back-to-back ranges correctly without
    needing to merge them first. The caller wraps this in single quotes —
    same as the proven `overlay=enable='between(t,{t},{end})'` pattern a few
    hundred lines down — so the commas need no escaping of their own.
    """
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, _ in ranges) or "0"


# -------- Loudness normalization (social-ready audio) -----------------------


# Social-media standard: -14 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# Matches YouTube / Instagram / TikTok / X / LinkedIn normalization targets.
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
    censor_ranges: list[tuple[float, float, str]] | None = None,
    censor_mode: str = "bleep",
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → out.

    If there are no overlays, no subtitles, and nothing to censor, just copy
    base to out.

    `censor_ranges` (from find_censor_ranges) silences or bleeps the audio
    over each (start, end) window on the OUTPUT timeline:
      - "mute":  the original audio is volume-gated to 0 during each window.
      - "bleep": the original audio is gated to 0 AND a 1kHz tone (gated ON
        only during those same windows) is mixed in — a true replacement,
        not just silence. `amix ... normalize=0` keeps this lossless: the two
        inputs never overlap in time (one is silent exactly where the other
        isn't), so summing them unscaled neither attenuates the untouched
        audio nor needs headroom.
    Either way this forces a real audio encode — `-c:a copy` cannot apply a
    filter — so censoring, unlike plain overlays/subtitles, cannot take the
    early-return copy path below even with nothing else to composite.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()
    censor_ranges = censor_ranges or []
    has_censor = bool(censor_ranges)

    if not has_overlays and not has_subs and not has_censor:
        # Nothing to do — just rename/copy base to final name
        run(["ffmpeg", "-y", "-i", str(base_path), "-c", "copy", str(out_path)], quiet=True)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        # Same dual-candidate resolution as subtitles — was plain resolve_path()
        # before, which silently mis-resolved the exact path shape the EDL
        # format example in SKILL.md itself uses ("edit/animations/.../render.mp4"
        # doubling up against edit_dir already being ".../edit"), surfacing only
        # as an opaque ffmpeg "No such file" minutes into a render.
        ov_path = resolve_edl_asset_path(ov["file"], edit_dir)
        if ov_path is None:
            tried = ", ".join(
                str(c) for c in (resolve_path(ov["file"], edit_dir), Path(ov["file"]).resolve())
            )
            sys.exit(f"overlay file in EDL not found: '{ov['file']}' (tried {tried})")
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        filter_parts.append(f"[{idx}:v]setpts=PTS-STARTPTS+{t}/TB[a{idx}]")

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[a{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        # Forward slashes even on Windows: a backslash is an escape character inside
        # a filtergraph, so a native "C:\Users\..." path makes ffmpeg fail every
        # subtitle burn. ffmpeg accepts "/" on Windows, so normalize before escaping
        # the drive colon.
        subs_abs = (
            str(subtitles_path.resolve())
            .replace("\\", "/")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )
        filter_parts.append(
            f"{current}subtitles='{subs_abs}':force_style='{SUB_FORCE_STYLE}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            # No video filtering at all (censor-only call: no overlays, no
            # subs). Bracketed "[0:v]" is a filtergraph PAD reference — there
            # is no such pad here, only the audio chain below populates
            # filter_complex — so -map needs the raw stream specifier "0:v"
            # (no brackets), same shape as audio_label's "0:a" default below.
            out_label = "0:v"

    # Audio: profanity censor, if requested. Comes after the video filter_parts
    # are finalized above so appending here doesn't disturb any index math for
    # the video chain; audio and video are independent chains in one graph.
    audio_label = "0:a"
    audio_codec_args = ["-c:a", "copy"]
    if has_censor:
        enable_expr = build_censor_enable_expr(censor_ranges)
        if censor_mode == "mute":
            filter_parts.append(f"[0:a]volume=enable='{enable_expr}':volume=0[outa]")
        else:  # bleep — replace, don't just silence
            sine_idx = 1 + len(overlays)
            inputs += ["-f", "lavfi", "-i", f"sine=frequency={CENSOR_BLEEP_HZ}:sample_rate=48000"]
            filter_parts.append(f"[0:a]volume=enable='{enable_expr}':volume=0[muted]")
            filter_parts.append(
                f"[{sine_idx}:a]volume=enable='{enable_expr}':volume={CENSOR_BLEEP_GAIN}[bleep]"
            )
            filter_parts.append("[muted][bleep]amix=inputs=2:duration=first:normalize=0[outa]")
        audio_label = "[outa]"
        audio_codec_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", audio_label,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        *audio_codec_args,
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}"
          + (f", censored: {len(censor_ranges)} word(s) ({censor_mode})" if has_censor else ""))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p, medium, CRF 22 — evaluable for QC, faster than final.",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-14 LUFS, -1 dBTP, LRA 11).",
    )
    ap.add_argument(
        "--fps",
        type=parse_fps,
        default=None,
        help="Output frame rate. Default: preserve the source's frame rate "
             "(falls back to 24 if it can't be probed). Pass e.g. --fps 30 or "
             "--fps 30000/1001 to force.",
    )
    ap.add_argument(
        "--lut",
        type=str,
        default=None,
        help="Apply a 3D LUT by name from the lut/ folder (e.g. --lut "
             "kodak_2383 for lut/kodak_2383.cube), or a path to a .cube file. "
             "Overrides the EDL's 'lut' field. Runs after any grade filter.",
    )
    ap.add_argument(
        "--censor-profanity",
        action="store_true",
        help="Redact profanity: captions show *** (only when generated by "
             "--build-subtitles — an EDL-provided subtitles file is left "
             "as-is), audio is bleeped. Wordlist: profanity/wordlist.txt, "
             "or --censor-wordlist.",
    )
    ap.add_argument(
        "--censor-wordlist",
        type=Path,
        default=None,
        help="Custom profanity wordlist (one stem per line). "
             "Default: profanity/wordlist.txt at the repo root.",
    )
    ap.add_argument(
        "--censor-mute",
        action="store_true",
        help="Silence profanity instead of bleeping it. Only with --censor-profanity.",
    )
    args = ap.parse_args()

    if args.censor_mute and not args.censor_profanity:
        ap.error("--censor-mute requires --censor-profanity")

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft, fps=args.fps, lut=args.lut
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    concat_segments(segment_paths, base_path, edit_dir)

    # 2b. Profanity: resolve the matcher once, find every flagged word's
    # output-timeline position. Independent of subtitle building, so audio
    # stays censored even with --no-subtitles.
    profanity_matcher = None
    censor_ranges: list[tuple[float, float, str]] = []
    if args.censor_profanity:
        stems = load_profanity_stems(args.censor_wordlist)
        profanity_matcher = build_profanity_matcher(stems)
        if profanity_matcher is None:
            print(f"warning: profanity wordlist is empty "
                  f"({args.censor_wordlist or 'profanity/wordlist.txt'}) — nothing to censor")
        else:
            censor_ranges = find_censor_ranges(edl, edit_dir, profanity_matcher)
            flagged = sorted({w for _, _, w in censor_ranges})
            print(f"profanity filter: {len(censor_ranges)} word(s) flagged"
                  + (f" — {', '.join(flagged)}" if flagged else ""))

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path, profanity_matcher=profanity_matcher)
        elif edl.get("subtitles"):
            subs_path = resolve_subtitles_path(edl["subtitles"], edit_dir)
            if args.censor_profanity:
                print("  note: EDL-provided subtitles are not generated here, so "
                      "--censor-profanity cannot redact them — only the audio "
                      "will be censored. Use --build-subtitles to censor captions too.")

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    censor_mode = "mute" if args.censor_mute else "bleep"
    if args.no_loudnorm:
        # Composite directly to final output
        build_final_composite(base_path, overlays, subs_path, out_path, edit_dir,
                               censor_ranges=censor_ranges, censor_mode=censor_mode)
    else:
        # Composite to a temp file, then run loudnorm → final output
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(base_path, overlays, subs_path, tmp_composite, edit_dir,
                               censor_ranges=censor_ranges, censor_mode=censor_mode)
        print("loudness normalization → social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
