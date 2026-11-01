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


def probe_display_dimensions(video: Path) -> tuple[int, int] | None:
    """(width, height) as actually displayed, i.e. after rotation-side-data
    normalization. Shared by is_portrait_source and probe_source_long_edge so
    the rotation-swap logic exists in exactly one place.
    """
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
            return None
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
        return w, h
    except (
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        OSError,
        OverflowError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def is_portrait_source(video: Path) -> bool:
    """Return True if the displayed video is portrait, including rotation."""
    dims = probe_display_dimensions(video)
    return dims is not None and dims[1] > dims[0]


def probe_source_long_edge(video: Path) -> int | None:
    """The longer display edge in pixels, rotation-aware. This is what the
    `width` parameter through extract_segment actually targets: scale WIDTH
    for a landscape source, scale HEIGHT for a portrait one — "long edge" is
    the one number both cases share, so it's the natural unit for "how big
    is this source" regardless of orientation.
    """
    dims = probe_display_dimensions(video)
    return max(dims) if dims else None


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


def probe_duration(video: Path) -> float | None:
    """Container duration in seconds, or None if it can't be probed."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        )
        text = out.stdout.strip()
        return float(text) if text else None
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def clamp_range_to_source(
    source: Path,
    start: float,
    end: float,
    fps: str | None = None,
    label: str = "",
) -> tuple[float, float]:
    """Clamp a [start, end) range so it never reads past the source's last frame.

    Word-boundary padding (Hard Rule 7) plus frame rounding can push `end`
    a touch past the file's real duration — harmless for `-t`-bounded reads
    (ffmpeg just truncates), but it silently shorts the segment: on a short
    clip (a title/end card is the case that actually bit this) the shortfall
    can be the whole clip, read as "this section never rendered." Clamping
    here, once, before extraction, is cheaper and more honest than debugging
    a missing segment after the fact.

    `fps` is this SOURCE's own margin, not the sequence rate: on a mixed-rate
    EDL (a 47.952fps camera cut next to a 30fps screen recording is the case
    this was found on) "2 frames of margin" means something different per
    file. Omit it to probe the source's own rate automatically.
    """
    duration = probe_duration(source)
    if duration is None:
        return start, end
    rate_str = fps or probe_source_fps(source) or "24"
    rate = Fraction(rate_str)
    limit = duration - 2 / float(rate)  # never the source's last frame or beyond
    if end > limit:
        print(f"  warning: {label or source.name} range end {end:.3f}s exceeds "
              f"source duration {duration:.3f}s — clamped to {max(start, limit):.3f}s")
        end = max(start, limit)
    return start, end


def clamp_edl_ranges(edl: dict, edit_dir: Path) -> int:
    """Clamp every EDL range's `end` to its own source's duration, IN PLACE.

    Mutates `edl["ranges"]` rather than returning a copy so every later
    consumer of this same EDL object — extract_all_segments, build_master_srt,
    find_censor_ranges, export_timeline.py's build_timeline — sees identical,
    already-clamped numbers. Call this once, right after loading the EDL,
    before anything else reads `ranges`: clamping deep inside one consumer
    only fixes what that one consumer renders, while every other output
    (captions, the exported XML, the reported total duration) silently keeps
    disagreeing with it.

    Returns how many ranges were clamped.
    """
    sources = edl.get("sources") or {}
    clamped = 0
    for r in edl.get("ranges") or []:
        src_path = resolve_path(sources[r["source"]], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        new_start, new_end = clamp_range_to_source(
            src_path, start, end, label=r.get("beat") or r["source"],
        )
        if new_end != end:
            r["end"] = new_end
            clamped += 1
    return clamped


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


DEFAULT_WIDTH = 1920  # last-resort fallback only — see probe_max_resolution below;
                       # render.py's own CLI default is the source's own max resolution, not this


def probe_max_resolution(edl: dict, edit_dir: Path) -> tuple[int, dict[str, int]]:
    """The long edge (pixels) of the HIGHEST-resolution source actually used
    in edl["ranges"], plus a per-source breakdown for the caller to print.

    This is render.py's real default output size when neither --width nor
    --res is given: previously a flat 1920 (1080p) regardless of what the
    footage actually was, silently downscaling 4K source and silently
    upscaling anything smaller. "Resolution" here means the per-source LONG
    EDGE (rotation-aware) — the same unit extract_segment's own `width`
    parameter already targets for both landscape and portrait sources — so a
    mix of orientations compares correctly without a separate portrait case.

    Falls back to DEFAULT_WIDTH only if nothing in the EDL could be probed
    (e.g. an empty EDL, or every source is unreadable).
    """
    sources = edl.get("sources") or {}
    used_names = {r["source"] for r in (edl.get("ranges") or [])}
    per_source: dict[str, int] = {}
    for name in used_names:
        if name not in sources:
            continue
        path = resolve_path(sources[name], edit_dir)
        edge = probe_source_long_edge(path)
        if edge:
            per_source[name] = edge
    if not per_source:
        return DEFAULT_WIDTH, per_source
    return max(per_source.values()), per_source


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
    width: int = DEFAULT_WIDTH,
) -> None:
    """Extract a cut range as its own MP4 with grade + LUT + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale from source resolution
    to `width` — render.py's CLI resolves this to the EDL's own max native
    resolution by default (see probe_max_resolution), or `--res`/`--width`
    if given; this function's own `width=DEFAULT_WIDTH` fallback only
    applies to a caller that doesn't pass one explicitly. Portrait sources
    (height > width) are scaled by height instead, to preserve orientation.

    Frame-exact by construction, not by luck: `-t` alone lets ffmpeg round the
    VIDEO stream up to the next whole frame while the AUDIO stream truncates
    to the exact requested duration — on their own that's sub-frame, but
    concat is lossless (`-c copy`, Rule 2) and never re-aligns anything, so
    the mismatch is pure addition across segments. On a real 50-cut timeline
    this drifted video ~1s ahead of audio by the end, and master.srt (built
    from the EDL's own seconds) drifted against the rendered MP4 right along
    with it. Fix: quantize `duration` to a whole number of output frames
    BEFORE building the command, pin the video stream to exactly that frame
    count (`-frames:v`, not `-t`), and force the audio to the same quantized
    duration explicitly (`apad,atrim`) before the fades run on top of it.

    Quality ladder — resolution (the `width` parameter) is the SAME across
    final/preview, only the encode settings change; draft always drops to a
    fixed 1280 regardless of `width`, since it's a cut-point check, not a
    preview of the delivery resolution:
      - final (default): libx264 fast CRF 20, at `width` (source's own max
                          native resolution unless --res/--width override it)
      - preview:          libx264 medium CRF 22, same `width` as final
      - draft:             libx264 ultrafast CRF 28, fixed 1280 (cut-point check only)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    portrait = is_portrait_source(source)
    draft_width = 1280  # fixed regardless of --res — draft is a speed tool, not a preview of the delivery resolution
    target_width = draft_width if draft else width
    scale = (
        f"scale=-2:{target_width}:flags=lanczos" if portrait
        else f"scale={target_width}:-2:flags=lanczos"
    )

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
    vf_parts.append("format=yuv420p")
    vf = ",".join(vf_parts)

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
    fps = Fraction(out_rate)

    # Quantize to a whole number of OUTPUT frames. nfr is what actually gets
    # encoded; `duration` is updated to match it exactly so every downstream
    # consumer (the afade timing below, the audio apad/atrim, any caller
    # accumulating seg_offset for captions/XML) works from the same number.
    nfr = max(1, round(duration * float(fps)))
    duration = nfr / float(fps)
    fade_out_start = max(0.0, duration - 0.03)

    af = (
        # 48kHz stereo first so apad/atrim/afade all operate on a known,
        # consistent format regardless of the source's own audio layout.
        "aresample=48000,aformat=channel_layouts=stereo,"
        # Force the EXACT quantized duration — apad covers a source that
        # undershoots (silence, not wrong-length fades), atrim covers a
        # source that overshoots (ffmpeg's own audio decode is sample-exact,
        # but the +0.5s input overshoot below reads well past `duration`).
        f"apad,atrim=0:{duration:.6f},"
        f"afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_start:.3f}:d=0.03"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        # Read past the quantized duration — -frames:v below is what actually
        # bounds the video output; this just guarantees enough source decoded
        # to reach that frame count without an early EOF truncating it short.
        "-t", f"{duration + 0.5:.3f}",
        # Explicit stream selection: a source can carry a data/tmcd (timecode)
        # stream alongside video+audio (common from cameras and OBS alike),
        # which a bare `ffmpeg -i ... -c:v ... -c:a ...` command happily
        # passes through into the segment. That stray stream can confuse a
        # later `-c copy` concat or an NLE import. -map pins exactly
        # video+audio; -write_tmcd 0 and -map_metadata -1 stop ffmpeg from
        # synthesizing or carrying forward timecode/metadata of its own.
        "-map", "0:v:0", "-map", "0:a:0", "-write_tmcd", "0", "-map_metadata", "-1",
        "-vf", vf,
        "-frames:v", str(nfr),
        "-af", af,
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-pix_fmt", "yuv420p", "-r", out_rate,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
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
    width: int = DEFAULT_WIDTH,
) -> tuple[list[Path], str]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns (the ordered list of segment paths, the resolved output fps
    string) — the caller needs the rate too, e.g. to size a post-render
    duration-drift tolerance in whole frames rather than a guessed constant.

    Assumes ranges are already clamped to their sources — call
    clamp_edl_ranges(edl, edit_dir) once before this, not per-call here, so
    every other consumer of the same EDL (captions, censor ranges, the XML
    export) agrees with what actually got rendered.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.

    LUT resolution, most specific wins: a range's own `"lut"` field, else
    `edl["luts"][source_name]`, else `lut` (CLI --lut) or the EDL's top-level
    `"lut"`. A single global LUT breaks the moment one source is log footage
    and another is a screen recording — those need different (or no) LUTs on
    the same timeline.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    global_lut = lut or edl.get("lut")
    per_source_luts: dict = edl.get("luts") or {}
    lut_filter_cache: dict[str, str] = {}

    def lut_filter_for(r: dict, src_name: str) -> str:
        name = r.get("lut") or per_source_luts.get(src_name) or global_lut
        if not name:
            return ""
        if name not in lut_filter_cache:
            lut_filter_cache[name] = resolve_lut_filter(name, edit_dir)
            print(f"  lut ({src_name}): {name}")
        return lut_filter_cache[name]

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
    print(f"extracting {len(ranges)} segment(s) -> {clips_dir.name}/  @ {out_rate} fps, {width}w"
          f"{' (forced fps)' if fps is not None else ' (fps from source)'}")
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
        seg_lut_filter = lut_filter_for(r, src_name)

        note = r.get("beat") or r.get("note") or ""
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  ({duration:5.2f}s)  {note}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        extract_segment(
            src_path, start, duration, seg_filter, out_path,
            preview=preview, draft=draft, rate=out_rate,
            lut_filter=seg_lut_filter, width=width,
        )
        seg_paths.append(out_path)

    return seg_paths, out_rate


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
    print(f"concat -> {out_path.name}")
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


# -------- "natural" subtitle style (sentence case, readable, no burn) -------
#
# bold-overlay (above) is a deliberate style: short, punchy, UPPERCASE,
# built to be burned into frame. It reads badly as a plain SRT a viewer
# toggles on in a native player — this style is for that case instead:
# sentence case, readable line lengths, cues that don't flash or overlap.

NATURAL_MAX_CHARS = 42
NATURAL_MAX_WORDS = 8
NATURAL_PAUSE_S = 0.45
NATURAL_MIN_CUE_S = 1.0


def chunk_words_natural(
    words: list[dict],
    max_chars: int = NATURAL_MAX_CHARS,
    max_words: int = NATURAL_MAX_WORDS,
    pause_s: float = NATURAL_PAUSE_S,
) -> list[list[dict]]:
    """Group words into reading-length cues: break on sentence punctuation,
    a pause >= pause_s, or hitting the word/character cap — whichever comes
    first. Unlike chunk_words (2-3 words, built to flash fast under a burned
    caption), this optimizes for a cue a viewer can actually read as text."""
    words = [w for w in words if (w.get("text") or "").strip()]
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        text = w["text"].strip()
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        joined_len = len(" ".join(x["text"].strip() for x in current))
        next_len = joined_len + 1 + len(nxt["text"].strip()) if nxt else joined_len
        if (
            nxt is None
            or text[-1:] in ".?!"
            or gap >= pause_s
            or len(current) >= max_words
            or next_len > max_chars
        ):
            chunks.append(current)
            current = []
    return chunks


def load_text_fixes(path: Path | None) -> list[tuple[str, str]]:
    """User-supplied ASR correction list for cue text: one `pattern<TAB>
    replacement` per line (`#` comments, blank lines skipped), applied as
    regex substitutions in order. Real transcripts mishear specific proper
    nouns/jargon consistently (a product name, a technical term) — this is
    how a project fixes "that's our effect" -> "that's our copy" without
    re-transcribing or patching the raw word data itself."""
    if path is None:
        return []
    if not path.exists():
        sys.exit(f"--text-fixes file not found: {path}")
    fixes: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip("\n")
        if not line.strip():
            continue
        if "\t" not in line:
            sys.exit(f"--text-fixes line has no tab separator: {line!r} "
                      f"(expected 'pattern<TAB>replacement')")
        pattern, replacement = line.split("\t", 1)
        if pattern.strip():
            fixes.append((pattern.strip(), replacement.strip()))
    return fixes


def _apply_text_fixes(text: str, fixes: list[tuple[str, str]]) -> str:
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    return text


def build_master_srt(
    edl: dict,
    edit_dir: Path,
    out_path: Path,
    profanity_matcher: re.Pattern | None = None,
    style: str = "bold-overlay",
    text_fixes: list[tuple[str, str]] | None = None,
) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    style="bold-overlay" (default): phrase-aware ~2-word chunks (chunk_words),
      UPPERCASE — built to be burned into frame via --build-subtitles.
    style="natural": reading-length cues (chunk_words_natural), sentence
      case only at sentence starts, cues stretched toward a 1s minimum
      without overlapping the next one — built to ship as a plain sidecar
      .srt (pair with --srt-only) for a native player's own caption renderer.

    Output times computed as word.start - segment_start + segment_offset.
    If `profanity_matcher` is given, each matched word is replaced with
    literal "***" before joining — see helpers/profanity.py. This is the
    caption side of --censor-profanity; find_censor_ranges() below is the
    audio side, computed independently so audio stays censored even when
    the EDL points at a hand-written subtitles file instead of this one.
    `text_fixes` (see load_text_fixes) are applied after profanity censoring,
    before casing — regex substitutions in cue text, e.g. ASR corrections.
    """
    transcripts_dir = edit_dir / "transcripts"
    sources = edl["sources"]
    fixes = text_fixes or []
    natural = style == "natural"

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
        chunks = chunk_words_natural(words_in_seg) if natural else chunk_words(words_in_seg)

        for chunk in chunks:
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
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    entries.sort(key=lambda e: e[0])

    if natural:
        # Stretch short cues toward NATURAL_MIN_CUE_S for readability, but
        # never past the start of the next cue (no overlaps) and never past
        # a small fixed step beyond the cue's own last word.
        for i in range(len(entries)):
            a, b, t = entries[i]
            next_start = entries[i + 1][0] if i + 1 < len(entries) else b + 1.0
            if b - a < NATURAL_MIN_CUE_S:
                b = min(max(b, a + NATURAL_MIN_CUE_S), next_start - 0.02, b + 0.5)
            else:
                b = min(b, next_start - 0.02)
            entries[i] = (a, max(b, a + 0.3), t)

        # Sentence case: capitalize only right after a sentence actually
        # ended (previous cue's text ended in . ? !), not every cue.
        prev_ended_sentence = True
        for i, (a, b, t) in enumerate(entries):
            t = _apply_text_fixes(t, fixes)
            t = t.rstrip(",;: ")
            if prev_ended_sentence and t:
                t = t[0].upper() + t[1:]
            prev_ended_sentence = t[-1:] in ".?!"
            entries[i] = (a, b, t)
    else:
        for i, (a, b, t) in enumerate(entries):
            t = _apply_text_fixes(t, fixes)
            t = t.rstrip(",;:").upper()  # bold-overlay: strip trailing punctuation, UPPERCASE
            entries[i] = (a, b, t)

    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"master SRT ({style}) -> {out_path.name} ({len(entries)} cues)")


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
        print(f"  loudnorm (1-pass preview) -> {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # No true-peak re-verify here, deliberately — preview trades accuracy
        # for speed everywhere else too; the guarantee is for --preview's
        # absence (the final render), which calls verify_and_limit_true_peak
        # below after its own two-pass loudnorm.
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
    print(f"  loudnorm pass 2: normalizing -> {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    verify_and_limit_true_peak(output_path)
    return True


def measure_true_peak(path: Path) -> float | None:
    """Actual true peak of a rendered file, in dBTP, via ebur128 — independent
    of whatever loudnorm itself reported, which is a measurement of its INPUT
    and its own internal gain model, not a guarantee about its output."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # ebur128's summary prints "True peak:\n    Peak:  X.X dBFS" near the end
    # of stderr; grabbing every "Peak:" match and taking the last one is more
    # robust to exact spacing/formatting than anchoring on the section label.
    matches = re.findall(r"Peak:\s*([-\d.]+)\s*dBFS", proc.stderr)
    return float(matches[-1]) if matches else None


def verify_and_limit_true_peak(path: Path) -> None:
    """Loudnorm's own linear-mode gain can still overshoot the true-peak
    target on real content — measured on a real render: loudnorm reported
    success at I=-14 LUFS, TP=-1 dBTP, but the actual output measured 0.0
    dBFS true peak. Re-measure the real output and, if it's over target,
    apply a brickwall limiter in place rather than trust the filter's own
    claim. 192kHz oversampling before the limiter is what makes ffmpeg's
    "true" (inter-sample) peak detection meaningful — limiting at the
    original sample rate only catches peaks that land exactly on a sample.
    """
    tp = measure_true_peak(path)
    if tp is None or tp <= LOUDNORM_TP:
        return
    print(f"  true peak {tp:.1f} dBTP exceeds target {LOUDNORM_TP} dBTP — applying a limiter")
    tmp = path.with_suffix(".prelimit.mp4")
    path.rename(tmp)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(tmp),
        "-c:v", "copy",
        "-af", "aresample=192000,alimiter=limit=0.8:level=false,aresample=48000",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    tmp.unlink(missing_ok=True)
    tp_after = measure_true_peak(path)
    if tp_after is not None:
        print(f"    re-measured: {tp_after:.1f} dBTP")


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
    print(f"compositing -> {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}"
          + (f", censored: {len(censor_ranges)} word(s) ({censor_mode})" if has_censor else ""))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output video path. Required unless --list-resolutions.")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: medium preset, CRF 22, same resolution the final "
             "will ship at — evaluable for QC, faster than final.",
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
    ap.add_argument(
        "--res",
        type=int,
        choices=(1080, 2160),
        default=None,
        help="Force output resolution: 1080 or 2160 (UHD), overriding the "
             "default of the source's own max native resolution. "
             "Overridden by --width if both are given.",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Exact output width in pixels (height follows the source aspect "
             "ratio), overriding the default of the source's own max native "
             "resolution. Overrides --res.",
    )
    ap.add_argument(
        "--list-resolutions",
        action="store_true",
        help="Print each EDL source's native resolution and the long edge "
             "render.py would default to, then exit without rendering. For "
             "presenting the resolution choice to the user before committing "
             "to a render.",
    )
    ap.add_argument(
        "--srt-only",
        action="store_true",
        help="Write master.srt but don't burn it into the video. Implies "
             "--build-subtitles; use with --subtitle-style natural for a "
             "sidecar meant to be read in a native player, not burned.",
    )
    ap.add_argument(
        "--subtitle-style",
        choices=("bold-overlay", "natural"),
        default="bold-overlay",
        help="bold-overlay (default): ~2-word UPPERCASE chunks, built to be "
             "burned into frame. natural: sentence-case reading-length cues "
             "(<=42 chars/<=8 words, break on pause, 1s minimum, no overlap) "
             "— meant for --srt-only, reads badly burned in.",
    )
    ap.add_argument(
        "--text-fixes",
        type=Path,
        default=None,
        help="Caption text corrections: a file of 'pattern<TAB>replacement' "
             "regex lines (# comments ok), applied to every cue. For "
             "consistent ASR mishears of a proper noun/term specific to this "
             "project — see helpers/render.py's load_text_fixes.",
    )
    args = ap.parse_args()

    if not args.list_resolutions and args.output is None:
        ap.error("-o/--output is required unless --list-resolutions")
    if args.censor_mute and not args.censor_profanity:
        ap.error("--censor-mute requires --censor-profanity")
    if args.width is not None and args.width <= 0:
        ap.error("--width must be a positive number of pixels")
    build_subtitles = args.build_subtitles or args.srt_only
    text_fixes = load_text_fixes(args.text_fixes)

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent

    if args.list_resolutions:
        max_edge, per_source_res = probe_max_resolution(edl, edit_dir)
        if not per_source_res:
            sys.exit("no probeable sources in this EDL's ranges")
        print("source resolutions (long edge, rotation-aware):")
        for name, edge in sorted(per_source_res.items(), key=lambda kv: -kv[1]):
            marker = "  <- max, render.py's default" if edge == max_edge else ""
            print(f"  {name:20} {edge}px{marker}")
        print(f"\nOther common choices: --res 1080 (1920px), --res 2160 (3840px, UHD), "
              f"or --width N for anything else.")
        return

    out_path = args.output.resolve()

    # 0. Clamp every range to its own source's duration, in place, before
    # anything below reads edl["ranges"] — a range that runs past the end of
    # its source silently shorts that segment (the whole clip, for a short
    # title/end card), not just trims a few frames off the tail.
    clamped = clamp_edl_ranges(edl, edit_dir)
    if clamped:
        print(f"clamped {clamped} range(s) to their source's duration (see warnings above)")

    # 0b. Resolution: --width or --res, if given, are explicit and win
    # outright. Otherwise default to the ACTUAL max native resolution across
    # the EDL's sources, probed just now, not a hardcoded 1080p — this was a
    # silent, fixed downscale of 4K source (or upscale of anything smaller)
    # regardless of what the footage actually was. An agent-driven session
    # should still surface the detected per-source resolutions (--list-
    # resolutions) and ask which one the user wants (see SKILL.md "Output
    # spec") rather than relying on this default alone — this is the
    # computation behind that choice, not a replacement for asking.
    if args.width:
        width = args.width
    elif args.res:
        width = 3840 if args.res == 2160 else DEFAULT_WIDTH
    else:
        width, per_source_res = probe_max_resolution(edl, edit_dir)
        if per_source_res:
            detail = ", ".join(f"{name}={edge}px" for name, edge in sorted(per_source_res.items()))
            print(f"resolution: {width}px long edge (max across sources: {detail})")

    # 1. Extract per-segment (auto-grade per range if EDL grade is "auto")
    segment_paths, resolved_rate = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft, fps=args.fps,
        lut=args.lut, width=width,
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

    # 2a. Verify nothing silently came up short. extract_segment is
    # frame-quantized and clamp_edl_ranges already keeps every range inside
    # its source, so this should always pass — it exists as a loud, named
    # failure instead of a visually-missing section discovered by eye days
    # later (the actual way this was found: a title card and an end card
    # both silently rendered short/empty on a real project).
    expected_duration = sum(float(r["end"]) - float(r["start"]) for r in edl["ranges"])
    actual_duration = probe_duration(base_path)
    if actual_duration is not None:
        drift = abs(actual_duration - expected_duration)
        # One frame of tolerance at the rate segments were actually encoded
        # at (returned by extract_all_segments above), not a guessed constant.
        frame_s = 1 / float(Fraction(resolved_rate))
        if drift > max(frame_s, 0.05):
            sys.exit(
                f"render verification failed: base.mp4 is {actual_duration:.3f}s but "
                f"the EDL's ranges sum to {expected_duration:.3f}s (drift {drift:.3f}s). "
                f"A segment likely rendered short or a range was silently dropped — "
                f"check the per-segment log above rather than shipping this render."
            )

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

    # 3. Subtitles: build if requested, resolve final path.
    # --srt-only writes the sidecar but never sets burn_path, so step 4 below
    # never burns it — the file exists at edit_dir/master.srt either way.
    burn_path: Path | None = None
    if not args.no_subtitles:
        if build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(
                edl, edit_dir, subs_path, profanity_matcher=profanity_matcher,
                style=args.subtitle_style, text_fixes=text_fixes,
            )
            if not args.srt_only:
                burn_path = subs_path
        elif edl.get("subtitles"):
            burn_path = resolve_subtitles_path(edl["subtitles"], edit_dir)
            if args.censor_profanity:
                print("  note: EDL-provided subtitles are not generated here, so "
                      "--censor-profanity cannot redact them — only the audio "
                      "will be censored. Use --build-subtitles to censor captions too.")

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    censor_mode = "mute" if args.censor_mute else "bleep"
    if args.no_loudnorm:
        # Composite directly to final output
        build_final_composite(base_path, overlays, burn_path, out_path, edit_dir,
                               censor_ranges=censor_ranges, censor_mode=censor_mode)
    else:
        # Composite to a temp file, then run loudnorm → final output
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(base_path, overlays, burn_path, tmp_composite, edit_dir,
                               censor_ranges=censor_ranges, censor_mode=censor_mode)
        print("loudness normalization -> social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
