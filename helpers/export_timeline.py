"""Export an EDL as an editable NLE timeline — XMEML ("Final Cut Pro XML").

XMEML (<xmeml version="5">, the sequence/track/clipitem schema, file
extension .xml) is the *older* Final Cut Pro 7 interchange format — not to be
confused with true FCPXML (<fcpxml version="1.1x">, the newer Final Cut Pro X
resource/spine schema). For DaVinci Resolve + Premiere Pro specifically
(not Final Cut Pro X), XMEML is the better-proven choice: it is literally
what Premiere's own File > Export > Final Cut Pro XML command produces,
Adobe's documented recommendation for third-party round-tripping, and Resolve
has had mature native import for it for years. True FCPXML imports fine into
Resolve but Premiere's support for it is newer and less reliable beyond
simple single-track cuts.

What survives the handoff and what doesn't:

  - Cuts (EDL `ranges`)         -> V1/A1 clips on the real source files.
                                   Exact — this is what EDL/XML formats exist for.
  - Overlays                    -> a V2 track, each clip at its
                                   start_in_output/duration. Composites over
                                   V1 the same way render.py's own `overlay`
                                   filter does (opaque unless the overlay file
                                   itself carries alpha).
  - `grade` / `lut`             -> NOT applied. There is no portable, verified
                                   way to embed a third-party LUT/filter node
                                   that both Premiere's Lumetri and Resolve's
                                   color page will load identically — faking
                                   one would silently do nothing (or worse,
                                   something wrong) on import. Written instead
                                   as a <comments> note on the sequence and on
                                   every affected clip: the exact grade/LUT
                                   value, for the editor to apply by hand. If
                                   it's a LUT, both apps can load that same
                                   .cube file directly on their color page.
  - `subtitles` (SRT)           -> NOT embedded. XMEML predates any standard
                                   caption-track element both apps agree on.
                                   Left as a sidecar file instead — both apps
                                   import a plain .srt natively (File >
                                   Import, or drag onto the timeline), so
                                   nothing is lost, it's just a second file.

Usage:
    python helpers/export_timeline.py <edl.json> -o <out.xml>
    python helpers/export_timeline.py <edl.json> -o <out.xml> --fps 30
    python helpers/export_timeline.py <edl.json>   # writes <edit_dir>/timeline.xml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

from render import (  # same directory
    parse_fps,
    probe_source_fps,
    resolve_edl_asset_path,
    resolve_path,
)

try:
    from render import find_lut_file  # same directory
except Exception:  # pragma: no cover — see render.py's own grade/profanity imports
    def find_lut_file(name, edit_dir=None):  # type: ignore
        return Path(name)

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1920, 1080


# -------- Frame-rate / timecode helpers --------------------------------------


def rate_to_timebase(rate: Fraction) -> tuple[int, bool]:
    """A canonical rate (e.g. 30000/1001) -> (timebase, is_ntsc) for xmeml.

    XMEML has no concept of a fractional frame rate: NTSC rates are stored as
    their rounded integer timebase (30, not 29.97) plus an <ntsc>TRUE</ntsc>
    flag that tells the app to treat it as drop-frame-adjacent 30000/1001.
    The exact-fraction NTSC family all reduce to a denominator of 1001; any
    other rational is treated as an honest non-NTSC rate and just rounded.
    """
    if rate.denominator == 1001:
        return round(rate), True
    if rate.denominator == 1:
        return int(rate), False
    return round(rate), False


def seconds_to_frames(t: float, rate: Fraction) -> int:
    return max(0, round(t * float(rate)))


# -------- Source media probing ------------------------------------------------


def probe_dimensions(video: Path) -> tuple[int, int]:
    """Width/height of a source's first video stream. Falls back to 1080p —
    a wrong guess here only affects the <samplecharacteristics> hint in the
    XML, never the actual exported timecodes, so failing soft is correct."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(video)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(out.stdout).get("streams") or []
        if streams:
            return int(streams[0]["width"]), int(streams[0]["height"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, KeyError, ValueError):
        pass
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def path_to_pathurl(p: Path) -> str:
    """Absolute path -> the file://localhost/... form xmeml expects.

    Forward slashes even on Windows (same reasoning as render.py's filtergraph
    path escaping — the two conventions just happen to agree here), drive
    letter kept bare per the FCP7/Premiere convention, every other path
    segment percent-encoded so spaces and non-ASCII names survive XML/URI
    parsing — the DJI LUT already in lut/ has both.
    """
    resolved = p.resolve()
    parts = resolved.parts
    # Windows: ('C:\\', 'Users', 'name', ...) -> drive letter bare, rest encoded.
    # POSIX:   ('/', 'Users', 'name', ...) -> no drive segment.
    if len(parts) and parts[0].endswith(":\\"):
        drive = parts[0].rstrip("\\")
        segments = [quote(seg) for seg in parts[1:]]
        return f"file://localhost/{drive}/" + "/".join(segments)
    segments = [quote(seg) for seg in parts if seg not in ("/",)]
    return "file://localhost/" + "/".join(segments)


# -------- XML construction ----------------------------------------------------


def _text(parent: ET.Element, tag: str, value: str) -> ET.Element:
    el = ET.SubElement(parent, tag)
    el.text = value
    return el


def _rate_block(parent: ET.Element, timebase: int, ntsc: bool) -> None:
    rate = ET.SubElement(parent, "rate")
    _text(rate, "timebase", str(timebase))
    _text(rate, "ntsc", "TRUE" if ntsc else "FALSE")


def _file_block(
    parent: ET.Element,
    file_id: str,
    name: str,
    path: Path,
    src_timebase: int,
    src_ntsc: bool,
    duration_frames: int,
    width: int,
    height: int,
    first_use: bool,
) -> None:
    """A <file> element. Only the FIRST clipitem referencing a given source
    carries the full definition; every later one is a bare self-closing
    reference by the same id — the xmeml convention for "this is the same
    media", without which some importers treat each occurrence as separate,
    re-linked media."""
    if not first_use:
        ET.SubElement(parent, "file", id=file_id)
        return

    file_el = ET.SubElement(parent, "file", id=file_id)
    _text(file_el, "name", name)
    _text(file_el, "pathurl", path_to_pathurl(path))
    _rate_block(file_el, src_timebase, src_ntsc)
    _text(file_el, "duration", str(duration_frames))
    media = ET.SubElement(file_el, "media")
    video = ET.SubElement(media, "video")
    sc = ET.SubElement(video, "samplecharacteristics")
    _text(sc, "width", str(width))
    _text(sc, "height", str(height))
    _rate_block(sc, src_timebase, src_ntsc)
    audio = ET.SubElement(media, "audio")
    asc = ET.SubElement(audio, "samplecharacteristics")
    _text(asc, "depth", "16")
    _text(asc, "samplerate", "48000")
    _text(audio, "channelcount", "2")


def build_timeline(
    edl: dict,
    edit_dir: Path,
    sequence_name: str = "video-use timeline",
    fps: str | None = None,
) -> tuple[ET.Element, dict]:
    """Returns (the <xmeml> root element, a notes dict for the caller to print).

    notes = {"grade": str|None, "lut": str|None, "lut_path": Path|None,
             "subtitles": str|None, "sources_missing_media": list[str]}
    """
    sources = edl["sources"]
    ranges = edl["ranges"]
    overlays = edl.get("overlays") or []

    # Same frame-rate resolution policy as render.py's extract_all_segments,
    # imported rather than reimplemented so the XML's timeline rate can never
    # drift from what a render.py pass on the same EDL would actually produce.
    if fps is not None:
        out_rate_str = parse_fps(str(fps))
    elif ranges:
        first_src = resolve_path(sources[ranges[0]["source"]], edit_dir)
        out_rate_str = probe_source_fps(first_src) or "24"
    else:
        out_rate_str = "24"
    seq_rate = Fraction(out_rate_str)
    seq_timebase, seq_ntsc = rate_to_timebase(seq_rate)

    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    if ranges:
        first_src = resolve_path(sources[ranges[0]["source"]], edit_dir)
        width, height = probe_dimensions(first_src)

    notes = {
        "grade": edl.get("grade"),
        "lut": edl.get("lut"),
        "lut_path": None,
        "subtitles": edl.get("subtitles"),
        "sources_missing_media": [],
    }
    if notes["lut"]:
        try:
            notes["lut_path"] = find_lut_file(notes["lut"], edit_dir)
        except SystemExit:
            notes["lut_path"] = None  # named LUT not found — note the name only

    root = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(root, "sequence")
    _text(sequence, "name", sequence_name)

    comment_lines = []
    if notes["grade"]:
        comment_lines.append(f"grade (not applied — reproduce by eye): {notes['grade']}")
    if notes["lut"]:
        lut_hint = str(notes["lut_path"]) if notes["lut_path"] else notes["lut"]
        comment_lines.append(f"LUT (not applied — load this .cube on the color page): {lut_hint}")
    if notes["subtitles"]:
        comment_lines.append(f"captions: import {notes['subtitles']} separately (File > Import)")
    if comment_lines:
        _text(sequence, "comments", "video-use export notes:\n" + "\n".join(comment_lines))

    media = ET.SubElement(sequence, "media")
    video_el = ET.SubElement(media, "video")
    vformat = ET.SubElement(video_el, "format")
    vsc = ET.SubElement(vformat, "samplecharacteristics")
    _text(vsc, "width", str(width))
    _text(vsc, "height", str(height))
    _text(vsc, "pixelaspectratio", "square")
    _rate_block(vsc, seq_timebase, seq_ntsc)

    v1 = ET.SubElement(video_el, "track")
    audio_el = ET.SubElement(media, "audio")
    a1 = ET.SubElement(audio_el, "track")

    # -------- V1 / A1: the cut, one clipitem pair per EDL range, back to
    # back on the output timeline (ranges never have gaps between them — same
    # seg_offset accounting render.py's build_master_srt uses for captions).
    file_ids: dict[str, str] = {}
    clip_counter = 0
    seg_offset_s = 0.0

    for r in ranges:
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        seg_start_s = float(r["start"])
        seg_end_s = float(r["end"])
        seg_duration_s = seg_end_s - seg_start_s

        if not src_path.exists():
            notes["sources_missing_media"].append(src_name)

        if src_name not in file_ids:
            file_ids[src_name] = f"file-{src_name}"
            first_use = True
        else:
            first_use = False

        src_rate_str = probe_source_fps(src_path) or out_rate_str
        src_rate = Fraction(src_rate_str)
        src_timebase, src_ntsc = rate_to_timebase(src_rate)
        src_w, src_h = probe_dimensions(src_path) if first_use else (width, height)

        timeline_start = seconds_to_frames(seg_offset_s, seq_rate)
        timeline_end = seconds_to_frames(seg_offset_s + seg_duration_s, seq_rate)
        # Source in/out use the SOURCE's own native rate — correct per-source
        # timing even when one clip is 24fps and another is 60fps, which the
        # single sequence rate above cannot represent on its own.
        src_in = seconds_to_frames(seg_start_s, src_rate)
        src_out = seconds_to_frames(seg_end_s, src_rate)
        # A whole-file duration estimate wide enough to contain this range;
        # xmeml wants *some* value here and real NLEs re-probe the media
        # themselves on import, so this only needs to not undershoot src_out.
        # +1s of padding keeps it a strict overestimate without re-probing
        # the actual file length.
        file_duration = src_out + src_timebase

        clip_counter += 1
        clip_id = f"clipitem-{clip_counter}"
        clip_name = r.get("beat") or src_name

        for track, tag in ((v1, "video"), (a1, "audio")):
            clipitem = ET.SubElement(track, "clipitem", id=f"{clip_id}-{tag}")
            _text(clipitem, "name", clip_name)
            _rate_block(clipitem, seq_timebase, seq_ntsc)
            _text(clipitem, "start", str(timeline_start))
            _text(clipitem, "end", str(timeline_end))
            _text(clipitem, "in", str(src_in))
            _text(clipitem, "out", str(src_out))
            note = r.get("quote") or r.get("reason")
            if note:
                _text(clipitem, "comments", str(note))
            _file_block(
                clipitem, file_ids[src_name], src_path.name, src_path,
                src_timebase, src_ntsc, file_duration, src_w, src_h, first_use,
            )
            first_use = False  # only the very first clipitem (video) gets the full <file>

        seg_offset_s += seg_duration_s

    # -------- V2: overlays, positioned by start_in_output/duration. A second
    # <track> in xmeml composites over the first by default (same effect as
    # render.py's own `overlay` filter — opaque unless the file has alpha).
    if overlays:
        v2 = ET.SubElement(video_el, "track")
        for i, ov in enumerate(overlays, start=1):
            # resolve_edl_asset_path, not plain resolve_path: an EDL's overlay
            # "file" is routinely written "edit/animations/.../render.mp4" (the
            # SKILL.md example does this), which would double up against
            # edit_dir already being ".../edit" — see render.py's own comment
            # on resolve_edl_asset_path. Export still proceeds on a miss (the
            # notes/warning below cover it) — an export shouldn't hard-fail
            # the way a real render does, the XML is still useful to relink by hand.
            ov_path = resolve_edl_asset_path(ov["file"], edit_dir) or resolve_path(ov["file"], edit_dir)
            t0 = float(ov["start_in_output"])
            dur = float(ov["duration"])
            ov_rate_str = probe_source_fps(ov_path) or out_rate_str
            ov_rate = Fraction(ov_rate_str)
            ov_timebase, ov_ntsc = rate_to_timebase(ov_rate)
            ov_w, ov_h = probe_dimensions(ov_path)
            frames = seconds_to_frames(dur, ov_rate)

            clipitem = ET.SubElement(v2, "clipitem", id=f"overlay-{i}")
            _text(clipitem, "name", ov_path.stem)
            _rate_block(clipitem, seq_timebase, seq_ntsc)
            _text(clipitem, "start", str(seconds_to_frames(t0, seq_rate)))
            _text(clipitem, "end", str(seconds_to_frames(t0 + dur, seq_rate)))
            _text(clipitem, "in", "0")
            _text(clipitem, "out", str(frames))
            _file_block(
                clipitem, f"file-overlay-{i}", ov_path.name, ov_path,
                ov_timebase, ov_ntsc, frames, ov_w, ov_h, True,
            )
            if not ov_path.exists():
                notes["sources_missing_media"].append(f"overlay: {ov_path.name}")

    total_frames = seconds_to_frames(seg_offset_s, seq_rate)
    _text(sequence, "duration", str(total_frames))
    _rate_block(sequence, seq_timebase, seq_ntsc)

    return root, notes


def write_xmeml(root: ET.Element, out_path: Path) -> None:
    ET.indent(root, space="    ")
    body = ET.tostring(root, encoding="unicode")
    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export an EDL as an XMEML (\"Final Cut Pro XML\") timeline "
                     "for DaVinci Resolve / Premiere Pro"
    )
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .xml path (default: <edit_dir>/timeline.xml)",
    )
    ap.add_argument(
        "--name", type=str, default="video-use timeline",
        help="Sequence name shown in the NLE's project panel.",
    )
    ap.add_argument(
        "--fps", type=parse_fps, default=None,
        help="Timeline frame rate. Default: probe the first source, same "
             "policy as render.py.",
    )
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = (args.output or (edit_dir / "timeline.xml")).resolve()

    root, notes = build_timeline(edl, edit_dir, sequence_name=args.name, fps=args.fps)
    write_xmeml(root, out_path)

    print(f"timeline -> {out_path}")
    print(f"  {len(edl.get('ranges') or [])} cut(s), {len(edl.get('overlays') or [])} overlay(s)")
    if notes["grade"]:
        print(f"  grade NOT applied (no NLE equivalent) — noted in the XML: {notes['grade']}")
    if notes["lut"]:
        hint = notes["lut_path"] or notes["lut"]
        print(f"  LUT NOT applied — load this file on the color page yourself: {hint}")
    if notes["subtitles"]:
        print(f"  captions NOT embedded (xmeml has no portable caption element) — "
              f"import {notes['subtitles']} separately, both apps read .srt directly")
    if notes["sources_missing_media"]:
        print(f"  warning: media not found at render time for: "
              f"{', '.join(notes['sources_missing_media'])} — the XML still "
              f"references these paths; relink in the NLE if they've moved.")


if __name__ == "__main__":
    main()
