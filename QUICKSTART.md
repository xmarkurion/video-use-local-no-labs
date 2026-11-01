# Quickstart

Two ways to run video-use: **with an agent** (Claude Code, Codex, etc. — conversational, the intended way) or **without one** (every helper is a plain Python CLI; drive it by hand). Both produce the same `edit/final.mp4`. Pick whichever fits the moment.

First-time setup either way: see [`install.md`](./install.md) for the full walkthrough (GPU check, deps). The condensed version:

```bash
git clone https://github.com/browser-use/video-use ~/Developer/video-use
cd ~/Developer/video-use
uv sync --extra local        # or: pip install -e ".[local]"
```

Needs `ffmpeg`/`ffprobe` on `PATH` and, for the default local transcription, an NVIDIA GPU with ~6GB VRAM (works on CPU too, just slower — see `install.md`). **No API key required.**

> **Windows note:** if a plain `python helpers/render.py ...` crashes with `UnicodeEncodeError` on a `→` character, that's your terminal's codepage, not a bug — run `set PYTHONIOENCODING=utf-8` once per session (PowerShell: `$env:PYTHONIOENCODING="utf-8"`) first.

---

## Option A — With an agent (recommended)

```bash
cd /path/to/your/footage
claude          # or: codex, or any agent with shell access
```

Then just say:

> edit these into a launch video

The agent inventories your clips, proposes a strategy in plain English, waits for your OK, then produces `edit/final.mp4` next to your sources. All outputs land in `<footage>/edit/` — nothing touches the video-use repo itself.

First time in a new agent, install itself with one message — see the [README setup prompt](./README.md#setup-prompt). After that, `SKILL.md` is what the agent actually reads each session; you never need to open it.

**Asking for the extras in this doc**, in plain English, works the same way:

> grade it with the DJI D-Log LUT in the lut folder

> bleep the swearing — audio and captions both

> export this as a Premiere/Resolve timeline instead of rendering it

The agent translates that into the right `--lut` / `--censor-profanity` flags on `render.py` for you.

---

## Option B — Without an agent (manual CLI)

Every helper is a standalone script. Running them by hand means **you** do the job an agent normally does — mainly, picking cut points and writing `edl.json` — using `takes_packed.md` as your reference. Everything else (extraction, grading, captioning, loudness) is mechanical.

```bash
cd ~/Developer/video-use
VIDEOS=/path/to/your/footage

# 1. Transcribe every clip (local GPU, free, cached — safe to re-run)
python helpers/transcribe_batch.py "$VIDEOS"

# 2. Pack transcripts into one readable file — this is what you read to pick cuts
python helpers/pack_transcripts.py --edit-dir "$VIDEOS/edit"
cat "$VIDEOS/edit/takes_packed.md"
```

`takes_packed.md` looks like this — `[start-end]` ranges are exactly what you copy into the EDL:

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

```bash
# 3. Write edl.json by hand (see the full schema in SKILL.md "EDL format").
#    Minimal example — one clip, one cut, snapped to the word boundaries above:
cat > "$VIDEOS/edit/edl.json" <<'EOF'
{
  "version": 1,
  "sources": {"C0103": "/path/to/your/footage/C0103.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.47, "end": 6.70, "beat": "HOOK"}
  ],
  "grade": "warm_cinematic"
}
EOF

# 4. Preview render (fast, 720p-ish, good enough to sanity-check cuts)
python helpers/render.py "$VIDEOS/edit/edl.json" -o "$VIDEOS/edit/preview.mp4" --preview --build-subtitles

# 5. Watch it. Happy? Final render:
python helpers/render.py "$VIDEOS/edit/edl.json" -o "$VIDEOS/edit/final.mp4" --build-subtitles
```

That's the whole loop: edit `edl.json`, re-run step 4, repeat until it's right, then step 5. `render.py` never re-transcribes (step 1 is cached per source), so iterating on cuts is fast.

**Optional: look before you cut.** For a visual gut-check at a specific timestamp instead of reading text:

```bash
python helpers/timeline_view.py "$VIDEOS/C0103.MP4" 2.0 7.0 -o /tmp/check.png
```

### Adding the extras manually

```bash
# A LUT from lut/ (drop your .cube file there first — see lut/README.md)
python helpers/render.py edl.json -o final.mp4 --build-subtitles \
  --lut "DJI OSMO Action 4 D-Log M to Rec.709 V1"

# Censor profanity: *** in captions, a bleep in the audio
python helpers/render.py edl.json -o final.mp4 --build-subtitles \
  --censor-profanity

# Same, but silence instead of a bleep tone
python helpers/render.py edl.json -o final.mp4 --build-subtitles \
  --censor-profanity --censor-mute

# Multi-speaker footage: diarize so captions/EDL can reference S0/S1/...
python helpers/transcribe_batch.py "$VIDEOS" --diarize --num-speakers 2

# Export the cut as an editable timeline instead of (or alongside) a render —
# opens in DaVinci Resolve or Premiere Pro. See SKILL.md "Exporting to an NLE"
# for exactly what does and doesn't survive the handoff (grade/LUT and
# captions do not — they're noted in the XML / left as a sidecar .srt).
python helpers/export_timeline.py edl.json -o edit/timeline.xml
```

(`--diarize` needs a free HuggingFace token in `.env` the first time — `install.md` covers getting one; every other command above needs nothing.)

### Cheat sheet — every flag that matters day to day

| Command | What it's for |
|---|---|
| `transcribe_batch.py <dir>` | Transcribe every clip (verbatim, local GPU, cached) |
| `transcribe_batch.py <dir> --diarize --num-speakers N` | + speaker labels |
| `transcribe_batch.py <dir> --audio-events` | + `(laughter)`/`(applause)` tags |
| `pack_transcripts.py --edit-dir <dir>` | Build the human-readable `takes_packed.md` |
| `timeline_view.py <video> <start> <end>` | Filmstrip + waveform PNG for one time range |
| `render.py <edl> -o <out> --preview` | Fast draft render |
| `render.py <edl> -o <out> --build-subtitles` | + burn captions from the transcript |
| `render.py <edl> -o <out> --lut <name>` | + apply a `.cube` from `lut/` |
| `render.py <edl> -o <out> --censor-profanity` | + bleep/`***` flagged words |
| `render.py <edl> -o <out> --censor-profanity --censor-mute` | …silence instead of a bleep |
| `grade.py --list-presets` | See the built-in color grade presets |
| `export_timeline.py <edl> -o <out.xml>` | Editable timeline for DaVinci Resolve / Premiere Pro, instead of a flattened render |

Full flag reference: `python helpers/<name>.py --help`. Full craft/judgment reference (cut rules, grading taste, subtitle styles, animation technique): [`SKILL.md`](./SKILL.md) — written for an agent to read, but it's plain markdown; read it yourself too.
