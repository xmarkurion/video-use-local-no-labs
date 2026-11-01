---
name: video-use
description: Edit any video by conversation. Transcribe, cut, color grade (incl. LUTs), generate overlay animations, burn or sidecar subtitles, censor profanity — for talking heads, montages, tutorials, travel, interviews. Also generates thumbnails, titles/descriptions, portrait YouTube Shorts/Reels/TikTok clips from a finished render, and exports an editable timeline to Premiere Pro / DaVinci Resolve instead of (or alongside) a flattened render. No presets, no menus. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
---

# Video Use

## Principle

1. **LLM reasons from raw transcript + on-demand visuals.** The only derived artifact that earns its keep is a packed phrase-level transcript (`takes_packed.md`). Everything else — filler tagging, retake detection, shot classification, emphasis scoring — you derive at decision time.
2. **Audio is primary, visuals follow.** Cut candidates come from speech boundaries and silence gaps. Drill into visuals only at decision points.
3. **Ask → confirm → execute → iterate → persist.** Never touch the cut until the user has confirmed the strategy in plain English.
4. **Generalize.** Do not assume what kind of video this is. Look at the material, ask the user, then edit.
5. **Artistic freedom is the default.** Every specific value, preset, font, color, duration, pitch structure, and technique in this document is a *worked example* from one proven video — not a mandate. Read them to understand what's possible and why each worked. Then make your own taste calls based on what the material actually is and what the user actually wants. **The only things you MUST do are in the Hard Rules section below.** Everything else is yours.
6. **Invent freely.** If the material calls for a technique not described here — split-screen, picture-in-picture, lower-third identity cards, reaction cuts, speed ramps, freeze frames, crossfades, match cuts, L-cuts, J-cuts, speed ramps over breath, whatever — build it. The helpers are ffmpeg and PIL. They can do anything the format supports. Do not wait for permission.
7. **Verify your own output before showing it to the user.** If you wouldn't ship it, don't present it.

## Hard Rules (production correctness — non-negotiable)

These are the things where deviation produces silent failures or broken output. They are not taste, they are correctness. Memorize them.

1. **Subtitles are applied LAST in the filter chain**, after every overlay. Otherwise overlays hide captions. Silent failure.
2. **Per-segment extract → lossless `-c copy` concat**, not single-pass filtergraph. Otherwise you double-encode every segment when overlays are added.
3. **30ms audio fades at every segment boundary** (`afade=t=in:st=0:d=0.03,afade=t=out:st={dur-0.03}:d=0.03`). Otherwise audible pops at every cut.
4. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start. Otherwise you see the middle of the animation during the overlay window.
5. **Master SRT uses output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`. Otherwise captions misalign after segment concat.
6. **Never cut inside a word.** Snap every cut edge to a word boundary from the transcript.
7. **Pad every cut edge.** Working window: 30–200ms. Forced-aligned timestamps drift ~30ms, hosted Scribe 50–100ms — padding absorbs the drift either way. Tighter for fast-paced, looser for cinematic.
8. **Word-level verbatim ASR only.** Never SRT/phrase mode (loses sub-second gap data). Never normalized fillers (loses editorial signal).
9. **Cache transcripts per source.** Never re-transcribe unless the source file itself changed — or unless you switched ASR backend or model, which the cache cannot see. Check `_video_use.model` in the JSON before trusting a transcript you did not make this session.
10. **Parallel sub-agents for multiple animations.** Never sequential. Spawn N at once via the `Agent` tool; total wall time ≈ slowest one.
11. **Strategy confirmation before execution.** Never touch the cut until the user has approved the plain-English plan.
12. **All session outputs in `<videos_dir>/edit/`.** Never write inside the `video-use/` project directory.
13. **Clamp every EDL range to its source's actual duration before rendering.** `render.py` and `export_timeline.py` both do this automatically (`clamp_edl_ranges`) — a range that runs even slightly past the end of its file doesn't error, it silently renders that segment short. Found in production: a title card and an end card both came up short/missing in a render because their EDL ranges ran a few frames past the actual card length. Don't skip the warning it prints when it clamps something — that warning is the only sign the EDL itself has an authoring bug worth fixing at the source.
14. **Verify the render, don't just trust it.** `render.py` checks the concatenated output's actual duration against the sum of the EDL's (clamped) ranges and fails loudly on a mismatch, rather than shipping a video that's quietly missing a section. If you ever bypass `render.py`'s own pipeline (hand-rolled ffmpeg), reproduce this check yourself before calling a render done.
15. **Render `--preview` by default. Full/final quality only on explicit request** ("ship it", "final", "that's the one", or equivalent). Never jump straight to a full render because the preview looked fine in your own pass — the user has to actually see it first. This is Principle 3 (confirm before execute) applied to the render step specifically, because it has been skipped in practice.

Everything else in this document is a worked example. Deviate whenever the material calls for it.

## Directory layout

The skill lives in `video-use/`. User footage lives wherever they put it. All session outputs go into `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md               ← memory; appended every session
    ├── takes_packed.md          ← phrase-level transcripts, the LLM's primary reading view
    ├── edl.json                 ← cut decisions
    ├── transcripts/<name>.json  ← cached raw word-level JSON (Scribe shape, either backend)
    ├── animations/slot_<id>/    ← per-animation source + render + reasoning
    ├── clips_graded/            ← per-segment extracts with grade + fades
    ├── master.srt               ← output-timeline subtitles
    ├── timeline.xml             ← XMEML export (export_timeline.py), if requested
    ├── thumbnails/              ← thumbnail.py candidates
    ├── shorts/                  ← shorts.py portrait clips
    ├── downloads/               ← yt-dlp outputs
    ├── verify/                  ← debug frames / timeline PNGs
    ├── preview.mp4
    └── final.mp4
```

## Setup

First-time install lives in `install.md` (clone, deps, ffmpeg, GPU check, skill registration). Don't re-run it every session; on cold start just verify:

- `ffmpeg` + `ffprobe` on PATH.
- Python deps installed, **including the `local` extra** (`uv sync --extra local`, or `pip install -e ".[local]"` inside the repo). `python -c "import whisperx"` is the fast check.
- CUDA visible to torch: `python -c "import torch; print(torch.cuda.is_available())"`. `False` on a machine with an NVIDIA card means a CPU-only torch wheel — see `install.md`. It still works on CPU, just slowly; tell the user rather than letting them wait.
- No API key is required. `HF_TOKEN` in `.env` only if the footage is multi-speaker and you want `--diarize`; `ELEVENLABS_API_KEY` only for the `--backend elevenlabs` fallback. Never write either to the user's `<videos_dir>`.
- Node.js + npm available if the session needs HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- `yt-dlp`, HyperFrames, Remotion, Manim installed only on first use.
- First-use animation setup happens inside the slot directory, never at the video-use repo root. HyperFrames can be invoked with `npx --yes hyperframes ...`; Remotion can be scaffolded with `npx create-video@latest` or installed as a project-local dependency before using its `remotion render` command.
- This skill vendors `skills/manim-video/`. Read its SKILL.md when building a Manim slot.

Helpers (`helpers/transcribe.py`, `helpers/render.py`, etc.) live alongside this SKILL.md. Resolve their paths relative to the directory containing this file — the skill is typically symlinked at `~/.claude/skills/video-use/` or `~/.codex/skills/video-use/`.

## Helpers

- **`transcribe.py <video>`** — single file, local GPU by default. `--diarize` (needs `HF_TOKEN`), `--num-speakers N`, `--audio-events` optional. Cached.
- **`transcribe_batch.py <videos_dir>`** — whole directory. Loads the models once and runs sequentially on the local backend; 4 parallel uploads on `--backend elevenlabs`. Use for multi-take.
- **`audio_events.py <path>`** — standalone AudioSet tagger, if you want `(laughter)`/`(applause)` spans for a source you already transcribed.
- **`pack_transcripts.py --edit-dir <dir>`** — `transcripts/*.json` → `takes_packed.md` (phrase-level, break on silence ≥ 0.5s).
- **`timeline_view.py <video> <start> <end>`** — filmstrip + waveform PNG. On-demand visual drill-down. **Not a scan tool** — use it at decision points, not constantly.
- **`render.py <edl.json> -o <out>`** — per-segment extract → concat → overlays (PTS-shifted) → subtitles LAST. `--preview` for a faster encode at the same resolution the final will ship at, not a separate downscale (Hard Rule 15: this is the default mental model — don't reach for a final render unasked; `--draft` is the one that actually drops to 720p, for cut-point checks only). `--build-subtitles` to generate master.srt inline; `--srt-only` writes it without burning; `--subtitle-style natural` for sentence-case readable cues instead of the bold-overlay default; `--text-fixes <file>` for project-specific ASR corrections. `--lut <name>` applies a `.cube` from `lut/` to every segment; a range's own `"lut"` field or the EDL's `"luts": {source: name}` map overrides it per-clip (useful the moment one source is log footage and another is a screen recording). Resolution defaults to the **highest native resolution among the EDL's sources** (`--list-resolutions` to see the detected numbers before committing); `--res 1080|2160` / `--width N` to force something else — see "Output spec". `--censor-profanity` (`--censor-mute` for silence instead of a bleep) redacts flagged words in captions and audio.
- **`grade.py <in> -o <out>`** — ffmpeg filter chain grade. Presets + `--filter '<raw>'` for custom.
- **`export_timeline.py <edl.json> -o <out.xml>`** — the EDL as an editable XMEML ("Final Cut Pro XML") timeline for DaVinci Resolve / Premiere Pro, instead of a flattened render. See "Exporting to an NLE" below.
- **`thumbnail.py <video>`** — 3 diverse candidate thumbnail frames (auto-picked by visual interest, or `--at <seconds>` for explicit picks), optionally with a `--title` burned into all 3 consistently so the comparison is "which frame" not "text or no text". `--size youtube|square|vertical`.
- **`shorts.py <video>`** — hand it any finished render and it generates 3 portrait (1080×1920, always — Shorts/Reels/TikTok all require it, there's no landscape mode to opt out of) clips by default: auto-picks 3 non-overlapping high-interest windows (speech density + visual interest), snaps every edge to a real word/sentence boundary (never a bare timestamp), reformats to vertical, burns captions, loudnorm per clip. See "Shorts" below.

For animations, create `<edit>/animations/slot_<id>/` with `Bash` and spawn a sub-agent via the `Agent` tool.

## Narrating progress

Several steps genuinely take a while — local transcription of a long batch, a full (non-preview) render, loudnorm's two-pass measurement. The helpers already print per-item progress to stdout (`transcribe_batch.py`'s `[N/Total]`, `render.py`'s per-segment `[00] source start-end`); don't let that be the only signal. Say what's happening in plain language as you go — *"transcribing clip 4 of 11..."*, *"rendering preview..."*, *"self-eval pass 2..."* — especially across tool calls that individually take more than a few seconds. Going silent for a long stretch reads as stuck even when it isn't.

## The process

1. **Inventory.** `ffprobe` every source (note resolution and frame rate while you're at it — this is what you'll offer the user a choice on in step 3/4). `transcribe_batch.py` on the directory. `pack_transcripts.py` to produce `takes_packed.md`. Sample one or two `timeline_view`s for a visual first impression.
2. **Pre-scan for problems.** One pass over `takes_packed.md` to note verbal slips, obvious mis-speaks, or phrasings to avoid. Plain list, feed into the editor brief.
3. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, pacing feel, must-preserve moments, must-cut moments, animation and grade preferences, subtitle needs, **output resolution** (name the actual detected numbers, not a generic "what resolution?" — see "Output spec"). Mention relevant capabilities the user might not know to ask for — a LUT if the footage looks log/flat, profanity censoring if the content has any, an NLE export if they might want to keep editing elsewhere, thumbnails for a publish-bound video. Do not use a fixed checklist — the right questions are different every time.
4. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, animation plan, grade direction, subtitle style, length estimate. **Wait for confirmation.**
5. **Execute.** Produce `edl.json` via the editor sub-agent brief. Drill into `timeline_view` at ambiguous moments. Build animations in parallel sub-agents. Apply grade per-segment. Compose via `render.py`.
6. **Preview.** `render.py --preview` — the default render target (Hard Rule 15). Do not render final quality until the user has seen a preview and confirmed it, even if nothing about the plan seems likely to change.
7. **Self-eval (before showing the user).** Run `timeline_view` on the **rendered output** (not the sources) at every cut boundary (±1.5s window). Check each image for:
   - Visual discontinuity / flash / jump at the cut
   - Waveform spike at the boundary (audio pop that slipped past the 30ms fade)
   - Subtitle hidden behind an overlay (Rule 1 violation)
   - Overlay misaligned or showing wrong frames (Rule 4 violation)

   Also sample: first 2s, last 2s, and 2–3 mid-points — check grade consistency, subtitle readability, overall coherence. Run `ffprobe` on the output to verify duration matches the EDL expectation — `render.py` already checks this itself and fails loudly on a mismatch (Hard Rule 14), but re-confirm if you rendered outside its own pipeline.

   Measure the audio, don't assume it: `ffmpeg -i out.mp4 -af ebur128=peak=true -f null -` for integrated loudness and true peak, plus RMS per section (dialogue, music-only, end card). An end card 15 dB under the dialogue, or effects louder than speech, is a bug. `render.py`'s own loudnorm step now re-measures and corrects true peak automatically (see "Music and sound effects" below) — this check is for anything that bypassed it. You cannot listen: say so, and report the numbers.

   For anything the user will publish (launch, promo, ad), also spawn one **critic sub-agent** with the rendered file, the EDL, and any reference videos the user gave. Brief it to roast, not to praise: a verdict, ranked problems with timecodes and evidence (frames, levels), and the 5 fixes to do first. Fresh eyes catch what the author stopped seeing — cut-off payoff lines, 0.5s memes, unreadable 28px text at phone size.

   If anything fails: fix → re-render → re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only present the preview once the self-eval passes.
8. **Iterate + persist.** Natural-language feedback, re-plan, re-render. Never re-transcribe. Final render ONLY on explicit confirmation (Hard Rule 15) — "looks good" about the preview is not the same thing as "render final." Append to `project.md`.
9. **Thumbnails + title/description (for anything publish-bound).** Once a render is confirmed: `thumbnail.py` on it for 3 frame candidates, present them for the user to pick rather than choosing yourself. For the title/description, **write them yourself from the material** — read `takes_packed.md` for the hook/quotes and propose 3-5 title options plus one short description; this is an editorial-judgment task matching Principle 1 (the LLM reasons from the transcript), not a job for a keyword-extraction script. Don't force this step on content that isn't going anywhere (an internal review cut, a quick trim) — ask if it's unclear whether the output is publish-bound. Also mention `shorts.py` here if the content could support short-form repurposing (a talking-head, a tutorial, anything with a few standalone-interesting moments) — see "Shorts" below. Offer it, don't run it unasked (Hard Rule 11 still applies to this step).

## Cut craft (techniques)

- **Audio-first.** Candidate cuts from word boundaries and silence gaps.
- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past punchlines to include reactions — the laugh IS the beat.
- **Speaker handoffs** benefit from air between utterances. Common values: 400–600ms. Less for fast-paced, more for cinematic. Taste call.
- **Audio events as signals.** `(laughs)`, `(sighs)`, `(applause)` mark beats. Extend past them.
- **Silence gaps are cut candidates.** Silences ≥400ms are usually the cleanest. 150–400ms phrase boundaries are usable with a visual check. <150ms is unsafe (mid-phrase).
- **Example cut padding** (the launch video shipped with this): 50ms before the first kept word, 80ms after the last. Tighter for montage energy, looser for documentary. Stay in the 30–200ms working window (Hard Rule 7).
- **Never reason audio and video independently.** Every cut must work on both tracks.

## Transcription

Local by default: WhisperX drives CrisperWhisper on the GPU, and word timestamps come from wav2vec2 forced alignment rather than Whisper's cross-attention DTW. Free, offline, no quota, and the footage never leaves the machine.

```bash
python helpers/transcribe_batch.py <videos_dir>                      # the normal case
python helpers/transcribe_batch.py <videos_dir> --diarize --num-speakers 2
python helpers/transcribe_batch.py <videos_dir> --audio-events
```

What each flag buys, and what it costs:

| Flag | Gives you | Cost |
|---|---|---|
| *(default)* | Verbatim text + ~30ms word timestamps | ~1.5GB model download once, ~3GB VRAM |
| `--diarize` | `speaker_id` per word | `HF_TOKEN`, +~1GB VRAM, ~0.3× realtime slower |
| `--audio-events` | `(laughter)` / `(applause)` spans | +~0.4GB VRAM. **~1s resolution** — editorial signal, never a cut point |
| `--model large-v3` | MIT-licensed weights | **Deletes every filler.** See the anti-patterns |
| `--backend elevenlabs` | Hosted Scribe | Costs credits, needs a key, uploads the audio |

Both backends write the same JSON shape, so nothing downstream cares which ran. A transcript records its own provenance in `_video_use`.

**Fillers come back as bracket tags.** CrisperWhisper writes them `[UH]`, `[UM]`, `[MM]` — not "um". Measured on a real 60s meeting take:

```
CrisperWhisper  My gosh you've already pres produced a PowerPoint presentation [UH] I think
                it's already on actually [UH] ... Okay, right, I know. Okay, right [UM] well
                this is the kick-off meeting for our our project.          (42 words)

large-v3        My gosh, you've already produced a PowerPoint presentation. I think it's
                already on, actually. ... This is the kick-off meeting for our project.
                                                                            (30 words)
```

`large-v3` dropped 29% of the words: all three fillers, the false start (`pres`), the repetition (`our our`), and one whole line. Each of those is a reason to prefer one take over another, and it is gone before you ever read the transcript.

Two consequences:
- **Finding fillers is now trivial** — match `^\[[A-Z]+\]$` instead of guessing which "um" was spoken. Use this for the cut-the-filler visual in the Animation section.
- **They never reach the picture.** `render.py` drops bracketed tokens from captions (`is_nonspeech_token`), so a kept filler shows no subtitle. Read `transcript["words"]` directly when picking cut edges — `_words_in_range` is the caption view, not the cut view.

**Verbatim is the whole point.** The default model keeps "um", "uh", stutters and false starts because that is the editorial signal you cut on — a take is worse than another take largely because of what it stumbles over. A transcript that reads cleanly has already thrown away the information you needed.

**What is genuinely worse than hosted Scribe.** Say so rather than papering over it:
- Audio events are coarse (~1s) and only cover a curated AudioSet list.
- Diarization is opt-in and needs a gated-model token.
- Non-English alignment depends on a wav2vec2 model existing for that language; `transcribe.py` warns and falls back to segment-level timing when one doesn't. Widen cut padding when you see that warning.

## The packed transcript (primary reading view)

`pack_transcripts.py` reads all `transcripts/*.json` and produces one markdown file where each take is a list of phrase-level lines, each prefixed with its `[start-end]` time range. Phrases break on any silence ≥ 0.5s OR speaker change. This is the artifact the editor sub-agent reads to pick cuts — it gives word-boundary precision from text alone at 1/10 the tokens of raw JSON.

Example line:
```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

## Editor sub-agent brief (for multi-take selection)

When the task is "pick the best take of each beat across many clips," spawn a dedicated sub-agent with a brief shaped like this. The structure is load-bearing; the pitch-shape example is not.

```
You are editing a <type> video. Pick the best take of each beat and 
assemble them chronologically by beat, not by source clip order.

INPUTS:
  - takes_packed.md (time-annotated phrase-level transcripts of all takes)
  - Product/narrative context: <2 sentences from the user>
  - Speaker(s): <name, role, delivery style note>
  - Expected structure: <pick an archetype or invent one>
  - Verbal slips to avoid: <list from the pre-scan pass>
  - Target runtime: <seconds>

Common structural archetypes (pick, adapt, or invent):
  - Tech launch / demo:   HOOK → PROBLEM → SOLUTION → BENEFIT → EXAMPLE → CTA
  - Tutorial:             INTRO → SETUP → STEPS → GOTCHAS → RECAP
  - Interview:            (QUESTION → ANSWER → FOLLOWUP) repeat
  - Travel / event:       ARRIVAL → HIGHLIGHTS → QUIET MOMENTS → DEPARTURE
  - Documentary:          THESIS → EVIDENCE → COUNTERPOINT → CONCLUSION
  - Music / performance:  INTRO → VERSE → CHORUS → BRIDGE → OUTRO
  - Or invent your own.

RULES:
  - Start/end times must fall on word boundaries from the transcript.
  - Pad cut boundaries (working window 30–200ms).
  - Prefer silences ≥ 400ms as cut targets.
  - Unavoidable slips are kept if no better take exists. Note them in "reason".
  - If over budget, revise: drop a beat or trim tails. Report total and self-correct.

OUTPUT (JSON array, no prose):
  [{"source": "C0103", "start": 2.42, "end": 6.85, "beat": "HOOK",
    "quote": "...", "reason": "..."}, ...]

Return the final EDL and a one-line total runtime check.
```

## Color grade (when requested)

Your job is to **reason about the image**, not apply a preset. Look at a frame (via `timeline_view`), decide what's wrong, adjust one thing, look again.

Mental model is ASC CDL. Per channel: `out = (in * slope + offset) ** power`, then global saturation. `slope` → highlights, `offset` → shadows, `power` → midtones.

**Example filter chains** (`grade.py` has `--list-presets`; use them as starting points or mix your own):

- **`warm_cinematic`** — retro/technical, subtle teal/orange split, desaturated. Shipped in a real launch video. Safe for talking heads.
- **`neutral_punch`** — minimal corrective: contrast bump + gentle S-curve. No hue shifts.
- **`none`** — straight copy. Default when the user hasn't asked.

For anything else — portraiture, nature, product, music video, documentary — invent your own chain. `grade.py --filter '<raw ffmpeg>'` accepts any filter string.

Hard rules: apply **per-segment during extraction** (not post-concat, which re-encodes twice). Never go aggressive without testing skin tones.

**3D LUTs.** A LUT is a different thing from `grade`: `grade` is corrective math (fix exposure/contrast/saturation), a LUT is a baked creative look or a camera log-to-Rec.709 conversion — a `.cube` file, dropped in `lut/` (see `lut/README.md`). Reference it by `--lut <name>` on `render.py`, or `"lut": "<name>"` in the EDL (CLI overrides EDL). Both can be on the same render — grade runs first, LUT last, same order a colorist layers a corrective node under a creative one. A camera/drone's own log-footage LUT (e.g. D-Log → Rec.709) almost always belongs here when the source is flat/log and the user hasn't manually graded it; ask if unsure which the footage is.

**Don't trust the container's color tags to tell you if footage is log.** A real DJI Osmo Action 4 D-Log M file is tagged `bt709` in its stream metadata — the tag describes the container's declared colorspace, not whether the actual pixel values are flat/log. Look at a frame (`timeline_view`): log footage is visibly washed out, low-contrast, desaturated, even when ffprobe insists it's already Rec.709.

**LUTs can differ per source on the same timeline.** A single EDL-level `"lut"` applies to every segment — fine until one source is log camera footage and another is a screen recording that needs no LUT at all (applying the camera's LUT to screen content looks wrong). Override per range with `"lut": "<name>"` on that range, or set `"luts": {"<source_name>": "<lut_name>"}` at the EDL level to map specific sources without repeating it on every range. Most specific wins: a range's own `lut` > `luts[source]` > the EDL's top-level `lut`.

## Subtitles (when requested)

Subtitles have three dimensions worth reasoning about: **chunking** (1/2/3/sentence per line), **case** (UPPER/Title/Natural), and **placement** (margin from bottom). The right combo depends on content.

**Worked styles** — pick, adapt, or invent:

**`bold-overlay`** — short-form tech launch, fast-paced social. ~2-word chunks, UPPERCASE, break on punctuation and pauses ≥ 0.3s, grow to 3 words rather than flash a cue < 0.35s (`chunk_words` in `render.py`), Helvetica 18 Bold, white-on-outline, `MarginV=35`. `render.py` ships with this as `SUB_FORCE_STYLE`.

```
FontName=Helvetica,FontSize=18,Bold=1,
PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,
BorderStyle=1,Outline=2,Shadow=0,
Alignment=2,MarginV=35
```

**`natural`** (`--subtitle-style natural`, shipped) — narrative, documentary, education, or any sidecar meant to be read in a native player's own caption renderer rather than burned. Reading-length cues (≤42 chars / ≤8 words), break on sentence punctuation or a pause ≥0.45s, sentence case (capitalizes only right after a cue actually ended a sentence), cues stretched toward a 1s minimum without ever overlapping the next one. Pair with `--srt-only` (writes `master.srt`, skips burning — bold-overlay's UPPERCASE 2-word chunks read badly as plain text, natural reads badly burned into frame at speed; pick the one that matches where the captions will actually be read) and `--text-fixes <file>` for project-specific ASR corrections (`pattern<TAB>replacement` per line — a proper noun or technical term the transcript consistently mishears, fixed once instead of patched by hand in every take).

Invent a third style if neither fits. Hard rules: subtitles LAST (Rule 1), output-timeline offsets (Rule 5).

## Profanity filter (when requested)

`render.py --censor-profanity`: captions show `***` in place of a flagged word, and the audio at that exact moment is bleeped (default, a 1kHz tone replacing it) or silenced (`--censor-mute`). Wordlist is `profanity/wordlist.txt` — plain text, one stem per line, user-editable, no code change needed (`--censor-wordlist <path>` for a custom one). Matching is stem + inflection (`-ing/-ed/-er/-ers/-es/-s`), not substring, so it never flags "assume" or "class" — see `helpers/profanity_filter.py` for the exact rule if you need to explain a miss or a false positive.

What it does NOT do, say so if asked:
- **Only redacts captions it generates itself.** Pair with `--build-subtitles`. An EDL-pointed, hand-written subtitles file is left untouched — audio still gets censored, captions don't; `render.py` prints this limitation when it applies.
- **Only catches what's in the wordlist.** The default list is mainstream non-slur profanity (`fuck`, `shit`, `bitch`, etc.) — deliberately no slurs shipped by default. Extend the wordlist yourself for stricter or region-specific moderation.
- **Censor padding (60ms each side) is separate from Hard Rule 7's cut-edge padding** — it exists to fully cover a word's audio even when alignment is a touch tight, not to absorb drift at a splice.

## Animations (when requested)

Animations match the content and the brand. **Get the palette, font, and visual language from the conversation** — never assume a default. If the user hasn't told you, propose a palette in the strategy phase and wait for confirmation before building anything.

**Tool options:**

Pick the engine per animation slot. Do not default to Remotion just because the animation is web-adjacent.

- **HyperFrames** — Browser-native HTML/CSS/GSAP video compositions: product UI motion, website-to-video or mockup-to-video captures, kinetic typography, landing-page/storyboard promos, data-driven UI states, transparent WebM overlays, and clips that need deterministic frame capture plus HyperFrames lint/validate/render checks. Best when the animation should be authored and verified like a web composition instead of a React component tree.
- **Remotion** — React/CSS compositions with component state, reusable React primitives, or an existing Remotion brand system. Best when the user specifically asks for React/Remotion or when React composition is the simpler authoring model.
- **Manim** — formal diagrams, state machines, equation derivations, graph morphs. Read `skills/manim-video/SKILL.md` and its references for depth.
- **PIL + PNG sequence + ffmpeg** — simple overlay cards: counters, typewriter text, single bar reveals, progressive draws. Fast to iterate, any aesthetic you want. The launch video used this.

For HyperFrames slots, scaffold the slot inside `edit/animations/slot_<id>/` with `npx --yes hyperframes init . --example blank --non-interactive --skip-skills`, build the HTML composition there, run the HyperFrames checks that fit the slot (`lint`, `validate`, and a draft render when practical), then produce the final overlay video with `npx --yes hyperframes render . -o render.mp4` or `--format webm -o render.webm` when alpha is required. Point the EDL overlay `file` at the actual rendered path.

For Remotion slots, keep the Remotion project isolated inside the same slot directory, scaffold with `npx create-video@latest` or install Remotion locally there, render the composition to `render.mp4` with the project-local `remotion render` command, and verify duration and dimensions with `ffprobe`.

None is mandatory. Invent hybrids if useful (e.g., PIL background with a HyperFrames or Remotion layer on top).

**Duration rules of thumb, context-dependent:**

- **Sync-to-narration explanations.** A viewer needs to parse the content at 1×. Rough floor 3s, typical 5–7s for simple cards, 8–14s for complex diagrams. The launch video shipped at 5–7s per simple card.
- **Beat-synced accents** (music video, fast montage). 0.5–2s is fine — they're visual accents, not information. The "readable at 1×" rule becomes *"recognizable at 1×"*, not *"fully parseable."*
- **Hold the final frame ≥ 1s** before the cut (universal).
- **Over voiceover:** total duration ≥ `narration_length + 1s` (universal).
- **Never parallel-reveal independent elements** — the eye can't track two new things at once. One thing, pause, next thing.

**Animation payoff timing (rule for sync-to-narration):** get the payoff word's timestamp. Start the overlay `reveal_duration` seconds earlier so the landing frame coincides with the spoken payoff word. Without this sync the animation feels disconnected.

**Easing** (universal — never `linear`, it looks robotic):

```python
def ease_out_cubic(t):    return 1 - (1 - t) ** 3
def ease_in_out_cubic(t):
    if t < 0.5: return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2
```

`ease_out_cubic` for single reveals (slow landing). `ease_in_out_cubic` for continuous draws.

**Typing text anchor trick:** center on the FULL string's width, not the partial-string width — otherwise text slides left during reveal.

**Example palette** (the launch video — one aesthetic among infinite):
- Background `(10, 10, 10)` near-black
- Accent `#FF5A00` / `(255, 90, 0)` orange
- Labels `(110, 110, 110)` dim gray
- Font: Menlo Bold at `/System/Library/Fonts/Menlo.ttc` (index 1)
- ≤ 2 accent colors, ~40% empty space, minimal chrome
- Result: terminal / retro tech feel

This is one style. If the brand is warm and serif, use that. If it's colorful and playful, use that. If the user handed you a style guide, follow it. If they didn't, propose one and confirm.

**Fonts fail silently.** A web font that didn't load renders in a fallback face with no error — the video ships in "almost Arial". In HyperFrames/Remotion, await the font load and then assert it: `if (!document.fonts.check('700 76px "Inter"')) throw new Error(...)`. In PIL, pass an explicit font path; never rely on the default.

**Worked example — "show the edit" hook** (a launch video for this tool). Instead of a title card, the first 3s visualize the editing itself: each transcript word pops in on its forced-aligned timestamp, with bars under it drawn from the real audio envelope; a filler ("ummm") grows letter by letter while it is spoken, turns orange and is cut out on screen at the same frame the audio cuts, and the next line lands immediately. It works because the picture is *driven by the same data as the sound* — one composition (Remotion) reads frame-exact word/envelope JSON produced in Python, so nothing can drift. Use the idea whenever the story is "we removed something": make the removal visible.

**Parallel sub-agent brief** — each animation is one sub-agent spawned via the `Agent` tool. Each prompt is self-contained (sub-agents have no parent context). Include:

1. One-sentence goal: *"Build ONE animation: [spec]. Nothing else."*
2. Absolute output path (`<edit>/animations/slot_<id>/render.mp4`)
3. Exact technical spec: resolution, fps, codec, pix_fmt, CRF, duration
4. Style palette as concrete values (RGB tuples, hex, or reference to a design system)
5. Font path with index
6. Frame-by-frame timeline (what happens when, with easing)
7. Anti-list ("no chrome, no extras, no titles unless specified")
8. Code pattern reference (copy helpers inline, don't import across slots)
9. Deliverable checklist (script, render, verify duration via ffprobe, report)
10. **"Do not ask questions. If anything is ambiguous, pick the most obvious interpretation and proceed."**

One sub-agent = one file (unique filenames, parallel agents don't overwrite each other).

## Music and sound effects (when requested)

Sound is where generated videos sound cheap. Worked rules from launch edits:

- **Fewer effects.** Every effect is tied to something visible (a cut, a landing, a click). ~20 stock whooshes/risers/impacts in 18s reads as generic; ~8 reads as designed.
- **Hit on the frame.** Most effects have an attack (silence or a build before the transient). Measure it (first sample above ~-30 dBFS of the peak) and start the file `attack` seconds *before* the visible contact frame.
- **Duck music under speech** (roughly -12 to -15 dB relative to its music-only level), and ramp it out before a stinger or end card instead of letting its own tail decay under your CTA.
- **Master once:** mix to PCM, then two-pass loudnorm (-14 LUFS, true peak ≤ -1 dBTP) on the final mix. Then measure per section (see Self-eval). `render.py`'s own loudnorm step re-measures the ACTUAL output's true peak afterward (via `ebur128`) and applies a limiter if it's still over target — two-pass loudnorm's own linear-mode gain can overshoot the TP ceiling on real content even while reporting success (found in production: reported -1 dBTP, measured output was 0.0 dBFS). Don't trust loudnorm's self-report; this is why the self-eval step re-measures independently too.
- **Music taste is the user's call.** Generated music defaults to "hype"; offer two contrasting beds and let the user listen. Don't claim a mix sounds good — you can only measure it.

## Output spec

**Resolution is a choice you offer, not a default you pick silently.** `render.py --list-resolutions` prints every source's actual native resolution and which one it would default to (the highest, so a 4K source is never silently downscaled and a 1080p one is never silently upscaled) — run it during Propose Strategy and put the real numbers in front of the user: *"your DJI footage is 3840px, the screen recording is 1920px — render at the full 3840, downscale to 1080p, or something else?"* Don't just mention that resolution is configurable; name the actual detected numbers so the choice is concrete. `--res 1080`/`--res 2160` cover the two common cases, `--width N` anything else. This applies to preview renders too — the preview should already be at the resolution the final will ship at, not a separate guess.

Frame rate and aspect are the other delivery-format questions, same "ask with real numbers" treatment: common targets are `1920×1080@24` cinematic, `1920×1080@30` screen content, `1080×1920@30` vertical social, `3840×2160@24` 4K cinema, `1080×1080@30` square — but match the actual source unless the user wants something specific. `--fps` on `render.py` overrides frame rate the same way `--width`/`--res` override resolution.

## EDL format

```json
{
  "version": 1,
  "sources": {"C0103": "/abs/path/C0103.MP4", "C0108": "/abs/path/C0108.MP4"},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85,
     "beat": "HOOK", "quote": "...", "reason": "Cleanest delivery, stops before slip at 38.46."},
    {"source": "C0108", "start": 14.30, "end": 28.90,
     "beat": "SOLUTION", "quote": "...", "reason": "Only take without the false start."}
  ],
  "grade": "warm_cinematic",
  "lut": "DJI OSMO Action 4 D-Log M to Rec.709 V1",
  "luts": {"C0108": "screen_recording_none"},
  "overlays": [
    {"file": "edit/animations/slot_1/render.mp4", "start_in_output": 0.0, "duration": 5.0}
  ],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

`grade` is a preset name or raw ffmpeg filter. `lut` is optional, a name from `lut/` or a path to a `.cube` file — `--lut` on the command line overrides it, and a range's own `"lut"` field or the `"luts"` per-source map overrides it per-clip (see "Color grade" above). `overlays` are rendered animation clips. `subtitles` is optional and applied LAST. Profanity censoring (`--censor-profanity`) is a render-time CLI flag, not an EDL field — it's a delivery-format choice (clean cut vs. uncensored), not an editorial decision baked into the cut itself. Every range's `end` is clamped to its source's actual duration before rendering (Hard Rule 13) — you don't need to hand-verify this yourself, but a clamp warning in the output means the EDL itself has an authoring bug (likely padding that overran a short clip) worth fixing at the source rather than just accepting the clamp.

## Shorts (when requested, or offer it for anything publish-bound)

`shorts.py` turns one finished render into 3 portrait clips — the workflow is literally "give it a final render and it generates the 3 shorts," no EDL or prior transcript required:

```bash
python helpers/shorts.py edit/final.mp4
```

Portrait (1080×1920) is not a choice, it's the only output — every short-form platform requires it, so there's nothing to default-enable. `--count` changes how many (default 3), `--duration` the target length in seconds (default 36, before boundary snapping — see below). Captions are on by default (bold-overlay, same proven style/positioning as `render.py`'s own); `--no-captions` turns them off. `--style crop` for a tight center-fill instead of the default `blur-bg` (blurred, scaled background behind the full uncropped source, centered — the standard "landscape repurposed as vertical" look; nothing from the original frame is ever lost, unlike a hard crop that can cut a face or a graphic off-screen).

**How candidates get picked:** `shorts.py` transcribes the input locally itself (cached — same backend as everything else in this skill, free, no quota) if it doesn't already have a transcript, then scores sliding windows by speech density (favor continuous talking over dead air) and visual interest (reused from `thumbnail.py`'s frame scoring), and picks the top 3 non-overlapping windows. Every window's edges then snap to a real word boundary, preferring a sentence-ending word near the target duration over an arbitrary timestamp — same "never cut inside a word" principle as Hard Rule 6, applied to shorts specifically. A long silence or visual-only stretch near a chosen start can make one clip run longer than the target rather than landing short or mid-word; that's working as intended, not a bug to fix by forcing the duration.

`--at <seconds>` (repeatable) picks explicit starts instead of auto-scoring — still snapped to a sentence boundary, still gets everything else (reformat, captions, loudnorm).

For this video specifically, titles/descriptions for each short are the same editorial-judgment call as the main video's (see "Thumbnails + title/description" in The process) — write them from what's actually in that clip, don't invent generic ones.

## Exporting to an NLE (Premiere Pro / DaVinci Resolve)

The default output is a flattened MP4. If the user wants to keep editing in Premiere or Resolve instead — ask when they say "export", "hand off", "open in Premiere/Resolve", or similar — use `export_timeline.py` on the same `edl.json` instead of (or alongside) `render.py`:

```bash
python helpers/export_timeline.py edit/edl.json -o edit/timeline.xml
```

It writes **XMEML** (`<xmeml version="5">`, the format Premiere's own File → Export → Final Cut Pro XML produces and Resolve has supported natively for years) — not true FCPXML. The two are both informally called "FCPXML" and are genuinely different schemas; XMEML is the one Adobe documents for third-party round-tripping with Premiere specifically, so it's the safer bet for this pair of apps. If the user explicitly wants Final Cut Pro X itself, that needs true FCPXML instead — different code path, not currently built.

**What survives the handoff, what doesn't — say this plainly when you hand the file over, don't let the user discover it by surprise:**

| | Result |
|---|---|
| Cuts (`ranges`) | ✅ exact — real source files, frame-accurate in/out |
| Overlays | ✅ a second video track (V2), same compositing model as the MP4 render |
| `grade` / `lut` | ⚠️ **not applied.** No portable way to embed a third-party color node both apps load identically — faking one would silently do nothing. Written as a comment on the sequence and on affected clips instead (exact preset/filter string, or the resolved `.cube` path to load on the color page by hand) — including a per-clip note when a range/source has its own LUT different from the global one. |
| `subtitles` | ⚠️ **not embedded** — xmeml predates any caption element both apps agree on. Left as the sidecar `.srt`; both apps import it directly (File → Import, or drag onto the timeline). |

**A clipitem's `in`/`out` are expressed at the SEQUENCE rate, not the source's own native rate** — this is load-bearing, not an implementation detail to skip past. A clipitem carries exactly one `<rate>` block, and both apps read `in`/`out` through *that* rate; there's no separate "read the source range at its own native fps" mode. Get this wrong (as an earlier version of `export_timeline.py` did) and a 47.952fps camera clip cut at source-second 113 gets written as `round(113 × 47.952)` and read back as `that_frame_count / sequence_fps` — roughly double the real position, past the end of the file on a real project, and Premiere shows the clip as hatched "media offline" bars. `out − in == end − start` always, by construction — if you ever hand-edit a generated XML and that invariant breaks, the clip will not play correctly.

Every range is clamped to its source's actual duration before export (same `clamp_edl_ranges` as `render.py`, Hard Rule 13) — a range that runs past the end of its file doesn't error on import, it just shows as a hatched/offline tail.

**Opening it:** File → Import (or drag the `.xml` onto the project panel) in either app. If the EDL had a LUT, load the noted `.cube` path onto the clip's color page by hand — Resolve: right-click the clip → LUT submenu (or the color page's node LUT picker); Premiere: Lumetri Color panel → Basic Correction → Input LUT. For the sidecar subtitles, File → Import the `.srt` directly in either app.

## Memory — `project.md`

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach
**Decisions:** take choices, cuts, grades, animations + why
**Reasoning log:** one-line rationale for non-obvious decisions
**Outstanding:** deferred items
```

On startup, read `project.md` if it exists and summarize the last session in one sentence before asking whether to continue.

## Anti-patterns

Things that consistently fail regardless of style:

- **Hierarchical pre-computed codec formats** with USABILITY / tone tags / shot layers. Over-engineering. Derive from the transcript at decision time.
- **Hand-tuned moment-scoring functions.** The LLM picks better than any heuristic you'll write.
- **Whisper SRT / phrase-level output.** Loses sub-second gap data. Always word-level verbatim.
- **Stock Whisper for a cut.** `large-v3` silently deletes the fillers, stutters and false starts that the edit exists to remove — the editorial signal is gone before you read the transcript. The default local model is CrisperWhisper, which is trained to keep them. If you must use `large-v3` (its licence is MIT, CrisperWhisper's weights are not), say so in `project.md` and stop promising filler-aware cuts.
- **Whisper's own word timestamps.** Whisper derives them from cross-attention DTW and they drift. Always go through forced alignment — `transcribe.py` does, via WhisperX.
- **Burning subtitles into base before compositing overlays.** Overlays hide them. (Hard Rule 1.)
- **Single-pass filtergraph when you have overlays.** Double re-encodes. Use per-segment extract → concat.
- **Linear animation easing.** Looks robotic. Always cubic.
- **Unverified web fonts.** A failed load silently falls back to a system face. Assert the font loaded before rendering.
- **Stock SFX on every transition.** Tie each effect to a visible event; cap the count.
- **Hard audio cuts at segment boundaries.** Audible pops. (Hard Rule 3.)
- **Typing text centered on the partial string.** Text slides left as it grows.
- **Sequential sub-agents for multiple animations.** Always parallel.
- **Editing before confirming the strategy.** Never.
- **Re-transcribing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
