---
name: video-use
description: Edit any video by conversation. Transcribe, cut, color grade, generate overlay animations, burn subtitles — for talking heads, montages, tutorials, travel, interviews. No presets, no menus. Ask questions, confirm the plan, execute, iterate, persist. Production-correctness rules are hard; everything else is artistic freedom.
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
- **`render.py <edl.json> -o <out>`** — per-segment extract → concat → overlays (PTS-shifted) → subtitles LAST. `--preview` for 720p fast. `--build-subtitles` to generate master.srt inline. `--lut <name>` applies a `.cube` from `lut/`. `--censor-profanity` (`--censor-mute` for silence instead of a bleep) redacts flagged words in captions and audio.
- **`grade.py <in> -o <out>`** — ffmpeg filter chain grade. Presets + `--filter '<raw>'` for custom.
- **`export_timeline.py <edl.json> -o <out.xml>`** — the EDL as an editable XMEML ("Final Cut Pro XML") timeline for DaVinci Resolve / Premiere Pro, instead of a flattened render. See "Exporting to an NLE" below.

For animations, create `<edit>/animations/slot_<id>/` with `Bash` and spawn a sub-agent via the `Agent` tool.

## The process

1. **Inventory.** `ffprobe` every source. `transcribe_batch.py` on the directory. `pack_transcripts.py` to produce `takes_packed.md`. Sample one or two `timeline_view`s for a visual first impression.
2. **Pre-scan for problems.** One pass over `takes_packed.md` to note verbal slips, obvious mis-speaks, or phrasings to avoid. Plain list, feed into the editor brief.
3. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, pacing feel, must-preserve moments, must-cut moments, animation and grade preferences, subtitle needs. Do not use a fixed checklist — the right questions are different every time.
4. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, animation plan, grade direction, subtitle style, length estimate. **Wait for confirmation.**
5. **Execute.** Produce `edl.json` via the editor sub-agent brief. Drill into `timeline_view` at ambiguous moments. Build animations in parallel sub-agents. Apply grade per-segment. Compose via `render.py`.
6. **Preview.** `render.py --preview`.
7. **Self-eval (before showing the user).** Run `timeline_view` on the **rendered output** (not the sources) at every cut boundary (±1.5s window). Check each image for:
   - Visual discontinuity / flash / jump at the cut
   - Waveform spike at the boundary (audio pop that slipped past the 30ms fade)
   - Subtitle hidden behind an overlay (Rule 1 violation)
   - Overlay misaligned or showing wrong frames (Rule 4 violation)

   Also sample: first 2s, last 2s, and 2–3 mid-points — check grade consistency, subtitle readability, overall coherence. Run `ffprobe` on the output to verify duration matches the EDL expectation.

   Measure the audio, don't assume it: `ffmpeg -i out.mp4 -af ebur128=peak=true -f null -` for integrated loudness and true peak, plus RMS per section (dialogue, music-only, end card). An end card 15 dB under the dialogue, or effects louder than speech, is a bug. You cannot listen: say so, and report the numbers.

   For anything the user will publish (launch, promo, ad), also spawn one **critic sub-agent** with the rendered file, the EDL, and any reference videos the user gave. Brief it to roast, not to praise: a verdict, ranked problems with timecodes and evidence (frames, levels), and the 5 fixes to do first. Fresh eyes catch what the author stopped seeing — cut-off payoff lines, 0.5s memes, unreadable 28px text at phone size.

   If anything fails: fix → re-render → re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only present the preview once the self-eval passes.
8. **Iterate + persist.** Natural-language feedback, re-plan, re-render. Never re-transcribe. Final render on confirmation. Append to `project.md`.

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

**`natural-sentence`** (if you invent this mode) — narrative, documentary, education. 4–7 word chunks, sentence case, break on natural pauses, `MarginV=60–80`, larger font for readability, slightly wider max-width. No shipped force_style — design one if you need it.

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
- **Master once:** mix to PCM, then two-pass loudnorm (-14 LUFS, true peak ≤ -1 dBTP) on the final mix. Then measure per section (see Self-eval).
- **Music taste is the user's call.** Generated music defaults to "hype"; offer two contrasting beds and let the user listen. Don't claim a mix sounds good — you can only measure it.

## Output spec

Match the source unless the user asked for something specific. Common targets: `1920×1080@24` cinematic, `1920×1080@30` screen content, `1080×1920@30` vertical social, `3840×2160@24` 4K cinema, `1080×1080@30` square. `render.py` defaults the scale to 1080p from any source; pass `--filter` or edit the extract command for other targets. Worth asking the user which delivery format matters.

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
  "overlays": [
    {"file": "edit/animations/slot_1/render.mp4", "start_in_output": 0.0, "duration": 5.0}
  ],
  "subtitles": "edit/master.srt",
  "total_duration_s": 87.4
}
```

`grade` is a preset name or raw ffmpeg filter. `lut` is optional, a name from `lut/` or a path to a `.cube` file — `--lut` on the command line overrides it. `overlays` are rendered animation clips. `subtitles` is optional and applied LAST. Profanity censoring (`--censor-profanity`) is a render-time CLI flag, not an EDL field — it's a delivery-format choice (clean cut vs. uncensored), not an editorial decision baked into the cut itself.

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
| `grade` / `lut` | ⚠️ **not applied.** No portable way to embed a third-party color node both apps load identically — faking one would silently do nothing. Written as a comment on the sequence and on affected clips instead (exact preset/filter string, or the resolved `.cube` path to load on the color page by hand). |
| `subtitles` | ⚠️ **not embedded** — xmeml predates any caption element both apps agree on. Left as the sidecar `.srt`; both apps import it directly (File → Import, or drag onto the timeline). |

Source rate handling is per-source (each clip's `in`/`out` uses its own file's native frame rate even in a mixed-rate EDL), while the timeline position (`start`/`end`) uses one sequence rate — same frame-rate resolution policy as `render.py` (`--fps` to override, else probes the first source).

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
