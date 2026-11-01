---
name: video-use-install
description: Install video-use into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and wire up ffmpeg + local GPU transcription so the user can start editing immediately.
---

# video-use install

Use this file only for first-time install or reconnect. For daily editing, read `SKILL.md`. Always read `helpers/` — that's where the scripts live.

## What you're doing

You're setting up a conversation-driven video editor for the user. After install, the user drops raw footage into any folder, runs their agent (`claude`, `codex`, etc.) there, and says "edit these into a launch video." You do the rest by reading `SKILL.md`.

Three things must exist on this machine:

1. The `video-use` repo cloned somewhere stable.
2. `ffmpeg` on `$PATH` (plus optional `yt-dlp` for online sources).
3. The `local` extra installed, for GPU transcription (`pip install -e ".[local]"`). No API key, no account, no quota.

And one thing must be true about the current agent:

4. It can discover `SKILL.md` — either via a global skills directory (`~/.claude/skills/`, `~/.codex/skills/`) or via a `CLAUDE.md` / system-prompt import.

## Install prompt contract

- Do everything yourself. Nothing here needs a secret. Only ask the user for confirmation before a package-manager install, and for a HuggingFace token *if* they want speaker diarization.
- Prefer a stable clone path like `~/Developer/video-use` (not `/tmp`, not `~/Downloads`).
- The skill references helpers by bare name (`transcribe.py`, `render.py`). That works because SKILL.md and `helpers/` ship together — keep them as siblings when you register the skill.
- After install, verify by running one real command against one real file. Don't declare success on file-existence checks alone.

## Steps

### 1. Clone to a stable path

```bash
test -d ~/Developer/video-use || git clone https://github.com/browser-use/video-use ~/Developer/video-use
cd ~/Developer/video-use
```

If the repo is already there, `git pull --ff-only` and continue.

### 2. Install Python deps

```bash
# Prefer uv if available; fall back to pip. Note the [local] extra.
command -v uv >/dev/null && uv sync --extra local || pip install -e ".[local]"
```

Base deps are `requests`, `librosa`, `matplotlib`, `pillow`, `numpy`. The `local` extra adds `whisperx` (which pulls `faster-whisper`, `ctranslate2`, `pyannote-audio` and `torch`) and `transformers`. That's ~3GB of wheels, mostly torch — say so before starting, it is not a fast step.

No console scripts — helpers are invoked directly as `python helpers/<name>.py`.

If `pip` resolves a CPU-only torch on a machine that has an NVIDIA GPU, install torch from the CUDA index first, then the extra:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[local]"
```

### 3. Install ffmpeg (+ optional yt-dlp)

`ffmpeg` and `ffprobe` are hard requirements. `yt-dlp` is only needed if the user wants to pull sources from URLs. Animation engines such as HyperFrames, Remotion, and Manim are installed lazily the first time a project actually needs them.

```bash
# macOS
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v yt-dlp >/dev/null || brew install yt-dlp     # optional

# Debian / Ubuntu
# sudo apt-get update && sudo apt-get install -y ffmpeg
# pip install yt-dlp

# Arch
# sudo pacman -S ffmpeg yt-dlp
```

If `brew` / `apt` / `pacman` requires a sudo prompt, tell the user the exact command and wait. Do not invent a password.

### 3b. Confirm the GPU is usable

Transcription runs locally on the GPU. Check what the machine actually has:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

- **8GB VRAM or more**: defaults are right. `float16` CrisperWhisper is ~3GB, plus ~0.4GB alignment, ~1GB diarization, ~0.4GB events.
- **6–8GB**: pass `--compute-type int8_float16` and `--batch-size 4`.
- **Under 6GB, or no CUDA**: it still runs, on CPU, slowly. Say so plainly and offer `--model large-v3-turbo` or `--backend elevenlabs` instead of letting the user discover it by waiting.
- **CUDA toolkit version does not have to match.** The pip wheels carry their own CUDA runtime; only the driver has to be recent enough. Do not install a system CUDA toolkit for this.

If `torch.cuda.is_available()` is `False` on a machine with an NVIDIA card, it is almost always a CPU-only torch wheel. Reinstall from the CUDA index (see step 2) rather than debugging further.

### 4. Register the skill with the current agent

Figure out which agent you are running under, and register once. A symlink of the whole repo directory is the right shape — helpers/ needs to sit next to SKILL.md.

- **Claude Code** (`~/.claude/` present):

    ```bash
    mkdir -p ~/.claude/skills
    ln -sfn ~/Developer/video-use ~/.claude/skills/video-use
    ```

- **Codex** (`$CODEX_HOME` set, or `~/.codex/` present):

    ```bash
    mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
    ln -sfn ~/Developer/video-use "${CODEX_HOME:-$HOME/.codex}/skills/video-use"
    ```

- **Hermes / Openclaw / another agent with a skills directory**: symlink `~/Developer/video-use` into that agent's skills directory under the name `video-use`. If the agent has no skills directory, add a line to its system prompt / config pointing at `~/Developer/video-use/SKILL.md` (e.g. an `@~/Developer/video-use/SKILL.md` import in a `CLAUDE.md`-equivalent).

If you can't tell which agent you're in, ask the user once: "which agent am I running under — Claude Code, Codex, or something else?" Then pick the right target.

### 5. Optional keys (there is no required one)

Local transcription needs no account. Two optional keys unlock extras — do not block the install on either.

**Speaker diarization (`--diarize`)** needs a HuggingFace read token, because pyannote's pipeline is gated.

1. Check whether it's already there:

    ```bash
    [ -n "$HF_TOKEN" ] && echo "env"
    grep -q '^HF_TOKEN=..' ~/Developer/video-use/.env 2>/dev/null && echo "dotenv"
    ```

2. If the user's footage is single-speaker (most talking-head and tutorial material), skip this entirely — don't ask. If it's interviews or multi-speaker, ask once:

    > Multi-speaker footage transcribes better with speaker labels. That needs a free HuggingFace token: accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1, then paste a read token from https://huggingface.co/settings/tokens. Skip it and I'll transcribe without speaker labels.

    ```bash
    printf 'HF_TOKEN=%s
' "$TOKEN" >> ~/Developer/video-use/.env
    chmod 600 ~/Developer/video-use/.env
    ```

**Hosted Scribe fallback (`--backend elevenlabs`)** needs `ELEVENLABS_API_KEY`. Only set it up if the user asks for it — it costs credits and the local backend is the default. Same `.env` file, same `chmod 600`.

Never echo a key or token back in tool output. Never commit `.env`.

### 6. Verify end-to-end

```bash
python ~/Developer/video-use/helpers/timeline_view.py --help >/dev/null && echo "helpers OK"
ffprobe -version | head -1
python -m pytest ~/Developer/video-use/tests -q
```

Then prove the transcription path really works, because this is the step that breaks. Unlike the old hosted setup, a real test is free — it costs one model download (~1.5GB, cached forever) and a few seconds of GPU:

```bash
# 5 seconds of generated speech is enough to exercise ffmpeg -> ASR -> alignment -> JSON.
cd /tmp && ffmpeg -y -f lavfi -i "sine=frequency=200:duration=5" -f lavfi   -i "color=c=black:s=320x240:d=5" -shortest -pix_fmt yuv420p probe.mp4 2>/dev/null
python ~/Developer/video-use/helpers/transcribe.py /tmp/probe.mp4
```

A sine tone has no speech, so **expect this to fail the silence guard or return zero words — that is a pass.** It proves ffmpeg extraction, model load and CUDA all work. If the user has a real clip handy, run it on that instead and check `edit/transcripts/*.json` has a `words` array with `start`/`end` on each entry.

If the model download is the slow part, warm it up explicitly so the first real edit isn't the one that waits:

```bash
python -c "import whisperx; whisperx.load_model('nyralabs/faster_CrisperWhisper', device='cuda', compute_type='float16')"
```

### 7. Hand off

Tell the user, in one short message:

- Where the skill is installed (`~/Developer/video-use`).
- That they should `cd` into their footage folder and start their agent there (e.g. `claude`).
- That a good first message is: *"edit these into a launch video"* or *"inventory these takes and propose a strategy."*
- That all outputs land in `<videos_dir>/edit/` — the repo stays clean.
- That transcription runs on their own GPU: no key, no quota, no per-minute cost, and footage never leaves the machine.

## Keeping the skill current

- `cd ~/Developer/video-use && git pull --ff-only` pulls the latest code. The symlink auto-picks it up on the next run.
- If `pyproject.toml` changed deps, re-run `uv sync --extra local` / `pip install -e ".[local]"` after pulling.

## Cold-start reminders

- Symlink the **whole directory**, not just `SKILL.md`. The helpers need to sit next to it.
- `.env` is entirely optional now. An empty or absent `HF_TOKEN` just means no speaker labels; don't treat it as a failed install.
- An empty key is the same as a missing one — don't assume existence means validity.
- `ffmpeg` from static builds works fine. Any modern (≥ 4.x) build is enough.
- **Ignore the `libtorchcodec` warning.** `torchcodec` links FFmpeg 4–7 and fails to load against FFmpeg 8, so pyannote prints a loading traceback on import. Nothing in this repo decodes through it: `whisperx.load_audio` and `audio_events.py` both shell out to the `ffmpeg` CLI, and `transcribe_local.py` hands every stage a decoded array rather than a path. Don't downgrade ffmpeg to silence it.
- `yt-dlp` is optional. Don't block install on it; install lazily the first time a user asks to pull from a URL.
- Node.js/npm are only needed for HyperFrames or Remotion slots. HyperFrames currently requires Node.js 22+.
- HyperFrames, Remotion, and Manim are optional animation engines. Don't install or prefer one globally during setup; pick the engine per animation slot in `SKILL.md`. HyperFrames can run through `npx --yes hyperframes ...` in the slot directory. Remotion can be scaffolded with `npx create-video@latest` or installed inside the slot before rendering.
- Do run a transcription check at install time. It is free on the local backend, and the model download is the one slow surprise worth getting out of the way.
- Model weights land in the HuggingFace cache (`~/.cache/huggingface`), not in the repo. ~1.5GB for CrisperWhisper, ~0.4GB each for alignment and diarization.
- CrisperWhisper's weights are CC-BY-NC-4.0 — research and personal use. If the user is editing commercial work, tell them once: either license it from nyra labs or pass `--model large-v3` (MIT, but it deletes fillers).
- If the user is on Linux without a package manager Claude recognizes, print the manual `ffmpeg` install URL and wait rather than guessing.
