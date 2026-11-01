# Rendering Reference

## Prerequisites

```bash
manim --version       # Manim CE
pdflatex --version    # LaTeX
ffmpeg -version       # ffmpeg
```

## CLI Reference

```bash
manim -ql script.py Scene1 Scene2    # draft (480p 15fps)
manim -qm script.py Scene1           # medium (720p 30fps)
manim -qh script.py Scene1           # production (1080p 60fps)
manim -ql --format=png -s script.py Scene1  # preview still (last frame)
manim -ql --format=gif script.py Scene1     # GIF output
```

## Quality Presets

| Flag | Resolution | FPS | Use case |
|------|-----------|-----|----------|
| `-ql` | 854x480 | 15 | Draft iteration (layout, timing) |
| `-qm` | 1280x720 | 30 | Preview (use for text-heavy scenes) |
| `-qh` | 1920x1080 | 60 | Production |

**Text rendering quality:** `-ql` (480p15) produces noticeably poor text kerning and readability. For scenes with significant text, preview stills at `-qm` to catch issues invisible at 480p. Use `-ql` only for testing layout and animation timing.

## Output Structure

```
media/videos/script/480p15/Scene1_Intro.mp4
media/images/script/Scene1_Intro.png  (from -s flag)
```

## Stitching with ffmpeg

```bash
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1_Intro.mp4'
file 'media/videos/script/480p15/Scene2_Core.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

## Add Voiceover

```bash
# Mux narration
ffmpeg -y -i final.mp4 -i narration.mp3 -c:v copy -c:a aac -b:a 192k -shortest final_narrated.mp4

# Concat per-scene audio first
cat > audio_concat.txt << 'EOF'
file 'audio/scene1.mp3'
file 'audio/scene2.mp3'
EOF
ffmpeg -y -f concat -safe 0 -i audio_concat.txt -c copy full_narration.mp3
```

## Add Background Music

```bash
ffmpeg -y -i final.mp4 -i music.mp3 \
  -filter_complex "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=shortest" \
  -c:v copy final_with_music.mp4
```

## GIF Export

```bash
ffmpeg -y -i scene.mp4 \
  -vf "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  output.gif
```

## Aspect Ratios

```bash
manim -ql --resolution 1080,1920 script.py Scene  # 9:16 vertical
manim -ql --resolution 1080,1080 script.py Scene  # 1:1 square
```

## Render Workflow

1. Draft render all scenes at `-ql`
2. Preview stills at key moments (`-s`)
3. Fix and re-render only broken scenes
4. Stitch with ffmpeg
5. Review stitched output
6. Production render at `-qh`
7. Re-stitch + add audio

## manim.cfg — Project Configuration

Create `manim.cfg` in the project directory for per-project defaults:

```ini
[CLI]
quality = low_quality
preview = True
media_dir = ./media

[renderer]
background_color = #0D1117

[tex]
tex_template_file = custom_template.tex
```

This eliminates repetitive CLI flags and `self.camera.background_color` in every scene.

## Sections — Chapter Markers

Mark sections within a scene for organized output:

```python
class LongVideo(Scene):
    def construct(self):
        self.next_section("Introduction")
        # ... intro content ...

        self.next_section("Main Concept")
        # ... main content ...

        self.next_section("Conclusion")
        # ... closing ...
```

Render individual sections: `manim --save_sections script.py LongVideo`
This outputs separate video files per section — useful for long videos where you want to re-render only one part.

## manim-voiceover Plugin (Recommended for Narrated Videos)

The official `manim-voiceover` plugin integrates TTS directly into scene code, auto-syncing animation duration to voiceover length. This is significantly cleaner than the manual ffmpeg muxing approach above.

### Installation

```bash
# Local and offline — the default for this repo. Apache-2.0, 82M params, runs on CPU
# in realtime and on any GPU far faster. Nothing leaves the machine.
pip install manim-voiceover kokoro soundfile

# Hosted alternatives, if a project specifically wants one:
pip install "manim-voiceover[gtts]"        # Google TTS (free, needs network)
pip install "manim-voiceover[elevenlabs]"  # ElevenLabs (best quality, costs credits)
pip install "manim-voiceover[azure]"       # Azure Cognitive Services
```

Two Kokoro gotchas: `manim-voiceover` 0.4.0 needs Python 3.11+, and Kokoro falls back to
`espeak-ng` for words its dictionary doesn't cover — install it (`brew install espeak-ng`,
`apt-get install espeak-ng`, or the Windows MSI) or unusual proper nouns come out garbled.

### Usage

manim-voiceover has no Kokoro service built in, so define one. It is ~30 lines, and every
other feature on this page (`tracker.duration`, bookmarks, caching) works through it unchanged,
because the base class handles all of that once `generate_from_text` returns a `VoiceoverData`.

Three details are easy to get wrong, and all three fail quietly:
`initialize_speech_service(self, kwargs)` instead of `super().__init__()`; the returned dict
must carry `input_data` as well as `original_audio`, or caching never hits; and
`remove_bookmarks()` must run before synthesis, or the voice reads the `<bookmark>` tags aloud.

```python
# kokoro_service.py — written against manim-voiceover 0.4.0
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline
from manim_voiceover._typing import VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import (
    PathLike,
    SpeechService,
    initialize_speech_service,
    path_to_string,
)

SAMPLE_RATE = 24000


class KokoroService(SpeechService):
    """Offline TTS via Kokoro-82M. Apache-2.0, runs on CPU or GPU.

    voice: af_heart, af_bella, am_michael, bf_emma, ... (see the model's VOICES.md)
    lang_code: 'a' US English, 'b' UK English, 'e' Spanish, 'f' French, 'j' Japanese, 'z' Chinese
    """

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        **kwargs: object,
    ) -> None:
        initialize_speech_service(self, kwargs)
        self.voice = voice
        self.speed = speed
        self._pipeline = KPipeline(lang_code=lang_code)

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> VoiceoverData:
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        # Voice and speed belong in the cache key: change either and the line must
        # re-render, or you get the old voice for the rest of the project.
        input_data = {
            "input_text": input_text,
            "service": "kokoro",
            "voice": self.voice,
            "speed": self.speed,
        }

        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".wav"
        else:
            audio_path = path_to_string(path)

        # Kokoro splits long input into segments and yields one per segment.
        chunks = [
            np.asarray(audio)
            for _, _, audio in self._pipeline(input_text, voice=self.voice, speed=self.speed)
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {input_text[:80]!r}")
        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        sf.write(str(Path(cache_dir) / audio_path), audio, SAMPLE_RATE)

        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }
```

Then use it exactly like any hosted service:

```python
from manim import *
from manim_voiceover import VoiceoverScene
from kokoro_service import KokoroService

class NarratedScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(KokoroService(voice="af_heart"))

        # Voiceover auto-controls scene duration
        with self.voiceover(text="Here is a circle being drawn.") as tracker:
            self.play(Create(Circle()), run_time=tracker.duration)

        with self.voiceover(text="Now let's transform it into a square.") as tracker:
            self.play(Transform(circle, Square()), run_time=tracker.duration)
```

### Key Features

- `tracker.duration` — total voiceover duration in seconds
- `tracker.time_until_bookmark("mark1")` — sync specific animations to specific words
- Auto-generates subtitle `.srt` files
- Caches audio locally — re-renders don't re-generate TTS
- Works with: Kokoro (offline, the local default above), pyttsx3 (offline, robotic), gTTS, Azure, ElevenLabs
- Caching is keyed on the text *and* the service config, so switching voice or engine re-renders only the lines that changed

### Bookmarks for Precise Sync

```python
with self.voiceover(text='This is a <bookmark mark="circle"/>circle.') as tracker:
    self.wait_until_bookmark("circle")
    self.play(Create(Circle()), run_time=tracker.time_until_bookmark("circle", limit=1))
```

This is the recommended approach for any video with narration. Prefer Kokoro unless the project has a reason to go hosted — it is free, offline, and good enough that the difference is not what makes a narrated explainer work. The manual ffmpeg muxing workflow above is still useful for adding background music or post-production audio mixing.
