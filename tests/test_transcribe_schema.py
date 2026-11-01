"""The local backend must emit exactly what the Scribe consumers already read.

`pack_transcripts.py` and `render.py` were written against ElevenLabs Scribe
responses. If `to_scribe_schema` drifts from that shape, cuts land on the wrong
frame and nothing fails loudly - so pin the contract here. These tests import
only the pure converter, so they run without torch, CUDA or a model download.
"""

import importlib.util
import unittest
from pathlib import Path


def _load(name, filename):
    path = Path(__file__).parents[1] / "helpers" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


local = _load("video_use_transcribe_local", "transcribe_local.py")
pack = _load("video_use_pack", "pack_transcripts.py")
render = _load("video_use_render", "render.py")


def seg(words, speaker=None):
    out = {"start": words[0][1], "end": words[-1][2], "words": [], "text": " ".join(w[0] for w in words)}
    for text, start, end in words:
        entry = {"word": text, "start": start, "end": end, "score": 0.9}
        if speaker:
            entry["speaker"] = speaker
        out["words"].append(entry)
    return out


# A verbatim take as CrisperWhisper would return it, filler included.
SEGMENTS = [
    seg([("So", 0.20, 0.38), ("um,", 0.44, 0.91), ("we", 1.02, 1.14), ("fix", 1.18, 1.46),
         ("this.", 1.50, 1.92)], speaker="SPEAKER_00"),
    seg([("Right.", 3.10, 3.60)], speaker="SPEAKER_01"),
]


class SpeakerNormalization(unittest.TestCase):
    def test_pyannote_label_becomes_scribe_label(self):
        # pack_transcripts strips a literal "speaker_" prefix to render the tag,
        # so a raw pyannote label would print as "SSPEAKER_00".
        self.assertEqual(local.normalize_speaker("SPEAKER_00"), "speaker_0")
        self.assertEqual(local.normalize_speaker("SPEAKER_07"), "speaker_7")

    def test_already_scribe_shaped_is_left_alone(self):
        self.assertEqual(local.normalize_speaker("speaker_3"), "speaker_3")

    def test_none_stays_none(self):
        self.assertIsNone(local.normalize_speaker(None))


class ScribeShape(unittest.TestCase):
    def setUp(self):
        self.out = local.to_scribe_schema(SEGMENTS, language_code="en")

    def test_top_level_keys(self):
        self.assertIn("words", self.out)
        self.assertIn("text", self.out)
        self.assertEqual(self.out["language_code"], "en")

    def test_only_known_entry_types(self):
        types = {w["type"] for w in self.out["words"]}
        self.assertTrue(types <= {"word", "spacing", "audio_event"}, types)

    def test_every_entry_has_timestamps(self):
        for w in self.out["words"]:
            self.assertIsNotNone(w.get("start"), w)
            self.assertIsNotNone(w.get("end"), w)
            self.assertGreaterEqual(w["end"], w["start"], w)

    def test_entries_are_time_ordered(self):
        starts = [w["start"] for w in self.out["words"]]
        self.assertEqual(starts, sorted(starts))

    def test_filler_survives_verbatim(self):
        spoken = [w["text"] for w in self.out["words"] if w["type"] == "word"]
        self.assertIn("um,", spoken)

    def test_punctuation_stays_attached(self):
        # render.chunk_words closes a caption cue on trailing punctuation; split
        # punctuation would break every cue boundary.
        spoken = [w["text"] for w in self.out["words"] if w["type"] == "word"]
        self.assertIn("this.", spoken)

    def test_spacing_entries_carry_the_gaps(self):
        spacing = [w for w in self.out["words"] if w["type"] == "spacing"]
        self.assertTrue(spacing)
        # The 1.18s pause between takes must appear as one spacing entry, or
        # pack_transcripts cannot find the phrase break.
        self.assertTrue(any(round(s["end"] - s["start"], 2) == 1.18 for s in spacing),
                        [round(s["end"] - s["start"], 2) for s in spacing])

    def test_speakers_are_per_word(self):
        by_text = {w["text"]: w.get("speaker_id")
                   for w in self.out["words"] if w["type"] == "word"}
        self.assertEqual(by_text["So"], "speaker_0")
        self.assertEqual(by_text["Right."], "speaker_1")


class AudioEvents(unittest.TestCase):
    def test_events_are_parenthesized_and_interleaved(self):
        out = local.to_scribe_schema(
            SEGMENTS, language_code="en",
            events=[{"text": "laughter", "start": 2.1, "end": 2.9}],
        )
        events = [w for w in out["words"] if w["type"] == "audio_event"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["text"], "(laughter)")

        idx = out["words"].index(events[0])
        self.assertLess(out["words"][idx - 1]["start"], 2.1)
        self.assertGreater(out["words"][idx + 1]["start"], 2.1)

    def test_already_wrapped_label_is_not_double_wrapped(self):
        out = local.to_scribe_schema(
            SEGMENTS, language_code="en",
            events=[{"text": "(applause)", "start": 2.1, "end": 2.9}],
        )
        events = [w for w in out["words"] if w["type"] == "audio_event"]
        self.assertEqual(events[0]["text"], "(applause)")

    def test_event_inherits_the_nearest_speaker(self):
        # pack_transcripts takes a phrase's speaker from its first entry, so an
        # untagged laugh just before a line would strip that line's speaker.
        out = local.to_scribe_schema(
            SEGMENTS, language_code="en",
            events=[{"text": "laughter", "start": 2.1, "end": 2.9}],
        )
        # 2.1 is 0.18s after speaker_0 stops and 1.0s before speaker_1 starts.
        event = next(w for w in out["words"] if w["type"] == "audio_event")
        self.assertEqual(event["speaker_id"], "speaker_0")

    def test_event_has_no_speaker_when_diarization_is_off(self):
        undiarized = [seg([("hello", 0.1, 0.5)])]
        out = local.to_scribe_schema(
            undiarized, language_code="en",
            events=[{"text": "laughter", "start": 1.0, "end": 1.8}],
        )
        event = next(w for w in out["words"] if w["type"] == "audio_event")
        self.assertNotIn("speaker_id", event)

    def test_events_do_not_become_words(self):
        # An audio event in the word stream would get burned into a caption.
        out = local.to_scribe_schema(
            SEGMENTS, language_code="en",
            events=[{"text": "laughter", "start": 2.1, "end": 2.9}],
        )
        self.assertNotIn("(laughter)", out["text"])


class UnalignableWords(unittest.TestCase):
    """WhisperX omits start/end for a word with no character in the wav2vec2
    dictionary. Downstream silently skips such a word, so fill it in."""

    def test_missing_timestamps_are_filled_from_neighbours(self):
        segments = [{
            "start": 0.0, "end": 2.0, "text": "hello ... world",
            "words": [
                {"word": "hello", "start": 0.1, "end": 0.5, "score": 0.9},
                {"word": "...", "score": 0.0},
                {"word": "world", "start": 1.5, "end": 1.9, "score": 0.9},
            ],
        }]
        out = local.to_scribe_schema(segments, language_code="en")
        spoken = [w for w in out["words"] if w["type"] == "word"]
        self.assertEqual([w["text"] for w in spoken], ["hello", "...", "world"])
        filler = spoken[1]
        self.assertGreaterEqual(filler["start"], 0.5)
        self.assertLessEqual(filler["end"], 1.5)

    def test_all_timestamps_missing_does_not_crash(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "hm",
                     "words": [{"word": "hm"}]}]
        out = local.to_scribe_schema(segments, language_code="en")
        self.assertEqual(out["words"][0]["start"], 0.0)


class DownstreamConsumers(unittest.TestCase):
    """The real contract: the existing readers must handle the output unchanged."""

    def setUp(self):
        self.out = local.to_scribe_schema(
            SEGMENTS, language_code="en",
            events=[{"text": "laughter", "start": 2.1, "end": 2.9}],
        )

    def test_pack_transcripts_groups_into_phrases(self):
        phrases = pack.group_into_phrases(self.out["words"], silence_threshold=0.5)
        # The 1.18s gap and the speaker change both fall between "this." and
        # "Right.", so those cannot land in one phrase.
        self.assertGreaterEqual(len(phrases), 2)
        self.assertIn("um,", phrases[0]["text"])
        self.assertEqual(phrases[0]["speaker_id"], "speaker_0")
        self.assertEqual(phrases[-1]["speaker_id"], "speaker_1")

    def test_render_selects_words_in_range(self):
        words = render._words_in_range(self.out, 0.0, 2.0)
        self.assertEqual([w["text"] for w in words],
                         ["So", "um,", "we", "fix", "this."])

    def test_filler_tags_never_reach_the_captions(self):
        # CrisperWhisper writes fillers as [UH] / [UM]. Keeping them in the
        # transcript is the point; burning them onto the picture is a bug.
        segments = [seg([("So", 0.2, 0.38), ("[UH]", 0.44, 0.91),
                         ("we", 1.02, 1.14), ("fix", 1.18, 1.46)])]
        out = local.to_scribe_schema(segments, language_code="en")

        spoken = [w["text"] for w in out["words"] if w["type"] == "word"]
        self.assertIn("[UH]", spoken)

        caption_words = render._words_in_range(out, 0.0, 2.0)
        self.assertEqual([w["text"] for w in caption_words], ["So", "we", "fix"])

    def test_nonspeech_token_detection(self):
        for tag in ("[UH]", "[UM]", "(laughter)", "[MM]", "<inaudible>", "[UH],"):
            self.assertTrue(render.is_nonspeech_token(tag), tag)
        for word in ("So", "we", "fix", "this.", "pres", "our"):
            self.assertFalse(render.is_nonspeech_token(word), word)

    def test_render_chunks_into_caption_cues(self):
        words = render._words_in_range(self.out, 0.0, 2.0)
        chunks = render.chunk_words(words)
        self.assertTrue(chunks)
        flat = [w["text"] for c in chunks for w in c]
        self.assertEqual(flat, ["So", "um,", "we", "fix", "this."])


if __name__ == "__main__":
    unittest.main()
