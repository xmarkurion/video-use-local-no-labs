"""Profanity detection + redaction, for `render.py --censor-profanity`.

Module deliberately NOT named `profanity.py`: the repo also has a top-level
`profanity/` data folder (the wordlist), and when a caller's sys.path puts
the repo root ahead of helpers/ (pytest does), `import profanity` would
silently resolve to that folder as an empty PEP 420 namespace package
instead of this file — no error, just missing attributes. Different names,
no ambiguity possible.

Stems live in a plain-text wordlist (default: `profanity/wordlist.txt` at the
repo root) so moderation is a data edit, not a code change. Matching is
stem + common-inflection, not substring: "ass" matches "ass" / "asses" /
"assed", never "assume" or "class" or "passenger". That trade favors missing
an unlisted word over mangling an innocent one — widen the wordlist yourself
for stricter coverage; this module just applies whatever list it's given.

Two call sites in render.py:
  - captions: `censor_word` replaces the profane core with literal "***",
    keeps surrounding punctuation.
  - audio: `is_profane` flags which transcript words to mute/bleep, by
    timestamp, in the rendered mix.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_WORDLIST = Path(__file__).resolve().parent.parent / "profanity" / "wordlist.txt"

# Suffixes that keep a stem profane: "fuck" -> "fucking", "fucked", "fucker",
# "fuckers", "fucks". Kept short on purpose — this is a filter for a cut, not
# a lemmatizer; it will miss "fuckery" and that's fine.
_INFLECTIONS = ("ing", "ers", "er", "ed", "es", "s")

# A token's "core": leading/trailing punctuation stripped, internal
# apostrophes and hyphens kept (so "mother-fucker" and "fuckin'" still match
# as one unit). Non-greedy core so trailing punctuation isn't eaten by it.
_WORD_CORE = re.compile(r"^([^\w]*)([\w'-]+?)([^\w]*)$", re.UNICODE)


def load_profanity_stems(path: Path | None = None) -> list[str]:
    """Read one stem per line from a wordlist file. Blank lines and lines
    starting with `#` are skipped. A missing file returns an empty list
    rather than raising — an empty list is a no-op filter, which is the
    right behavior if someone deletes the wordlist rather than a crash."""
    path = path or DEFAULT_WORDLIST
    if not path.exists():
        return []
    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            stems.append(line)
    return stems


def build_profanity_matcher(stems: list[str]) -> re.Pattern | None:
    """Compile stems into one pattern: `^(?:stem1|stem2|...)(?:ing|ers|...)?$`.

    Longest-first alternation so a stem that is itself a prefix of another
    stem (none ship by default, but a custom wordlist might add one) can't
    shadow the more specific match.
    """
    if not stems:
        return None
    ordered = sorted({s.strip().lower() for s in stems if s.strip()}, key=len, reverse=True)
    if not ordered:
        return None
    stem_group = "|".join(re.escape(s) for s in ordered)
    suffix_group = "|".join(_INFLECTIONS)
    return re.compile(rf"^(?:{stem_group})(?:{suffix_group})?$", re.IGNORECASE)


def _core(text: str) -> str:
    """The alphanumeric heart of a token, punctuation stripped: "fuck," -> "fuck"."""
    m = _WORD_CORE.match(text)
    return m.group(2) if m else text


def is_profane(text: str, matcher: re.Pattern | None) -> bool:
    """True if `text`'s word-core matches the compiled matcher."""
    if matcher is None or not text:
        return False
    return bool(matcher.match(_core(text).lower()))


def censor_word(text: str, matcher: re.Pattern | None) -> str:
    """Replace a profane core with literal "***"; everything else is untouched.

    "fuck," -> "***,"   "Fucking!" -> "***!"   "duck" -> "duck" (no match)
    """
    if matcher is None or not text:
        return text
    m = _WORD_CORE.match(text)
    if not m:
        return text
    lead, core, trail = m.groups()
    if matcher.match(core.lower()):
        return f"{lead}***{trail}"
    return text
