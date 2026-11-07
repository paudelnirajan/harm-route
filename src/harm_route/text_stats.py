"""Counts-only quality stats for generations: repetition (degeneracy) and output script.

Language-agnostic by design: repetition is measured on model token ids (works for scripts without spaces),
script comes from Unicode character names. Nothing here returns or prints text.
"""

import unicodedata
from collections import Counter


def distinct_ngram_ratio(ids: list[int], n: int = 4) -> float | None:
    """Unique n-grams / total n-grams. ~0.9+ for normal text, low for loops. None if too short."""
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def max_line_repeat(text: str) -> int:
    """How many times the most common non-empty line occurs (1 = no repeated lines, 0 = empty)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return max(Counter(lines).values()) if lines else 0


def char_script(ch: str) -> str | None:
    """Unicode script-ish label of a letter from its name ('LATIN', 'DEVANAGARI', 'CJK', ...). None if not a letter."""
    if not ch.isalpha():
        return None
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:  # unnamed code point
        return "UNKNOWN"


def dominant_script(text: str) -> tuple[str | None, float]:
    """Most common script among letters and its share of letters. (None, 0.0) if no letters."""
    c = Counter(s for s in map(char_script, text) if s)
    if not c:
        return None, 0.0
    script, k = c.most_common(1)[0]
    return script, k / sum(c.values())
