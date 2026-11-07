"""Fast tests for text_stats (no model)."""

from harm_route.text_stats import distinct_ngram_ratio, dominant_script, max_line_repeat


def test_distinct_ngram_ratio():
    assert distinct_ngram_ratio(list(range(10))) == 1.0  # all unique
    assert distinct_ngram_ratio([1, 2] * 50) < 0.05  # 2-token loop: only 2 distinct 4-grams
    assert distinct_ngram_ratio([1, 2, 3]) is None  # shorter than n


def test_max_line_repeat():
    assert max_line_repeat("") == 0
    assert max_line_repeat("a\nb\n\nc") == 1
    assert max_line_repeat("I cannot.\nI cannot.\n  I cannot.  \nok") == 3


def test_dominant_script():
    assert dominant_script("Hello, world 123") == ("LATIN", 1.0)
    s, share = dominant_script("नमस्ते hi")  # Devanagari letters outnumber Latin
    assert s == "DEVANAGARI" and 0.5 < share < 1
    assert dominant_script("123 !!") == (None, 0.0)
