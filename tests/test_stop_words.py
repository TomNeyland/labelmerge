from __future__ import annotations

from semdedup._stop_words import strip_stop_words


def test_strip_basic():
    result = strip_stop_words("serum cortisol concentration", ["serum", "concentration"])
    assert result == "cortisol"


def test_strip_preserves_order():
    result = strip_stop_words("hello big world", ["big"])
    assert result == "hello world"


def test_strip_case_insensitive():
    result = strip_stop_words("Serum Cortisol", ["serum"])
    assert result == "Cortisol"


def test_strip_all_words_returns_original():
    result = strip_stop_words("concentration level", ["concentration", "level"])
    assert result == "concentration level"


def test_strip_no_stop_words():
    result = strip_stop_words("hello world", [])
    assert result == "hello world"
