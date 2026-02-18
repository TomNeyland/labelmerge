from __future__ import annotations


def strip_stop_words(text: str, stop_words: list[str]) -> str:
    """Remove stop words from text, preserving word order.

    Returns the text with stop words removed. If all words are stop words,
    returns the original text unchanged (we still need something to embed).
    """
    words = text.split()
    filtered = [w for w in words if w.lower() not in {s.lower() for s in stop_words}]
    if not filtered:
        return text
    return " ".join(filtered)
