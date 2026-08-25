"""Shingling and Jaccard similarity for near-duplicate detection.

Word-level shingles rather than character n-grams: an agent transcript's
near-duplicates differ by a token or two (a timestamp, a line number, a
retry count), not by a handful of characters. A word shingle set absorbs
that kind of edit cleanly, since only the shingles touching the changed
word move; character n-grams would treat the same edit as noise smeared
across a much larger fraction of the set.
"""

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text):
    return _WORD_RE.findall(text.lower())


def shingles(text, size=5):
    """Return the set of ``size``-word shingles in ``text``.

    A shingle is a tuple of ``size`` consecutive lowercased words, so
    comparison is insensitive to case and to punctuation between words.
    Text with fewer than ``size`` words yields a single shingle covering
    everything it has, so short segments can still be compared instead of
    always coming back empty.
    """
    words = _words(text)
    if not words:
        return set()
    if len(words) <= size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a, b):
    """Jaccard similarity between two shingle sets: |intersection| / |union|.

    Two empty sets are treated as identical (1.0) rather than undefined,
    since two segments that produced no shingles are indistinguishable to
    this measure.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
