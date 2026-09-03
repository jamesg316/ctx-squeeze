"""TF-IDF keyword density scoring and budget-constrained selection.

Segments are the unit, not sentences or tokens: a segment is already a
whole paragraph or a whole code block, so scoring it and keeping or
dropping it whole is what lets ``select_by_score`` guarantee it never
produces a half-sentence or a broken fence.
"""

import math
import re

from ctx_squeeze.tokens import estimate_tokens

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Small, hand-picked rather than pulled from a corpus: this only needs to
# keep the most common function words from drowning out the content words
# that make one paragraph more central than another.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on for with as at by from
    is are was were be been being it its this that these those i you he
    she we they them his her their our your not no so do does did has
    have had will would can could should may might must about into over
    after before up down out than just also there
    """.split()
)


def _words(text):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def score_segments(segments):
    """Score each segment by the TF-IDF density of its non-stopword terms.

    For each word occurrence, its weight is ``log((N + 1) / (df + 1)) + 1``
    (smoothed inverse document frequency, where a "document" is a
    segment). A segment's score is the mean of those weights over its
    words, so a short paragraph packed with rare terms scores as high as
    a long one saying the same thing at greater length.

    Code segments always score 0.0: keyword density is a prose measure,
    and scoring code by its word-like tokens would rank a block by
    identifier choice rather than by whether it matters. That does not
    exclude code from selection, it just means code has to be pulled in
    by `head-tail` or kept by budget headroom rather than by score.
    """
    n = len(segments)
    if n == 0:
        return []

    doc_words = [[] if s.is_code else _words(s.text) for s in segments]

    df = {}
    for words in doc_words:
        for w in set(words):
            df[w] = df.get(w, 0) + 1

    scores = []
    for words in doc_words:
        if not words:
            scores.append(0.0)
            continue
        idf_sum = sum(math.log((n + 1) / (df[w] + 1)) + 1.0 for w in words)
        scores.append(idf_sum / len(words))
    return scores


def select_by_score(segments, budget):
    """Pick the highest-scoring segments that fit in ``budget`` tokens.

    Segments are considered in score order (ties keep document order,
    since Python's sort is stable) and taken greedily while they still
    fit; a low scorer that would blow the budget is skipped rather than
    truncated, so every segment in the result is intact. The returned
    list is restored to original document order, which is what callers
    need to render elision markers and head/tail context correctly.
    """
    if budget <= 0 or not segments:
        return []

    scores = score_segments(segments)
    order = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)

    selected = set()
    used = 0
    for i in order:
        cost = estimate_tokens(segments[i].text)
        if used + cost > budget:
            continue
        selected.add(i)
        used += cost

    return [segments[i] for i in sorted(selected)]
