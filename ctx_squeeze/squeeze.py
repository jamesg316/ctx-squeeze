"""The document pipeline: dedupe, score, and head-tail composed under one budget.

Each stage below operates on the same segment list and only narrows it
further; nothing is ever put back once dropped. A stage decides which
segments survive, but none of them know about the final rendered text, so
the actual budget check happens once, at the end, against the real output
(elision markers included) rather than against the sum of segment costs
that stage math used internally.
"""

from dataclasses import dataclass

from ctx_squeeze.dedupe import jaccard, shingles
from ctx_squeeze.score import select_by_score
from ctx_squeeze.segments import split_segments
from ctx_squeeze.tokens import estimate_tokens, truncate_to_tokens


@dataclass(frozen=True)
class SqueezeResult:
    text: str
    original_tokens: int
    final_tokens: int
    segments_in: int
    segments_out: int
    notes: list


def _dedupe(segments, jaccard_threshold, shingle_size, notes):
    """Drop segments whose shingles are near-identical to one already kept.

    Comparison is against everything kept so far, not just the previous
    segment, since a repeated traceback can reappear several segments later
    with other output in between. Code segments are exempt: two blocks that
    both happen to read a config file back look near-identical by shingle
    overlap even when the values they print differ, and dropping code on
    that basis risks losing the one retry that actually changed something.
    """
    kept = []
    kept_shingles = []
    dropped = 0
    for seg in segments:
        if seg.is_code:
            kept.append(seg)
            kept_shingles.append(None)
            continue
        sh = shingles(seg.text, size=shingle_size)
        if any(other is not None and jaccard(sh, other) >= jaccard_threshold for other in kept_shingles):
            dropped += 1
            continue
        kept.append(seg)
        kept_shingles.append(sh)

    if dropped:
        notes.append(f"dedupe dropped {dropped} near-duplicate segment(s)")
    return kept


def _head_tail(segments, budget, head_ratio):
    """Keep a prefix and a suffix of ``segments`` sized by ``head_ratio``.

    The two ends are filled independently against their own sub-budgets, so
    a small head_ratio can still leave room for a generous tail even though
    neither end knows what the other kept. They cannot overlap: the tail is
    only ever filled from the segments the head did not already take.
    """
    if budget <= 0 or not segments:
        return []

    head_budget = int(budget * head_ratio)
    tail_budget = budget - head_budget

    head = []
    used = 0
    for seg in segments:
        cost = estimate_tokens(seg.text)
        if used + cost > head_budget:
            break
        head.append(seg)
        used += cost

    remaining = segments[len(head):]
    tail = []
    used = 0
    for seg in reversed(remaining):
        cost = estimate_tokens(seg.text)
        if used + cost > tail_budget:
            break
        tail.append(seg)
        used += cost
    tail.reverse()

    return head + tail


_STAGES = {"dedupe": _dedupe, "score": select_by_score, "head-tail": _head_tail}

# Exposed so the cli can validate --strategy before doing any work, instead
# of letting an invalid stage name surface as a ValueError mid-pipeline.
STRATEGY_STAGES = tuple(_STAGES)


def _render(all_segments, kept, use_marker):
    kept_ids = {id(seg) for seg in kept}
    parts = []
    elided_run = 0

    def flush_run():
        if elided_run and use_marker:
            parts.append(f"[{elided_run} segments elided]")

    for seg in all_segments:
        if id(seg) in kept_ids:
            flush_run()
            elided_run = 0
            parts.append(seg.text)
        else:
            elided_run += 1
    flush_run()

    return "\n\n".join(parts)


def squeeze(text, budget, strategy="score", head_ratio=0.5, jaccard_threshold=0.8, shingle_size=5, use_marker=True):
    """Compact ``text`` to fit within ``budget`` estimated tokens.

    ``strategy`` is a comma-separated pipeline of stage names applied left
    to right, each narrowing the segment list the next stage sees:

    - ``dedupe`` drops segments whose shingles are near-duplicates of one
      already kept (see ``--jaccard`` / ``--shingle-size``).
    - ``score`` keeps the highest TF-IDF segments that fit in ``budget``.
    - ``head-tail`` keeps a prefix and a suffix split by ``head_ratio``.

    Segments dropped along the way are replaced by an
    ``[N segments elided]`` marker in the output so the shape of what was
    removed stays visible. The marker text itself costs tokens that stage
    budgets never accounted for, so the result is hard-truncated as a last
    resort if it still comes in over budget after rendering.
    """
    original_tokens = estimate_tokens(text)
    all_segments = split_segments(text)
    notes = []

    if not all_segments:
        return SqueezeResult(
            text="", original_tokens=original_tokens, final_tokens=0,
            segments_in=0, segments_out=0, notes=notes,
        )

    stages = [s.strip() for s in strategy.split(",") if s.strip()] or ["score"]

    segments = all_segments
    for stage in stages:
        try:
            fn = _STAGES[stage]
        except KeyError:
            raise ValueError(f"unknown strategy stage: {stage!r}") from None
        if stage == "dedupe":
            segments = fn(segments, jaccard_threshold, shingle_size, notes)
        elif stage == "score":
            segments = fn(segments, budget)
        else:
            segments = fn(segments, budget, head_ratio)

    result_text = _render(all_segments, segments, use_marker)
    final_tokens = estimate_tokens(result_text)

    if final_tokens > budget:
        result_text = truncate_to_tokens(result_text, budget)
        final_tokens = estimate_tokens(result_text)
        notes.append("hard-truncated to fit budget after segment selection")

    return SqueezeResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        segments_in=len(all_segments),
        segments_out=len(segments),
        notes=notes,
    )
