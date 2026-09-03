from ctx_squeeze import Segment, score_segments, select_by_score


def _seg(text, is_code=False, start=1):
    return Segment(text=text, start_line=start, end_line=start, is_code=is_code)


def test_no_segments_scores_empty():
    assert score_segments([]) == []


def test_code_segment_always_scores_zero():
    segs = [_seg("```\nx = rare_identifier_name\n```", is_code=True)]
    assert score_segments(segs) == [0.0]


def test_empty_text_segment_scores_zero():
    segs = [_seg(""), _seg("some ordinary words about nothing in particular")]
    scores = score_segments(segs)
    assert scores[0] == 0.0


def test_segment_with_rare_terms_outscores_generic_one():
    segs = [
        _seg("the runner image was bumped after the retry"),
        _seg("xenolith spectrograph anomaly detected in the runner telemetry"),
    ]
    scores = score_segments(segs)
    assert scores[1] > scores[0]


def test_repeated_shared_terms_score_lower_than_unique_ones():
    # "runner" appears in every segment so it carries no idf weight; a
    # segment made only of shared words should score at or below one that
    # mixes in a term found nowhere else.
    segs = [
        _seg("runner runner runner"),
        _seg("runner runner unique"),
        _seg("runner runner runner"),
    ]
    scores = score_segments(segs)
    assert scores[1] > scores[0]
    assert scores[0] == scores[2]


def test_select_by_score_prefers_higher_scoring_segments():
    segs = [
        _seg("generic filler words about nothing much at all", start=1),
        _seg("xenolith spectrograph anomaly telemetry cascade failure", start=3),
    ]
    # Budget only fits one segment; the distinctive one should win.
    budget = max(len(s.text) // 4 for s in segs)
    selected = select_by_score(segs, budget)
    assert selected == [segs[1]]


def test_select_by_score_preserves_document_order():
    segs = [_seg("alpha bravo charlie", start=1), _seg("delta echo foxtrot", start=3)]
    selected = select_by_score(segs, budget=1000)
    assert selected == segs


def test_select_by_score_zero_budget_selects_nothing():
    segs = [_seg("anything at all", start=1)]
    assert select_by_score(segs, budget=0) == []


def test_select_by_score_empty_segments_selects_nothing():
    assert select_by_score([], budget=100) == []


def test_select_by_score_skips_segment_too_big_for_budget():
    segs = [
        _seg("short", start=1),
        _seg("a much longer segment that costs far more estimated tokens than the budget allows", start=2),
    ]
    selected = select_by_score(segs, budget=3)
    assert selected == [segs[0]]


def test_select_by_score_never_exceeds_budget():
    segs = [_seg(f"segment number {i} with some filler words in it", start=i) for i in range(10)]
    budget = 20
    from ctx_squeeze import estimate_tokens

    selected = select_by_score(segs, budget)
    total = sum(estimate_tokens(s.text) for s in selected)
    assert total <= budget
