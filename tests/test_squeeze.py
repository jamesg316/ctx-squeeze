from ctx_squeeze import estimate_tokens, squeeze


def test_empty_text_returns_empty_result():
    result = squeeze("", budget=100)
    assert result.text == ""
    assert result.original_tokens == 0
    assert result.final_tokens == 0
    assert result.segments_in == 0
    assert result.segments_out == 0


def test_score_strategy_never_exceeds_budget():
    text = "\n\n".join(f"Paragraph {i} about something entirely different each time." for i in range(20))
    result = squeeze(text, budget=40, strategy="score")
    assert result.final_tokens <= 40
    assert result.segments_out < result.segments_in


def test_score_strategy_marks_elided_segments():
    # The second paragraph is far too expensive to fit next to the first no
    # matter which one scores higher, so the outcome (keep the cheap one,
    # elide the costly one) is deterministic regardless of score order.
    keep_para = "keep me around please friend"
    drop_para = "x" * 800
    text = "\n\n".join([keep_para, drop_para])
    budget = estimate_tokens(keep_para) + 20
    result = squeeze(text, budget=budget, strategy="score")
    assert result.segments_out == 1
    assert keep_para in result.text
    assert "segments elided" in result.text


def test_dedupe_then_score_drops_repeats_and_notes_it():
    repeat = "The build failed after the runner image was bumped to a new version."
    text = "\n\n".join([repeat, "Something unrelated about a different topic entirely.", repeat, repeat])
    result = squeeze(text, budget=1000, strategy="dedupe,score")
    assert any("dedupe dropped" in note for note in result.notes)
    assert result.segments_out < result.segments_in


def test_head_tail_keeps_first_and_last_segments():
    paragraphs = [f"Paragraph number {i} with some filler words in it for padding." for i in range(10)]
    text = "\n\n".join(paragraphs)
    cost = estimate_tokens(paragraphs[0])
    # Room for exactly one paragraph on each side plus enough slack that the
    # elision marker and separators can never push the render over budget.
    budget = cost * 2 + 20
    result = squeeze(text, budget=budget, strategy="head-tail", head_ratio=0.5)
    assert paragraphs[0] in result.text
    assert paragraphs[-1] in result.text
    assert result.final_tokens <= budget


def test_no_marker_omits_elision_text():
    text = "\n\n".join(f"Paragraph {i} about something entirely different each time." for i in range(20))
    result = squeeze(text, budget=40, strategy="score", use_marker=False)
    assert "segments elided" not in result.text


def test_unknown_stage_raises():
    try:
        squeeze("some text here", budget=10, strategy="not-a-real-stage")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy stage")


def test_result_fits_budget_even_when_marker_overhead_pushes_it_over():
    # A budget so tight that scored segment costs alone fit, but the
    # rendered "[N segments elided]" marker text pushes the total over;
    # squeeze must still return something within budget.
    text = "\n\n".join(f"Paragraph {i} with some short filler text." for i in range(30))
    result = squeeze(text, budget=5, strategy="score")
    assert result.final_tokens <= 5


def test_whole_document_kept_when_it_already_fits():
    text = "One short paragraph."
    result = squeeze(text, budget=1000, strategy="score")
    assert result.text == text
    assert result.final_tokens == estimate_tokens(text)
