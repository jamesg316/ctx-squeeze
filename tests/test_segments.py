from ctx_squeeze import split_segments


def test_empty_text_has_no_segments():
    assert split_segments("") == []


def test_single_paragraph():
    segs = split_segments("one line\nanother line")
    assert len(segs) == 1
    assert segs[0].text == "one line\nanother line"
    assert segs[0].start_line == 1
    assert segs[0].end_line == 2
    assert segs[0].is_code is False


def test_blank_line_separates_paragraphs():
    segs = split_segments("first\n\nsecond")
    assert [s.text for s in segs] == ["first", "second"]
    assert [(s.start_line, s.end_line) for s in segs] == [(1, 1), (3, 3)]


def test_multiple_blank_lines_do_not_produce_empty_segments():
    segs = split_segments("first\n\n\n\nsecond")
    assert len(segs) == 2
    assert [s.text for s in segs] == ["first", "second"]


def test_fenced_code_block_is_one_segment():
    text = "before\n\n```python\nx = 1\ny = 2\n```\n\nafter"
    segs = split_segments(text)
    assert [s.text for s in segs] == ["before", "```python\nx = 1\ny = 2\n```", "after"]
    assert segs[1].is_code is True
    assert segs[1].start_line == 3
    assert segs[1].end_line == 6


def test_blank_lines_inside_a_code_block_do_not_split_it():
    text = "```\nx = 1\n\ny = 2\n```"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].is_code is True
    assert segs[0].text == text


def test_unterminated_fence_runs_to_end_of_text():
    text = "intro\n\n```\ncode without a close\nstill code"
    segs = split_segments(text)
    assert len(segs) == 2
    assert segs[1].is_code is True
    assert segs[1].text == "```\ncode without a close\nstill code"
    assert segs[1].end_line == 5


def test_tilde_fence_is_recognized():
    text = "~~~\nfenced with tildes\n~~~"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].is_code is True


def test_different_fence_markers_do_not_close_each_other():
    text = "```\nline one\n~~~\nline two\n```"
    segs = split_segments(text)
    assert len(segs) == 1
    assert segs[0].is_code is True
    assert segs[0].text == text


def test_non_code_segments_are_never_marked_as_code():
    segs = split_segments("plain paragraph text")
    assert segs[0].is_code is False
