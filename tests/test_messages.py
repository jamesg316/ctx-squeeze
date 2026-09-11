from ctx_squeeze import parse_messages, prune_messages, to_dicts
from ctx_squeeze.messages import Message


def _openai(role, content, **extra):
    return {"role": role, "content": content, **extra}


def test_parse_plain_openai_messages():
    history = [_openai("system", "be careful"), _openai("user", "hello")]
    parsed = parse_messages(history)
    assert [m.role for m in parsed] == ["system", "user"]
    assert [m.text for m in parsed] == ["be careful", "hello"]


def test_parse_openai_tool_call_and_result():
    history = [
        _openai("assistant", None, tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ]),
        _openai("tool", "file contents here", tool_call_id="call_1"),
    ]
    parsed = parse_messages(history)
    assert parsed[0].tool_call_ids == ("call_1",)
    assert parsed[1].tool_result_ids == ("call_1",)


def test_parse_anthropic_content_blocks():
    history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"}],
        },
    ]
    parsed = parse_messages(history)
    assert parsed[0].text == "let me check"
    assert parsed[0].tool_call_ids == ("toolu_1",)
    assert parsed[1].tool_result_ids == ("toolu_1",)
    assert "file contents" in parsed[1].text


def test_parse_preserves_raw_for_round_trip():
    original = _openai("user", "hello there")
    parsed = parse_messages([original])
    assert to_dicts(parsed) == [original]


def test_system_messages_always_survive():
    history = [_openai("system", "system prompt")] + [
        _openai("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(20)
    ]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=1, recent_turns=0)
    assert result.messages[0].role == "system"
    assert result.messages[0].text == "system prompt"


def test_recent_turns_survive_whole_even_over_budget():
    history = [_openai("user", "old question")] + [
        _openai("user", "x" * 400),
        _openai("assistant", "y" * 400),
    ]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=1, recent_turns=1)
    kept_text = [m.text for m in result.messages if m.role != "system" or "elided" not in m.text]
    assert "x" * 400 in kept_text
    assert "y" * 400 in kept_text


def test_older_messages_dropped_and_marked():
    history = [_openai("user", f"filler message number {i} with some words") for i in range(10)]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=10, recent_turns=1)
    assert result.messages_out < result.messages_in
    assert any("elided" in m.text for m in result.messages)


def test_no_marker_omits_elision_text():
    history = [_openai("user", f"filler message number {i} with some words") for i in range(10)]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=10, recent_turns=1, use_marker=False)
    assert all("elided" not in m.text for m in result.messages)


def test_tool_result_never_separated_from_its_call():
    history = (
        [_openai("user", f"padding turn {i} with extra words to cost tokens") for i in range(6)]
        + [
            _openai("assistant", None, tool_calls=[
                {"id": "call_9", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ]),
            _openai("tool", "tool output", tool_call_id="call_9"),
        ]
        + [_openai("user", "final question")]
    )
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=1000, recent_turns=1)
    roles_kept = [(m.role, m.tool_call_ids, m.tool_result_ids) for m in result.messages]
    has_call = any(ids for _, ids, _ in roles_kept)
    has_result = any(ids for _, _, ids in roles_kept)
    assert has_call == has_result
    assert has_call is True


def test_pinned_tool_results_reports_ids_kept_by_linking():
    history = (
        [_openai("user", f"padding turn {i} with extra words to cost tokens") for i in range(6)]
        + [
            _openai("assistant", None, tool_calls=[
                {"id": "call_9", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ]),
            _openai("tool", "tool output", tool_call_id="call_9"),
        ]
        + [_openai("user", "final question")]
    )
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=1000, recent_turns=1)
    assert "call_9" in result.pinned_tool_results


def test_empty_history_returns_empty_result():
    result = prune_messages([], budget=100)
    assert result.messages == []
    assert result.messages_in == 0
    assert result.messages_out == 0
    assert result.pinned_tool_results == frozenset()


def test_whole_transcript_kept_when_it_fits_budget():
    history = [_openai("system", "prompt"), _openai("user", "hi"), _openai("assistant", "hello")]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=1000, recent_turns=2)
    assert result.messages_out == 3
    assert result.final_tokens == result.original_tokens


def test_final_tokens_never_exceed_budget_when_only_optional_messages_are_dropped():
    history = [_openai("user", f"filler message number {i} with some words") for i in range(30)]
    parsed = parse_messages(history)
    result = prune_messages(parsed, budget=50, recent_turns=1)
    assert result.final_tokens <= 50
    assert result.messages_out < result.messages_in


def test_to_dicts_round_trips_marker_messages():
    marker_msg = Message(role="system", text="[2 earlier messages elided]", raw={"role": "system", "content": "[2 earlier messages elided]"})
    assert to_dicts([marker_msg]) == [{"role": "system", "content": "[2 earlier messages elided]"}]
