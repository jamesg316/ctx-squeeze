from ctx_squeeze import estimate_tokens, truncate_to_tokens


def test_empty_string_is_free():
    assert estimate_tokens("") == 0


def test_whitespace_only_is_free():
    assert estimate_tokens("   \t  ") == 0


def test_short_word_still_costs_one_token():
    # 0.25 tokens by the raw rate, but nothing real is free.
    assert estimate_tokens("a") == 1
    assert estimate_tokens("ab") == 1


def test_four_char_word_is_one_token():
    assert estimate_tokens("aaaa") == 1


def test_longer_word_scales_at_four_chars_per_token():
    assert estimate_tokens("aaaaaaaa") == 2


def test_digit_run_at_three_chars_per_token():
    assert estimate_tokens("123") == 1
    assert estimate_tokens("123456") == 2


def test_cjk_is_one_token_per_character():
    assert estimate_tokens("中文") == 2


def test_cjk_adjacent_to_latin_word_does_not_merge():
    # A naive word-boundary regex would swallow the CJK run into the
    # preceding Latin word and undercount it.
    assert estimate_tokens("test中文") == estimate_tokens("test") + estimate_tokens("中文")


def test_newlines_cost_half_a_token_each():
    assert estimate_tokens("\n\n\n\n") == 2


def test_symbols_cost_more_than_free_whitespace():
    assert estimate_tokens("!!!!!") > estimate_tokens("     ")


def test_whitespace_between_words_is_free():
    assert estimate_tokens("word word") == estimate_tokens("wordword")


def test_monotonic_in_text_length():
    short = "The quick brown fox"
    longer = short + " jumps over the lazy dog"
    assert estimate_tokens(longer) > estimate_tokens(short)


def test_truncate_respects_budget():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    for budget in (0, 1, 5, 25, 100):
        truncated = truncate_to_tokens(text, budget)
        assert estimate_tokens(truncated) <= budget


def test_truncate_is_a_prefix():
    text = "The quick brown fox jumps over the lazy dog."
    truncated = truncate_to_tokens(text, 3)
    assert text.startswith(truncated)


def test_truncate_noop_when_already_under_budget():
    text = "short text"
    assert truncate_to_tokens(text, 1000) == text


def test_truncate_zero_budget_is_empty():
    assert truncate_to_tokens("anything at all", 0) == ""


def test_truncate_negative_budget_is_empty():
    assert truncate_to_tokens("anything at all", -5) == ""
