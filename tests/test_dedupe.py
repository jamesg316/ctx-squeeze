from ctx_squeeze import jaccard, shingles


def test_empty_text_has_no_shingles():
    assert shingles("") == set()


def test_short_text_yields_one_shingle_covering_everything():
    result = shingles("one two three", size=5)
    assert result == {("one", "two", "three")}


def test_shingle_count_matches_word_count_minus_size_plus_one():
    text = "a b c d e f g"
    result = shingles(text, size=3)
    assert len(result) == 5  # 7 words, size 3 -> 5 windows


def test_shingles_are_case_insensitive():
    assert shingles("Hello World", size=2) == shingles("hello world", size=2)


def test_shingles_ignore_punctuation():
    assert shingles("hello, world!", size=2) == shingles("hello world", size=2)


def test_jaccard_identical_sets_is_one():
    s = shingles("the quick brown fox jumps", size=3)
    assert jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    a = shingles("the quick brown fox", size=2)
    b = shingles("completely unrelated sentence text", size=2)
    assert jaccard(a, b) == 0.0


def test_jaccard_two_empty_sets_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_partial_overlap():
    a = {("a", "b"), ("b", "c"), ("c", "d")}
    b = {("b", "c"), ("c", "d"), ("d", "e")}
    # intersection {(b,c),(c,d)} = 2, union has 4 members
    assert jaccard(a, b) == 0.5


def test_near_duplicate_beats_unrelated_text():
    # A retry log repeated with only its timestamp changed: most of the
    # 5-word windows survive untouched, so similarity should stay high even
    # though the raw text is not identical.
    original = (
        "The nightly job started failing on Tuesday after the runner image "
        "was bumped to a newer minor version and every run now spends "
        "eleven minutes reinstalling dependencies from scratch before the "
        "tests can even begin to execute at 03:14 in the morning cycle"
    )
    retried = original.replace("03:14", "03:47")
    unrelated = (
        "Quarterly revenue guidance was revised upward after strong demand "
        "for the new hardware line offset softness in the subscription "
        "segment and management expects margins to hold steady into next year"
    )

    a = shingles(original, size=5)
    b = shingles(retried, size=5)
    c = shingles(unrelated, size=5)

    assert jaccard(a, b) > 0.5
    assert jaccard(a, b) > jaccard(a, c)
