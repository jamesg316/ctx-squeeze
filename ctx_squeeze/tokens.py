"""Character-class token estimator.

No tokenizer, no vocabulary file, no network call: just a per-character
classification whose per-class rates were picked to track a BPE tokenizer
within about 10% on English prose. See the README for the rationale.
"""

TOKENS_PER_NEWLINE = 0.5
TOKENS_PER_SYMBOL = 0.6
CHARS_PER_TOKEN_WORD = 4.0
CHARS_PER_TOKEN_DIGIT = 3.0

# CJK scripts pack much more meaning per character than Latin script, so
# each character is charged a full token rather than divided into a run.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana, katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7A3),  # hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``.

    Whitespace other than newlines is free (it separates words a tokenizer
    would merge into one token anyway). Everything else is charged per
    character-run: letters at four chars/token, digits at three chars/token,
    each CJK character as one token, each newline as half a token, and any
    other symbol as 0.6 of a token.
    """
    if not text:
        return 0

    total = 0.0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]

        if ch == "\n":
            total += TOKENS_PER_NEWLINE
            i += 1
        elif ch.isspace():
            i += 1
        elif _is_cjk(ch):
            total += 1.0
            i += 1
        elif ch.isdigit():
            j = i + 1
            while j < length and text[j].isdigit():
                j += 1
            total += (j - i) / CHARS_PER_TOKEN_DIGIT
            i = j
        elif ch.isalpha():
            j = i + 1
            while j < length and text[j].isalpha() and not _is_cjk(text[j]):
                j += 1
            total += (j - i) / CHARS_PER_TOKEN_WORD
            i = j
        else:
            total += TOKENS_PER_SYMBOL
            i += 1

    # Round-to-even would send short but real content (a single short word,
    # one piece of punctuation) to zero tokens, which is never actually free.
    if text.strip() and total < 1.0:
        return 1
    return round(total)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate ``text`` to at most ``max_tokens`` estimated tokens.

    This is a blunt fallback: it cuts wherever the budget runs out, with no
    regard for word or code-block boundaries. Callers that care about that
    (the segment-based strategies) should prefer dropping whole segments.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low]
