"""Paragraph and code-block segmentation.

Splits text into the units that later pipeline stages (dedupe, score,
select) operate on. A segment is never partial: a paragraph is whole, and
a fenced code block is whole even if that means it runs to the end of the
text because its fence was never closed. Line numbers are 1-based and
inclusive, and exist so the cli can report where an elided segment used to
sit.
"""

from dataclasses import dataclass

_FENCE_MARKERS = ("```", "~~~")


@dataclass(frozen=True)
class Segment:
    text: str
    start_line: int
    end_line: int
    is_code: bool


def split_segments(text):
    """Split ``text`` into a list of :class:`Segment`.

    Blank lines separate paragraphs. A line whose stripped form starts with
    a fence marker opens a code segment that runs until a line starting
    with the same marker, or to the end of the text if it is never closed.
    """
    if not text:
        return []

    lines = text.split("\n")
    n = len(lines)
    segments = []
    buf = []
    buf_start = None

    def flush(end_line):
        nonlocal buf, buf_start
        if buf:
            segments.append(
                Segment(text="\n".join(buf), start_line=buf_start, end_line=end_line, is_code=False)
            )
            buf = []
            buf_start = None

    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()
        fence = next((m for m in _FENCE_MARKERS if stripped.startswith(m)), None)

        if fence is not None:
            flush(i)
            code_start = i + 1
            code_lines = [line]
            i += 1
            while i < n:
                code_lines.append(lines[i])
                i += 1
                if lines[i - 1].strip().startswith(fence):
                    break
            segments.append(
                Segment(text="\n".join(code_lines), start_line=code_start, end_line=i, is_code=True)
            )
            continue

        if stripped == "":
            flush(i)
            i += 1
            continue

        if buf_start is None:
            buf_start = i + 1
        buf.append(line)
        i += 1

    flush(n)
    return segments
