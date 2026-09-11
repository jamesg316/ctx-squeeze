"""Message-level pruning for chat transcripts.

A transcript is not an undifferentiated block of text the way a document
is: messages have roles, a tool call is meaningless without the result
that answers it (and vice versa), and the turns closest to "now" matter
more than anything earlier. Pruning at the message level keeps that
structure intact instead of slicing through it the way segment-based
squeezing would.

Both the OpenAI message shape (``content`` is a string, tool calls live in
a ``tool_calls`` list, tool results are ``role: "tool"`` with a
``tool_call_id``) and the Anthropic shape (``content`` is a list of typed
blocks, tool calls are ``tool_use`` blocks, tool results are
``tool_result`` blocks usually inside a user message) are accepted, since
an agent transcript assembled from either API looks the same once it is
parsed into :class:`Message`.
"""

from dataclasses import dataclass

from ctx_squeeze.tokens import estimate_tokens


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    tool_call_ids: tuple = ()
    tool_result_ids: tuple = ()
    raw: object = None


@dataclass(frozen=True)
class PruneResult:
    messages: list
    original_tokens: int
    final_tokens: int
    messages_in: int
    messages_out: int
    pinned_tool_results: frozenset


def _text_from_content(content):
    parts = []
    if isinstance(content, str):
        if content:
            parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                if block.get("text"):
                    parts.append(block["text"])
            elif block_type == "tool_result":
                nested = block.get("content")
                if isinstance(nested, str):
                    if nested:
                        parts.append(nested)
                elif isinstance(nested, list):
                    for sub in nested:
                        if isinstance(sub, dict) and sub.get("type") == "text" and sub.get("text"):
                            parts.append(sub["text"])
    return parts


def parse_messages(history):
    """Parse a list of message dicts (OpenAI or Anthropic shape) into :class:`Message`.

    The original dict is kept verbatim on ``raw`` so ``to_dicts`` can hand
    it back unchanged; only the fields ``prune_messages`` actually needs to
    reason about (role, text, and the ids that link a tool call to its
    result) are pulled out here.
    """
    messages = []
    for raw in history:
        role = raw.get("role", "")
        text_parts = _text_from_content(raw.get("content"))
        tool_call_ids = []
        tool_result_ids = []

        for block in raw.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") is not None:
                tool_call_ids.append(block["id"])
            elif block.get("type") == "tool_result" and block.get("tool_use_id") is not None:
                tool_result_ids.append(block["tool_use_id"])

        for call in raw.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("id") is None:
                continue
            tool_call_ids.append(call["id"])
            arguments = (call.get("function") or {}).get("arguments")
            if arguments:
                text_parts.append(str(arguments))

        if raw.get("tool_call_id") is not None:
            tool_result_ids.append(raw["tool_call_id"])

        messages.append(
            Message(
                role=role,
                text="\n".join(text_parts),
                tool_call_ids=tuple(tool_call_ids),
                tool_result_ids=tuple(tool_result_ids),
                raw=raw,
            )
        )
    return messages


def _marker(count):
    noun = "message" if count == 1 else "messages"
    text = f"[{count} earlier {noun} elided]"
    return Message(role="system", text=text, raw={"role": "system", "content": text})


def to_dicts(messages):
    """Render a list of :class:`Message` back to plain dicts, ready for ``json.dumps``."""
    return [m.raw for m in messages]


def prune_messages(messages, budget, recent_turns=2, use_marker=True):
    """Prune ``messages`` to fit ``budget`` estimated tokens, structurally.

    Three rules take priority over the budget and are never violated:

    - every system message survives.
    - the last ``recent_turns`` user turns survive whole, along with
      everything after them (the assistant replies and tool traffic that
      answers the current turn).
    - a tool call and the tool result that answers it are never separated:
      keeping one pulls the other in regardless of where it sits.

    Whatever budget is left after those mandatory messages is spent on
    older messages, newest first, each pulled in together with whichever
    tool call or result it is paired with. Runs of consecutive dropped
    messages are replaced by a single ``[N earlier messages elided]``
    system message so the shape of what was removed stays visible. That
    marker text costs tokens the budget pass did not account for, so if
    the rendered result still comes in over budget, the most recently
    added optional messages are dropped first (oldest content survives
    least, since it was the weakest candidate to begin with) until it
    fits or nothing optional is left to give up.
    """
    messages = list(messages)
    n = len(messages)
    original_tokens = sum(estimate_tokens(m.text) for m in messages)

    if n == 0:
        return PruneResult(
            messages=[], original_tokens=0, final_tokens=0,
            messages_in=0, messages_out=0, pinned_tool_results=frozenset(),
        )

    mandatory = {i for i, m in enumerate(messages) if m.role == "system"}

    recent_start = n
    user_turns_seen = 0
    for i in range(n - 1, -1, -1):
        if user_turns_seen >= recent_turns:
            break
        recent_start = i
        if messages[i].role == "user":
            user_turns_seen += 1
    mandatory.update(range(recent_start, n))

    call_index = {}
    result_index = {}
    for i, m in enumerate(messages):
        for cid in m.tool_call_ids:
            call_index.setdefault(cid, i)
        for rid in m.tool_result_ids:
            result_index.setdefault(rid, i)

    def partners_of(i):
        m = messages[i]
        found = set()
        for cid in m.tool_call_ids:
            j = result_index.get(cid)
            if j is not None:
                found.add(j)
        for rid in m.tool_result_ids:
            j = call_index.get(rid)
            if j is not None:
                found.add(j)
        return found

    frontier = set(mandatory)
    while frontier:
        newly = set()
        for i in frontier:
            for j in partners_of(i):
                if j not in mandatory:
                    mandatory.add(j)
                    newly.add(j)
        frontier = newly

    included = set(mandatory)
    used = sum(estimate_tokens(messages[i].text) for i in included)
    added_groups = []

    for i in sorted((j for j in range(n) if j not in included), reverse=True):
        if i in included:
            continue
        group = ({i} | partners_of(i)) - included
        if not group:
            continue
        cost = sum(estimate_tokens(messages[j].text) for j in group)
        if used + cost > budget:
            continue
        included |= group
        used += cost
        added_groups.append(group)

    def render(included_set):
        rendered = []
        run = 0
        for i in range(n):
            if i in included_set:
                if run:
                    if use_marker:
                        rendered.append(_marker(run))
                    run = 0
                rendered.append(messages[i])
            else:
                run += 1
        if run and use_marker:
            rendered.append(_marker(run))
        return rendered

    rendered = render(included)
    final_tokens = sum(estimate_tokens(m.text) for m in rendered)

    while final_tokens > budget and added_groups:
        included -= added_groups.pop()
        rendered = render(included)
        final_tokens = sum(estimate_tokens(m.text) for m in rendered)

    pinned_tool_results = frozenset(
        id_
        for i in included
        if i < recent_start
        for id_ in messages[i].tool_call_ids + messages[i].tool_result_ids
    )

    return PruneResult(
        messages=rendered,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        messages_in=n,
        messages_out=len(included),
        pinned_tool_results=pinned_tool_results,
    )
