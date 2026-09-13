"""Command-line entry point.

Thin wiring only: parse the options table from the README into argparse,
dispatch to ``squeeze`` or the message-pruning pipeline depending on
``--messages``, and print the result. Anything that looks like a decision
(what to drop, how to score) belongs in the library modules, not here.
"""

import argparse
import json
import sys

from ctx_squeeze.messages import parse_messages, prune_messages, to_dicts
from ctx_squeeze.squeeze import squeeze


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit long documents and chat transcripts into an LLM context budget.",
    )
    parser.add_argument("input", help="path to the input file, or - for stdin")
    parser.add_argument("--budget", type=int, required=True, help="target size in estimated tokens")
    parser.add_argument("--strategy", default="score", help="comma-separated pipeline: head-tail, score, dedupe")
    parser.add_argument("--head-ratio", type=float, default=0.5, help="share of the budget spent on the head in head-tail")
    parser.add_argument("--jaccard", type=float, default=0.8, help="similarity at which two segments count as duplicates")
    parser.add_argument("--shingle-size", type=int, default=5, help="words per shingle in the dedupe stage")
    parser.add_argument("--messages", action="store_true", help="treat the input as a JSON chat transcript")
    parser.add_argument("--recent-turns", type=int, default=2, help="user turns kept whole in --messages mode")
    parser.add_argument("--no-marker", action="store_true", help="omit the [N segments elided] markers")
    parser.add_argument("--stats", action="store_true", help="print a token summary to stderr")
    parser.add_argument("--json", action="store_true", help="emit a JSON report instead of plain text")
    parser.add_argument("-o", "--output", metavar="PATH", help="write the result to a file instead of stdout")
    return parser


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_output(path, text):
    if path is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_document(args, text):
    result = squeeze(
        text,
        budget=args.budget,
        strategy=args.strategy,
        head_ratio=args.head_ratio,
        jaccard_threshold=args.jaccard,
        shingle_size=args.shingle_size,
        use_marker=not args.no_marker,
    )

    if args.stats:
        print(
            f"kept {result.segments_out} of {result.segments_in} segments | "
            f"{result.original_tokens} -> {result.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )
        for note in result.notes:
            print(note, file=sys.stderr)

    if args.json:
        return json.dumps(
            {
                "text": result.text,
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "segments_in": result.segments_in,
                "segments_out": result.segments_out,
                "notes": result.notes,
            },
            indent=2,
        )
    return result.text


def _run_messages(args, text):
    history = json.loads(text)
    messages = parse_messages(history)
    result = prune_messages(
        messages,
        budget=args.budget,
        recent_turns=args.recent_turns,
        use_marker=not args.no_marker,
    )

    if args.stats:
        print(
            f"kept {result.messages_out} of {result.messages_in} messages | "
            f"{result.original_tokens} -> {result.final_tokens} tokens (budget {args.budget})",
            file=sys.stderr,
        )

    dicts = to_dicts(result.messages)
    if args.json:
        return json.dumps(
            {
                "messages": dicts,
                "original_tokens": result.original_tokens,
                "final_tokens": result.final_tokens,
                "messages_in": result.messages_in,
                "messages_out": result.messages_out,
                "pinned_tool_results": sorted(result.pinned_tool_results),
            },
            indent=2,
        )
    return json.dumps(dicts, indent=2)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    text = _read_input(args.input)

    if args.messages:
        output = _run_messages(args, text)
    else:
        output = _run_document(args, text)

    _write_output(args.output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
