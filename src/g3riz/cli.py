from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # A Russian answer sheet printed through the console's ANSI codepage comes
    # back as mojibake and is unreadable as a record.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(prog="g3-riz")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="census the frozen field from its cell manifests")
    p.add_argument("--full", action="store_true",
                   help="include the per-timeframe lists; default is the compact census")

    p = sub.add_parser("query", help="basic cold-agent field query")
    p.add_argument("--instrument", required=True, choices=("ES", "NQ", "YM"))
    p.add_argument("--tf", type=int, required=True)
    p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("entry-check", help="does the entry path transfer the object?")
    p.add_argument("mode", choices=("make", "score"))
    p.add_argument("--instrument", default="NQ", choices=("ES", "NQ", "YM"))
    p.add_argument("--tf", type=int, default=54)
    p.add_argument("--seed", help="hex; omit on make to draw a fresh one")
    p.add_argument("--paper", type=Path, help="make: where to write the questions")
    p.add_argument("--key", type=Path, help="make: where to write the key; score: where to read it")
    p.add_argument("--answers", type=Path, help="score: the reader's JSON")

    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.command == "status":
        from .store import read_status

        status = read_status(repo / "data" / "field", full=args.full)
        state_path = repo / "PROJECT_STATE.json"
        if state_path.exists():
            status["project_state"] = json.loads(state_path.read_text(encoding="utf-8"))
        print(json.dumps(status, indent=2))
    elif args.command == "query":
        from .query import Field

        field = Field(repo, args.instrument)
        table = field.passports(tf=args.tf).slice(0, args.limit)
        print(table.to_pandas().to_string(index=False))
    elif args.command == "entry-check":
        from .entry_check import build_paper, paper_markdown, score_anchors
        from .query import Field

        if args.mode == "make":
            paper, key = build_paper(Field(repo, args.instrument), args.tf, args.seed)
            if args.paper:
                args.paper.write_text(paper_markdown(paper), encoding="utf-8", newline="\n")
            if args.key:
                args.key.write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                    encoding="utf-8", newline="\n")
            print(json.dumps({"paper_sha256": paper["paper_sha256"],
                              "items": [i["id"] for i in paper["items"]],
                              "key_written": bool(args.key)}, ensure_ascii=False))
        else:
            key = json.loads(args.key.read_text(encoding="utf-8"))
            answers = json.loads(args.answers.read_text(encoding="utf-8"))
            result = score_anchors(key, answers)
            for pid, spec in key.items():
                if pid.startswith("_"):
                    continue
                result["items"][pid]["axis"] = spec["axis"]
                result["items"][pid]["readings"] = spec["readings"]
                result["items"][pid]["answered_free_text"] = {
                    k: v for k, v in answers.get(pid, {}).items()
                    if k not in spec["anchor"]}
            print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
