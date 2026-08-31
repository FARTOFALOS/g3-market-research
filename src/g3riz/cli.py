from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
