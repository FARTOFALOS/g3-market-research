from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

from .field import build_cell
from .market import ingest_source
from .query import Field
from .resources import estimated_cell_peak_bytes, live_policy
from .store import consolidate, read_status


def _tfs(value: str) -> list[int]:
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    values = sorted(out)
    if not values or values[0] < 1 or values[-1] > 1440:
        raise argparse.ArgumentTypeError("timeframes must be within 1..1440")
    return values


def _build_one(args: tuple[str, str, str, int, bool]) -> dict:
    market_root, field_root, instrument, tf, force = args
    return build_cell(Path(market_root), Path(field_root), instrument, tf, force=force)


def _print_build(result: dict) -> None:
    print(f"{result['instrument']} TF {result['tf_minutes']:4d}: "
          f"passports={result['passports']} events={result['events']} "
          f"elapsed={result['elapsed_seconds']:.2f}s "
          f"{'REUSED' if result['reused'] else 'BUILT'}", flush=True)


def _run_build_group(work: list[tuple[str, str, str, int, bool]], workers: int) -> None:
    if not work:
        return
    if workers == 1:
        for item in work:
            _print_build(_build_one(item))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, item): item[3] for item in work}
        for future in as_completed(futures):
            _print_build(future.result())


def _run_weighted(work: list[tuple[str, str, str, int, bool]], workers: int,
                  ram_budget: int, repo: Path, disk_reserve: int,
                  measured_tf1_peak: int | None) -> None:
    """Persistent worker pool with a live weighted-RAM admission rule."""
    if not work:
        return
    weight_of = lambda item: estimated_cell_peak_bytes(item[3], measured_tf1_peak)
    pending = sorted(work, key=weight_of, reverse=True)
    running: dict = {}
    running_bytes = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            submitted = False
            if shutil.disk_usage(repo).free < disk_reserve:
                raise RuntimeError("disk reserve reached; completed cells are safe, free space before resume")
            index = 0
            while index < len(pending) and len(running) < workers:
                item = pending[index]
                weight = weight_of(item)
                if not running or running_bytes + weight <= ram_budget:
                    pending.pop(index)
                    future = pool.submit(_build_one, item)
                    running[future] = weight
                    running_bytes += weight
                    submitted = True
                else:
                    index += 1
            if running and (not submitted or len(running) >= workers or not pending):
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    running_bytes -= running.pop(future)
                    _print_build(future.result())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="g3-riz")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="verify and create one canonical market spine")
    p.add_argument("--instrument", required=True, choices=("ES", "NQ", "YM"))
    p.add_argument("--source", type=Path, required=True)

    p = sub.add_parser("build", help="build or resume timeframe cells")
    p.add_argument("--instrument", required=True, choices=("ES", "NQ", "YM"))
    p.add_argument("--tfs", type=_tfs, default=list(range(1, 1441)))
    p.add_argument("--workers", type=int, default=0,
                   help="upper bound; 0 derives a safe cap from live CPU/RAM")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("consolidate", help="atomically assemble all 1440 cells")
    p.add_argument("--instrument", required=True, choices=("ES", "NQ", "YM"))

    sub.add_parser("status", help="derive current state from cell manifests")

    p = sub.add_parser("query", help="basic cold-agent field query")
    p.add_argument("--instrument", required=True, choices=("ES", "NQ", "YM"))
    p.add_argument("--tf", type=int, required=True)
    p.add_argument("--limit", type=int, default=5)

    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    market_root = repo / "data" / "market"
    field_root = repo / "data" / "field"
    if args.command == "ingest":
        result = ingest_source(args.source, market_root / args.instrument, args.instrument)
        print(json.dumps(result, indent=2))
    elif args.command == "build":
        work = [(str(market_root / args.instrument), str(field_root), args.instrument, tf, args.force)
                for tf in args.tfs]
        policy = live_policy(repo)
        workers = policy.auto_worker_cap if args.workers <= 0 else min(args.workers, policy.auto_worker_cap)
        print("scheduler=" + json.dumps({**policy.to_dict(), "workers": workers}), flush=True)
        if workers == 1:
            _run_build_group(work, 1)
        else:
            _run_weighted(work, workers, policy.build_ram_budget_bytes, repo,
                          policy.disk_reserve_bytes, policy.measured_tf1_peak_bytes)
    elif args.command == "consolidate":
        print(json.dumps(consolidate(field_root, args.instrument), indent=2))
    elif args.command == "status":
        print(json.dumps(read_status(field_root), indent=2))
    elif args.command == "query":
        field = Field(repo, args.instrument)
        table = field.passports(tf=args.tf).slice(0, args.limit)
        print(table.to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
