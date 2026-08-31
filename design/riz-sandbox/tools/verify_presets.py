"""Cross-check every sandbox preset: artboard JS vs the independent Python port.

    node tools/dump_presets.mjs > tools/presets.json
    python tools/verify_presets.py

The artboard runs its own copy of the machine; `riz_machine.py` is a separate
port written from the same Pine source. If they disagree on any zone's final
state, one of them drifted — that is exactly what this catches. Run it after
touching either the artboard's `machine()` or a preset's bars.

Exit code 0 = every preset agrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from riz_machine import Bar, is_blue, run  # noqa: E402

HERE = Path(__file__).resolve().parent
PRESETS = HERE / "presets.json"


def key(z: dict) -> tuple:
    """The comparable end state of one zone."""
    return (round(z["t"], 4), round(z["b"], 4), z["nx"], z["ti"],
            bool(z["nn"]), bool(z["ss"]), bool(z["dead"]))


def main() -> int:
    if not PRESETS.exists():
        print(f"missing {PRESETS.name} — run: node tools/dump_presets.mjs > tools/presets.json")
        return 2

    data = json.loads(PRESETS.read_text(encoding="utf-8"))
    failures = 0

    for name, p in data.items():
        bars = [Bar(b[0], b[1], b[2], b[3], b[4] if len(b) > 4 else "") for b in p["bars"]]
        zones, _ = run(bars)

        mine = sorted(key(z) for z in zones)
        theirs = sorted(key(z) for z in p["zones"])

        blue_mine = sorted((round(z["t"], 4), round(z["b"], 4)) for z in zones if is_blue(z))
        blue_theirs = sorted((round(z["t"], 4), round(z["b"], 4))
                             for z in p["zones"] if z.get("blue"))

        ok = mine == theirs and blue_mine == blue_theirs
        blue_txt = ", ".join(f"{t}/{b}" for t, b in blue_mine) or "none"
        print(f"{'ok ' if ok else 'DIFF'}  {name:<8} {len(bars):>2} bars, "
              f"{len(zones)} zones, blue: {blue_txt}")

        if not ok:
            failures += 1
            for row in sorted(set(mine) ^ set(theirs)):
                side = "python" if row in mine else "artboard"
                print(f"        only in {side}: {row}")

    print()
    if failures:
        print(f"{failures} preset(s) disagree — reconcile before publishing.")
        return 1
    print(f"all {len(data)} presets agree across both implementations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
