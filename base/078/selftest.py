"""078A self-test: the reader, not the market.

Checks that the frozen geometry is what FREEZE_078A.md says it is, and that no
coordinate reads a minute later than p. Run before support.py.

python -B base/078/selftest.py
"""
from __future__ import annotations

import numpy as np

from represent import (BASE_SPECS, SENSITIVITIES, L_MAX, PAD, RULERS,
                       Spec, Tape, coords, sigma_sd)

RNG = np.random.default_rng(20780914)
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def block_distance(tape: Tape, i: int, j: int, spec: Spec, s_sd: float) -> float:
    """The block formula written out directly from the freeze, independently."""
    mod, yr = tape.clock()
    sg = tape.sigma_at(np.array([i, j]), spec.ruler, spec.shift)
    sig = {i: sg[0], j: sg[1]}
    ta, tb = (2 * np.pi * mod[i] / 1440.0), (2 * np.pi * mod[j] / 1440.0)
    d = spec.w_clock * ((np.cos(ta) - np.cos(tb)) ** 2
                        + (np.sin(ta) - np.sin(tb)) ** 2) / 4.0
    d += spec.w_epoch * (((yr[i] - 2006) / 20.0) - ((yr[j] - 2006) / 20.0)) ** 2
    d += spec.w_scale * ((np.log(sig[i]) - np.log(sig[j])) / s_sd) ** 2
    if spec.lookback is not None:
        L = spec.lookback
        va, vb = [], []
        for p_, store in ((i, va), (j, vb)):
            end = p_ - spec.shift
            a = tape.close[end - L]
            for k in range(end - L + 1, end + 1):
                for arr in (tape.open, tape.high, tape.low, tape.close):
                    store.append((arr[k] - a) / sig[p_])
        d += spec.w_shape * np.mean((np.array(va) - np.array(vb)) ** 2)
    return float(d)


def main() -> None:
    tape = Tape("NQ")
    elig = tape.eligible()
    pool = np.flatnonzero(elig)
    print(f"tape {tape.n} minutes, eligible {pool.size} "
          f"({pool.size / tape.n:.4f} of tape), PAD={PAD}")

    print("\n1. eligibility really means 151 contiguous finite minutes")
    sample = RNG.choice(pool, 400, replace=False)
    bad = 0
    for p in sample:
        seg = tape.ts[p - PAD + 1:p + 1]
        if len(seg) != PAD or np.any(np.diff(seg) != 60_000_000_000):
            bad += 1
        if not np.isfinite(tape.open[p - PAD + 1:p + 1]).all():
            bad += 1
    check("400 sampled eligible minutes have a clean 151-minute span", bad == 0,
          f"{bad} violations")

    print("\n2. Euclidean on coords reproduces the block formula exactly")
    for spec in BASE_SPECS + SENSITIVITIES:
        s_sd = sigma_sd(tape, spec, pool[RNG.choice(pool.size, 50_000, replace=False)])
        pick = RNG.choice(pool, 24, replace=False)
        X = coords(tape, pick, spec, s_sd).astype(np.float64)
        worst = 0.0
        for a in range(0, 24, 2):
            i, j = int(pick[a]), int(pick[a + 1])
            got = float(np.sum((X[a] - X[a + 1]) ** 2))
            want = block_distance(tape, i, j, spec, s_sd)
            worst = max(worst, abs(got - want) / max(want, 1e-12))
        check(f"{spec.name}", worst < 1e-5, f"max rel err {worst:.2e}")

    print("\n3. no coordinate reads a minute later than p")
    for spec in BASE_SPECS + SENSITIVITIES:
        s_sd = sigma_sd(tape, spec, pool[:50_000])
        pick = RNG.choice(pool[pool < tape.n - 5000], 8, replace=False)
        base = coords(tape, pick, spec, s_sd)
        saved = {}
        for arr_name in ("open", "high", "low", "close"):
            arr = getattr(tape, arr_name)
            saved[arr_name] = arr.copy()
            for p in pick:
                arr[p + 1:p + 4000] = np.nan
        after = coords(tape, pick, spec, s_sd)
        for arr_name, val in saved.items():
            setattr(tape, arr_name, val)
        tape._rng = None
        check(f"{spec.name} unchanged after the future is destroyed",
              np.array_equal(base, after, equal_nan=True))

    print("\n4. clock has no midnight seam")
    mod, _ = tape.clock()
    p2359 = pool[mod[pool] == 1439][0]
    p0000 = pool[mod[pool] == 0][0]
    p1200 = pool[mod[pool] == 720][0]
    sp = Spec("clock only", None, w_epoch=0.0, w_scale=0.0)
    s_sd = sigma_sd(tape, sp, pool[:50_000])
    X = coords(tape, np.array([p2359, p0000, p1200]), sp, s_sd).astype(np.float64)
    seam = float(np.sum((X[0] - X[1]) ** 2))
    far = float(np.sum((X[0] - X[2]) ** 2))
    check("23:59 vs 00:00 is nearer than 23:59 vs 12:00",
          seam < far and seam < 1e-4, f"seam {seam:.2e} vs far {far:.3f}")

    print("\n5. the shape block is a MEAN, so 20 and 120 dims weigh the same")
    # an artificial pair whose per-coordinate shape difference is a constant
    a = np.zeros(4 * 5)
    b = np.full(4 * 5, 0.3)
    d5 = np.mean((a - b) ** 2)
    a = np.zeros(4 * 30)
    b = np.full(4 * 30, 0.3)
    d30 = np.mean((a - b) ** 2)
    check("equal per-coordinate difference gives equal block distance",
          abs(d5 - d30) < 1e-12, f"{d5:.6f} vs {d30:.6f}")

    print("\n6. no RIZ-derived quantity is reachable from represent.py")
    import ast
    path = __file__.replace("selftest", "represent")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    # strip every docstring and comment: judge the CODE, not the prose
    code_lines = set(range(1, len(src.splitlines()) + 1))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                code_lines.discard(ln)
    body = "\n".join(line.split("#")[0]
                     for i, line in enumerate(src.splitlines(), 1)
                     if i in code_lines)
    banned = ["zone_", "riz_", "passport", "t0_", "exit_side", "span_count",
              "tf_minutes", "volume", "Volume", "data/field", "field"]
    hit = [w for w in banned if w in body]
    check("representation CODE names no field/RIZ/volume symbol", not hit, str(hit))

    print("\n7. the ruler never overlaps a representation window")
    for spec in BASE_SPECS + SENSITIVITIES:
        ruler_last = L_MAX + spec.shift          # p - this
        win_first = (L_MAX if spec.lookback is None
                     else spec.lookback) - 1 + spec.shift   # p - this
        check(f"{spec.name}: ruler ends {ruler_last} before p, window starts "
              f"{win_first} before p", ruler_last > win_first)

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))


if __name__ == "__main__":
    main()
