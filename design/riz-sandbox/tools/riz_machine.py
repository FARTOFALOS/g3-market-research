"""Reference port of the RIZ BLUE v1.0 `f_machine()` state machine.

Ported line by line from `reference/pine/RIZ_BLUE_v1.0.pine`. The Pine
machine is code-identical to the one inside `RIZ v9.7.pine` — verified by
stripping comments from both `f_machine()` bodies: 167 lines each, same
SHA-256. Only the comments differ, so there is no separate "v1.0 logic".

This module exists so any agent can re-run a scenario without opening the
sandbox, and so sandbox presets can be re-verified after an edit.

Scope: confirmed bars only, which is what the Pine machine itself does. The
developing-bar branch (Pine's export block: is2xPreview, nAp/sAp, the brk*
recomputation) is NOT here — it lives in the sandbox artboard's `dyn()`,
because it only affects display, never machine state.

Usage
-----
    from riz_machine import run, Bar
    zones, log = run([Bar(o, h, l, c) for ... ])

Every zone dict carries: t (top), b (bottom), bull (True=BISI), nn/ss
(north/south alive), ac (activated), cr (bar index of last accepted span),
ti (tier: 0 latent, 1/2 active, 3 breaker), t3 (breaker magnet is south),
nx (accepted span count), born, dead, dead_at.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple


class Bar(NamedTuple):
    o: float
    h: float
    l: float
    c: float
    t: str = ""          # optional label, e.g. "11:47"


ZCAP = 40                # Pine keeps at most `zcap` zones, oldest dropped


def is_blue(z: dict) -> bool:
    """strict2x on confirmed bars: tier 1/2, nx>=2, BOTH boundaries alive."""
    return (not z["dead"]) and z["ti"] in (1, 2) and z["nx"] >= 2 and z["nn"] and z["ss"]


def _mk(t: float, b: float, bull: bool, i: int) -> dict:
    return dict(t=t, b=b, bull=bull, nn=True, ss=True, ac=False, cr=-1,
                ti=0, t3=False, nx=0, born=i, dead=False, dead_at=-1,
                x1_at=-1, x2_at=-1)


def run(bars: Iterable[Bar], zcap: int = ZCAP):
    """Replay the machine over confirmed bars. Returns (zones, log).

    `log` entries are dicts: {i, kind, zone, text} where kind is one of
    born / x1 / x2 / xn / ret / brk / dead.
    """
    bars = list(bars)
    zones: list[dict] = []
    log: list[dict] = []

    def note(i, kind, z, text):
        log.append({"i": i, "kind": kind, "zone": z, "text": text})

    for i, bar in enumerate(bars):
        o, h, l, c = bar.o, bar.h, bar.l, bar.c
        bmin, bmax = min(o, c), max(o, c)

        # ── existing zones, newest first (Pine iterates the array backwards)
        for k in range(len(zones) - 1, -1, -1):
            z = zones[k]
            if z["dead"]:
                continue
            zt, zb = z["t"], z["b"]

            if z["ti"] == 3:
                # breaker: deleted when price returns to the far side
                if (h >= zb) if z["t3"] else (l <= zt):
                    z["dead"], z["dead_at"] = True, i
                    note(i, "dead", z, "breaker worked off, zone deleted")
                continue

            # SPAN = the candle BODY traded the zone's whole range.
            span = bmin < zb and bmax > zt
            if span and (not z["ac"] or i >= z["cr"] + 2):
                if not z["ac"]:
                    z["ac"] = True
                    z["ti"] = 1 if (z["nn"] and z["ss"]) else 2
                    z["nx"], z["x1_at"] = 1, i
                    note(i, "x1", z, f"span #1 activation, nx=1, tier={z['ti']}")
                else:
                    z["nx"] += 1
                    if z["nx"] == 2:
                        z["x2_at"] = i
                    kind = "x2" if z["nx"] == 2 else "xn"
                    note(i, kind, z, f"span #{z['nx']}")
                z["cr"] = i
            else:
                n_alive, s_alive = z["nn"], z["ss"]
                sbk = (not span) and s_alive and not n_alive and o > zb and c < zb
                nbk = (not span) and n_alive and not s_alive and o < zt and c > zt
                sbk2 = ((not span) and n_alive and s_alive
                        and o > zb and c < zb and h >= zt and c <= zt)
                nbk2 = ((not span) and n_alive and s_alive
                        and o < zt and c > zt and l <= zb and c >= zb)
                if sbk or nbk or sbk2 or nbk2:
                    z["nn"] = z["ss"] = False
                    z["ti"] = 3
                    z["t3"] = bool(sbk or sbk2)
                    z["ac"] = True
                    z["cr"] = i
                    branch = "sbk" if sbk else "nbk" if nbk else "sbk2" if sbk2 else "nbk2"
                    magnet = "south" if z["t3"] else "north"
                    note(i, "brk", z, f"BREAKER via {branch}, magnet {magnet}")
                else:
                    # a wick whose RANGE covers a boundary retires that side
                    if l <= zt <= h and z["nn"]:
                        z["nn"] = False
                        note(i, "ret", z, "north retired by wick")
                    if l <= zb <= h and z["ss"]:
                        z["ss"] = False
                        note(i, "ret", z, "south retired by wick")
                    if not z["nn"] and not z["ss"]:
                        z["dead"], z["dead_at"] = True, i
                        note(i, "dead", z, "both sides retired, zone deleted")

        # ── births, evaluated AFTER the update loop, on the same bar
        if i >= 2:
            c1, c2, c3 = bars[i - 2], bars[i - 1], bars[i]

            # BISI: gap low[C3] > high[C1], displacement gated on C2's BODY
            if (c3.l > c1.h
                    and max(c2.o, c2.c) > c1.h
                    and min(c2.o, c2.c) < c3.l):
                bt1 = max(c1.o, c1.c)
                bt2 = max(c2.o, c2.c)
                c2bb = min(c2.o, c2.c)
                c3bb = min(c3.o, c3.c)
                nb = bt1 if c2bb > bt1 else c1.h
                nt = c3bb if c3bb > bt2 else c3.l
                if nt > nb:
                    z = _mk(nt, nb, True, i)
                    zones.append(z)
                    note(i, "born", z, f"BISI born latent at C3 close — {nt} / {nb}")

            # SIBI: gap high[C3] < low[C1], same body gate mirrored
            if (c3.h < c1.l
                    and min(c2.o, c2.c) < c1.l
                    and max(c2.o, c2.c) > c3.h):
                bb1 = min(c1.o, c1.c)
                bb2 = min(c2.o, c2.c)
                c2bt = max(c2.o, c2.c)
                c3bt = max(c3.o, c3.c)
                nt = bb1 if c2bt < bb1 else c1.l
                nb = c3bt if c3bt < bb2 else c3.h
                if nt > nb:
                    z = _mk(nt, nb, False, i)
                    zones.append(z)
                    note(i, "born", z, f"SIBI born latent at C3 close — {nt} / {nb}")

        while len(zones) > zcap:
            zones.pop(0)

    return zones, log


def describe(bars: Iterable[Bar], zones, log) -> str:
    """Human-readable event ladder, one line per machine event."""
    bars = list(bars)
    out = []
    for e in log:
        z = e["zone"]
        label = bars[e["i"]].t or f"bar {e['i']}"
        out.append(f"{label:>8}  {z['t']:.2f}/{z['b']:.2f}  {e['text']}")
    return "\n".join(out)


if __name__ == "__main__":  # smoke test: the verified NQ 11m blue 2X
    real = [
        Bar(29680.75, 29703.25, 29625, 29650.25, "09:46"),
        Bar(29650.75, 29706, 29566, 29572.5, "09:57"),
        Bar(29573.25, 29617.25, 29505, 29596, "10:08"),
        Bar(29596.75, 29675.75, 29596.25, 29653.5, "10:19"),
        Bar(29653.5, 29713, 29632, 29700.75, "10:30"),
        Bar(29700.25, 29769.75, 29686.5, 29754.25, "10:41"),
        Bar(29754.25, 29811.5, 29753.25, 29790, "10:52"),
        Bar(29789.5, 29806.25, 29735.25, 29758.5, "11:03"),
        Bar(29757.75, 29775, 29738, 29768.5, "11:14"),
        Bar(29768.5, 29791.25, 29758.75, 29762.25, "11:25"),
        Bar(29762.5, 29769, 29660.5, 29662.75, "11:36"),
        Bar(29662.75, 29672.25, 29547.5, 29557.25, "11:47"),
        Bar(29557.25, 29579.75, 29509.75, 29578.75, "11:58"),
    ]
    zones, log = run(real)
    print(describe(real, zones, log))
    blue = [z for z in zones if is_blue(z)]
    assert len(blue) == 1, blue
    assert (blue[0]["t"], blue[0]["b"]) == (29625.0, 29617.25), blue[0]
    print("\nOK: single blue 2X at 29625.00 / 29617.25 — matches the chart.")
