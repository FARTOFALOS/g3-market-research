"""075: read one scene the way a person reads a chart -- minute by minute, on
closes, with the zone in view and nothing from the future on the line being read.

    python -B work/075/read_film.py <row> [k_from] [k_to]
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def render(row, zone_top, zone_bottom, exit_side, k0=0, k1=40, mark=None):
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    p = pos[row]
    w = max(zone_top - zone_bottom, .25)
    south = exit_side == 'south'
    near, far = (zone_bottom, zone_top) if south else (zone_top, zone_bottom)
    t0 = pd.Timestamp(common.t0_list()[row]).tz_localize('UTC').tz_convert('America/New_York')
    print(f'scene row={row}  T0 {t0:%Y-%m-%d %H:%M} New York  zone [{zone_bottom} .. {zone_top}] '
          f'width {w:g}  price left {exit_side}')
    print(f'{"min":>4} {"time":>6} {"open":>10} {"high":>10} {"low":>10} {"close":>10} '
          f'{"vs zone":>8} {"away_w":>8} {"UDBI":>5}  note')
    prevH = prevL = None
    for k in range(k0, min(k1, common.HOR) + 1):
        j = p[k]
        if j < 0:
            print(f'{k:4d}      -   minute absent from the spine')
            prevH = prevL = None
            continue
        O, H, L, C = float(o[j]), float(h[j]), float(l[j]), float(c[j])
        tt = pd.Timestamp(ts[j]).tz_localize('UTC').tz_convert('America/New_York')
        if south:
            st = 'E' if H < zone_bottom else ('F' if L > zone_top else 'Z')
            dep = (zone_bottom - C) / w
        else:
            st = 'E' if L > zone_top else ('F' if H < zone_bottom else 'Z')
            dep = (C - zone_top) / w
        if prevH is None:
            lt = '-'
        else:
            up, dn = H > prevH, L < prevL
            lt = 'U' if up and not dn else ('D' if dn and not up else ('B' if up and dn else 'I'))
        note = mark.get(k, '') if mark else ''
        print(f'{k:4d} {tt:%H:%M} {O:10.2f} {H:10.2f} {L:10.2f} {C:10.2f} '
              f'{st:>8} {dep:8.2f} {lt:>5}  {note}')
        prevH, prevL = H, L


def main():
    row = int(sys.argv[1])
    d = common.scene_table()
    s = d[d.row == row].iloc[0]
    render(row, float(s.zone_top), float(s.zone_bottom), s.t0_exit_side,
           int(sys.argv[2]) if len(sys.argv) > 2 else 0,
           int(sys.argv[3]) if len(sys.argv) > 3 else 40)


if __name__ == '__main__':
    main()
