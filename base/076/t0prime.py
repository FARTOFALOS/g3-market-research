"""Executes FROZEN_T0PRIME.md. New object; old T0 is not continued.

Step order is mandatory: build the object, report the honest denominator and the
sigma-sufficiency check, then test whether the outcome is already exhausted by
the simple geometry of the contact minute. No state search before that.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import open_y as M

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
NS = 60_000_000_000
SEARCH = 240          # how far forward T0' is looked for, declared
CAP = 60              # outcome cap, declared
MINDIFF = 10          # sigma sufficiency threshold, declared


def build(inst, tf):
    A, se = M.market(inst)
    ts = A['close_ts_utc_ns']
    rows = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                         columns=['zone_bottom', 'zone_top', 't0_spine_pos',
                                  't0_exit_side']).to_pylist()
    rows.sort(key=lambda r: r['t0_spine_pos'])
    out, tally = [], dict(no_contact=0, bad_prefix=0, bad_window=0, bad_zone=0)
    for r in rows:
        t0 = int(r['t0_spine_pos']); p = t0 + 2
        si = se['lookup'].get(int(A['session_id'][t0]))
        if si is None:
            tally['bad_prefix'] += 1; continue
        an = int(se['session_open_utc_ns'][si])
        k = ((int(ts[t0]) - an) // NS - 1) // tf
        bg = an + k * tf * NS
        st = int(np.searchsorted(ts[:p + 1], bg, side='right'))
        tw = np.asarray(ts[st:p + 1])
        if len(tw) < 3 or int(tw[0]) != bg + NS or np.any(np.diff(tw) != NS):
            tally['bad_prefix'] += 1; continue
        w = r['zone_top'] - r['zone_bottom']
        if w <= 0:
            tally['bad_zone'] += 1; continue
        north = r['t0_exit_side'] == 'north'
        ex = r['zone_top'] if north else r['zone_bottom']
        # ---- find T0': first closed minute after p whose range touches the boundary
        hi = st + SEARCH
        if hi + CAP + 1 >= len(A['close']):
            tally['bad_window'] += 1; continue
        seg = np.asarray(ts[p + 1:hi])
        if len(seg) == 0 or np.any(np.diff(seg) != NS):
            tally['bad_window'] += 1; continue
        sh = np.asarray(A['high'][p + 1:hi], float)
        sl = np.asarray(A['low'][p + 1:hi], float)
        olow = (sl - ex) if north else (ex - sh)
        hit = np.where(olow <= 0)[0]
        if len(hit) == 0:
            tally['no_contact'] += 1; continue
        q = p + 1 + int(hit[0])                      # index of T0'
        # ---- state: everything through T0' inclusive
        tpre = np.asarray(ts[st:q + 1])
        if len(tpre) < 3 or np.any(np.diff(tpre) != NS):
            tally['bad_prefix'] += 1; continue
        cpre = np.asarray(A['close'][st:q + 1], float)
        if not np.isfinite(cpre).all():
            tally['bad_prefix'] += 1; continue
        dif = np.diff(cpre)
        sig = float(np.std(dif)) if len(dif) >= 2 else np.nan
        if not np.isfinite(sig) or sig <= 0:
            tally['bad_prefix'] += 1; continue
        oh = (float(A['high'][q]) - ex) if north else (ex - float(A['low'][q]))
        ol = (float(A['low'][q]) - ex) if north else (ex - float(A['high'][q]))
        oc = (float(A['close'][q]) - ex) if north else (ex - float(A['close'][q]))
        # ---- outcome: race A vs B, strictly after T0'
        ft = np.asarray(ts[q + 1:q + 1 + CAP])
        if len(ft) < CAP or np.any(np.diff(ft) != NS):
            tally['bad_window'] += 1; continue
        fc = np.asarray(A['close'][q + 1:q + 1 + CAP], float)
        if not np.isfinite(fc).all():
            tally['bad_window'] += 1; continue
        ofc = (fc - ex) if north else (ex - fc)
        ia = np.where(ofc <= 0)[0]
        ib = np.where(ofc >= w)[0]
        a = ia[0] if len(ia) else 10 ** 9
        b = ib[0] if len(ib) else 10 ** 9
        outcome = 'C' if a == b == 10 ** 9 else ('A' if a < b else 'B')
        out.append(dict(outcome=outcome, w=w, sigma=sig, ndiff=len(dif),
                        depth_s=-ol / sig, depth_w=-ol / w,
                        close_s=oc / sig, close_w=oc / w,
                        w_over_s=w / sig,
                        wait=q - p, day=int(A['session_id'][q]),
                        date=str(np.datetime64(int(ts[q]), 'ns'))[:10]))
    return out, tally


if __name__ == '__main__':
    DATA = {}
    print('OBJECT T0-PRIME: honest denominator')
    print('%-10s %6s | %6s %6s %6s | %s' % ('set', 'n', 'A', 'B', 'C', 'not built'))
    for inst, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
        d, t = build(inst, tf)
        DATA[(inst, tf)] = d
        o = [x['outcome'] for x in d]
        print('%-10s %6d | %.3f %.3f %.3f | %s'
              % ('%s tf%d' % (inst, tf), len(d),
                 o.count('A') / len(o), o.count('B') / len(o), o.count('C') / len(o), t),
              flush=True)

    print('\nSIGMA SUFFICIENCY (state window = interval start .. T0 prime)')
    for k, d in DATA.items():
        nd = np.array([x['ndiff'] for x in d])
        print('   %-10s increments: p5=%d median=%d ; share with <%d: %.3f'
              % ('%s tf%d' % k, np.percentile(nd, 5), np.median(nd), MINDIFF,
                 float((nd < MINDIFF).mean())))

    print('\nIS THE OUTCOME ALREADY EXHAUSTED BY THE GEOMETRY OF T0-PRIME?')
    print('A-rate (accepted back) split at the NQ tf54 median of each coordinate')
    ref = DATA[('NQ', 54)]
    COORD = ['depth_s', 'depth_w', 'close_s', 'close_w', 'w_over_s', 'sigma', 'wait']
    med = {c: float(np.median([x[c] for x in ref])) for c in COORD}
    print('%-10s %s' % ('coord', ' '.join('%-18s' % ('%s tf%d' % k) for k in DATA)))
    for c in COORD:
        cells = []
        for k, d in DATA.items():
            v = np.array([x[c] for x in d]); y = np.array([x['outcome'] == 'A' for x in d], float)
            hi, lo = v > med[c], v <= med[c]
            cells.append('%.3f/%.3f' % (y[hi].mean(), y[lo].mean())
                         if hi.sum() >= 15 and lo.sum() >= 15 else '   --   ')
        print('%-10s %s   (cut %.3f)' % (c, ' '.join('%-18s' % x for x in cells), med[c]))
    print('\ncell = A-rate above cut / A-rate below cut')
