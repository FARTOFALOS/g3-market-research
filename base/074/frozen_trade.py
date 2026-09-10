"""074: the frozen executable construction, exactly as specified, no fitting.

Signal   the first Z minute after T0 that did not reach the far boundary and
         closed past the middle of the zone toward it. Known at that close.
Entry    the open of the next minute; north T0 exit -> short, south -> long.
         No next open, a gap in the grid, or a change of session -> censored.
Stop     one tick beyond the exit boundary of the original side.
Target 1 the far boundary.
Target 2 one zone width beyond the far boundary, fixed in advance.
Order    a minute holding both stop and target is ambiguous; neither side wins.
Timeout  unresolved by +50 or by the end of the continuous session: out at the
         last available close.
Money    NQ point $20 (tick 0.25 = $5); $15 per round turn ($5 commission and
         0.5 point of total slippage), the assumption of S-09.

Nothing here is chosen after seeing PnL. Run: python -B base/074/frozen_trade.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branches import minute_states, validate                # noqa: E402
from first_difference import scenes                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'work/074'
OUT = Path(__file__).resolve().parent
TICK, POINT, COST = 0.25, 20.0, 15.0


def session_window(t0):
    """session id on the same T0+k grid the price window uses"""
    path = CACHE / 'NQ_session51.npy'
    if not path.exists():
        m = ROOT / 'data/market/NQ'
        ts = np.load(m / 'close_ts_utc_ns.npy', mmap_mode='r')
        sid = np.load(m / 'session_id.npy', mmap_mode='r')
        want = t0[:, None] + np.arange(51)[None, :] * 60_000_000_000
        pos = np.searchsorted(ts, want)
        safe = np.minimum(pos, len(ts) - 1)
        hit = (pos < len(ts)) & (ts[safe] == want)
        np.save(path, np.where(hit, np.asarray(sid)[safe], -1).astype('int32'))
    return np.load(path)


def build(xl, sess, s):
    bad = validate(xl, s)
    r = s.row.to_numpy()
    O, H, L, C = [xl[r, :, i] for i in range(4)]
    S = sess[r]
    top, bot = s.zone_top.to_numpy(), s.zone_bottom.to_numpy()
    south = s.t0_exit_side.to_numpy() == 'south'
    w = np.maximum(top - bot, .25)
    near = np.where(south, bot, top)        # the exit boundary of the T0 side
    far = np.where(south, top, bot)
    q = minute_states(xl, s)
    z = np.array([v.index('Z') + 1 if 'Z' in v else 0 for v in q])
    ok = (bad == '') & (z > 0) & (z <= 49)
    i = np.flatnonzero(ok)
    zi = z[i]
    reach = np.where(south[i], H[i, zi] - near[i], near[i] - L[i, zi]) / w[i]
    close = np.where(south[i], C[i, zi] - near[i], near[i] - C[i, zi]) / w[i]
    trig = (reach < 1) & (close > .5)
    i, zi = i[trig], zi[trig]
    e = zi + 1
    entry = O[i, e]
    long_ = south[i]                        # south exit -> far side is above
    live = np.isfinite(entry) & (S[i, e] == S[i, zi]) & (S[i, e] >= 0)
    sgn = np.where(long_, 1.0, -1.0)
    stop = np.where(long_, near[i] - TICK, near[i] + TICK)
    t1 = far[i]
    t2 = np.where(long_, far[i] + w[i], far[i] - w[i])
    return dict(i=i, zi=zi, e=e, entry=entry, sgn=sgn, live=live, stop=stop,
                t1=t1, t2=t2, H=H[i], L=L[i], C=C[i], S=S[i],
                year=pd.to_datetime(s.t0_ts_ns.to_numpy()[i]).year)


def resolve(b, target):
    n = len(b['i'])
    res = np.full(n, 'timeout', dtype=object)
    pts = np.full(n, np.nan)
    for k in range(n):
        if not b['live'][k]:
            res[k] = 'censored'
            continue
        sg, en = b['sgn'][k], b['entry'][k]
        st, tg = b['stop'][k], target[k]
        last = en
        for m in range(b['e'][k], 51):
            if b['S'][k, m] != b['S'][k, b['e'][k]] or not np.isfinite(b['H'][k, m]):
                break
            hi, lo = b['H'][k, m], b['L'][k, m]
            hit_s = lo <= st if sg > 0 else hi >= st
            hit_t = hi >= tg if sg > 0 else lo <= tg
            if hit_s and hit_t:
                res[k] = 'ambiguous'
                pts[k] = np.nan
                break
            if hit_s:
                res[k] = 'stop'
                pts[k] = (st - en) * sg
                break
            if hit_t:
                res[k] = 'target'
                pts[k] = (tg - en) * sg
                break
            last = b['C'][k, m]
        else:
            res[k] = 'timeout'
            pts[k] = (last - en) * sg
        if res[k] == 'timeout' and np.isnan(pts[k]):
            pts[k] = (last - en) * sg
    return res, pts


def report(res, pts, yr, name):
    keep = np.isin(res, ['stop', 'target', 'timeout'])
    net = pts[keep] * POINT - COST
    out = dict(name=name, signals=int(len(res)),
               censored=int((res == 'censored').sum()),
               ambiguous=int((res == 'ambiguous').sum()),
               traded=int(keep.sum()),
               target=float((res[keep] == 'target').mean()),
               stop=float((res[keep] == 'stop').mean()),
               timeout=float((res[keep] == 'timeout').mean()),
               gross_per_trade=float(np.nanmean(pts[keep] * POINT)),
               net_per_trade=float(np.nanmean(net)),
               net_total=float(np.nansum(net)),
               t_stat=float(np.nanmean(net) / (np.nanstd(net) / np.sqrt(keep.sum()))))
    out['by_epoch'] = {}
    for lo, hi in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        m = keep & (yr >= lo) & (yr <= hi)
        if m.sum():
            out['by_epoch'][f'{lo}-{hi}'] = dict(
                n=int(m.sum()), net=float(np.nanmean(pts[m] * POINT - COST)))
    print(f"\n== {name}: signals={out['signals']} censored={out['censored']} "
          f"ambiguous={out['ambiguous']} traded={out['traded']}")
    print(f"   target={out['target']:.3f} stop={out['stop']:.3f} timeout={out['timeout']:.3f}")
    print(f"   gross=${out['gross_per_trade']:.2f}  net=${out['net_per_trade']:.2f}  "
          f"total=${out['net_total']:.0f}  t={out['t_stat']:.2f}")
    print('   by epoch:', {k: f"n={v['n']} ${v['net']:.2f}" for k, v in out['by_epoch'].items()})
    return out


def main():
    t0, X, xl, d = scenes()
    sess = session_window(t0)
    bs = [build(xl, sess, d.iloc[k:k + 40000].reset_index(drop=True))
          for k in range(0, len(d), 40000)]
    b = {k: np.concatenate([q[k] for q in bs]) for k in bs[0]}
    rep = {}
    for nm, tg in [('Target 1 - far boundary', b['t1']),
                   ('Target 2 - one zone width beyond', b['t2'])]:
        res, pts = resolve(b, tg)
        rep[nm] = report(res, pts, b['year'], nm)
    (OUT / 'frozen_trade.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
