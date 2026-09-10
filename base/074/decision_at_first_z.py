"""074: two questions about the difference found at the first returning minute.

1. Does it hold separately inside each rich state and inside each epoch?
2. Is a decision formulable at the close of that minute, before the branch?

Everything used is closed at that minute: the zone borders, the exit side, the
close of the candle. The race that follows - which border of the zone is touched
first afterwards - is measured structurally, in the film's own candles. No money,
no Volume, no outcome label as an input.

Run: python -B base/074/decision_at_first_z.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branches import classify, minute_states, validate      # noqa: E402
from first_difference import scenes, membership             # noqa: E402

OUT = Path(__file__).resolve().parent
BUCKET = ['touch', '<half', '>half']


def frame(xl, s):
    bad = validate(xl, s)
    q = minute_states(xl, s)
    lab, info = zip(*[classify(v) for v in q])
    lab = np.array(lab, dtype=object)
    lab[bad != ''] = bad[bad != '']
    r = s.row.to_numpy()
    O, H, L, C = [xl[r, :, i] for i in range(4)]
    top, bot = s.zone_top.to_numpy(), s.zone_bottom.to_numpy()
    south = s.t0_exit_side.to_numpy() == 'south'
    w = np.maximum(top - bot, .25)
    near, far = np.where(south, bot, top), np.where(south, top, bot)
    z = np.array([i['first_Z'] or 0 for i in info])
    use = (z > 0) & np.isin(lab, ['TRAVERSE', 'EXIT-RECLAIM', 'ZONE-ABSORB', 'OSCILLATION'])
    i = np.flatnonzero(use)
    zi = z[i]
    reach = np.where(south[i], H[i, zi] - near[i], near[i] - L[i, zi]) / w[i]
    close = np.where(south[i], C[i, zi] - near[i], near[i] - C[i, zi]) / w[i]
    # the race after that close: which border is met first
    n = len(i)
    hit_far = np.full(n, 99)
    hit_near = np.full(n, 99)
    for k in range(1, 51):
        m = zi + k
        ok = m <= 50
        idx = np.flatnonzero(ok)
        hf = np.where(south[i][idx], H[i[idx], m[idx]] >= far[i][idx],
                      L[i[idx], m[idx]] <= far[i][idx])
        hn = np.where(south[i][idx], L[i[idx], m[idx]] <= near[i][idx],
                      H[i[idx], m[idx]] >= near[i][idx])
        hit_far[idx[hf & (hit_far[idx] == 99)]] = k
        hit_near[idx[hn & (hit_near[idx] == 99)]] = k
    scale = np.nanmedian(H[i, :8] - L[i, :8], 1)
    scale = np.where(scale > 0, scale, np.nan)   # a flat film has no candle scale
    return pd.DataFrame(dict(
        branch=lab[i], reach=reach,
        close=pd.cut(close, [-9, 0, .5, 1], labels=BUCKET),
        minute=zi, year=s.year.to_numpy()[i] if 'year' in s else
        pd.to_datetime(s.t0_ts_ns.to_numpy()[i]).year,
        seed=s.get('seed', pd.Series(['-'] * len(s))).to_numpy()[i],
        far_first=hit_far < hit_near, no_touch=(hit_far == 99) & (hit_near == 99),
        to_far=np.abs(far[i] - C[i, zi]) / scale, to_near=np.abs(C[i, zi] - near[i]) / scale,
        wait=np.minimum(hit_far, hit_near)))


def part(df, title):
    d = df[(df.reach < 1) & df['close'].notna()]
    p = pd.crosstab(d['close'], d.branch, normalize='index').round(3)
    p['far_first'] = d.groupby('close', observed=False).far_first.mean().round(3)
    p['n'] = d['close'].value_counts()
    print(f'\n== {title} (n={len(d)})')
    print(p.reindex(BUCKET).to_string())
    return p.reindex(BUCKET).to_dict()


def main():
    t0, X, xl, d = scenes()
    d['year'] = pd.to_datetime(d.t0_ts_ns).dt.year.to_numpy()
    prep = json.load(open(OUT / 'preparation.json', encoding='utf-8'))
    rep = {}
    for p in prep:
        if p['seed'] == 6727:
            continue
        s = d[membership(X, p)[d.row.to_numpy()]].reset_index(drop=True)
        rep[f"state_{p['seed']}"] = part(frame(xl, s), f"rich state {p['seed']}")
    mem = np.zeros(len(X), bool)
    for p in prep:
        if p['seed'] != 6727:
            mem |= membership(X, p)
    rest = d[~mem[d.row.to_numpy()]].reset_index(drop=True)
    parts = [frame(xl, rest.iloc[k:k + 40000].reset_index(drop=True))
             for k in range(0, len(rest), 40000)]
    R = pd.concat(parts, ignore_index=True)
    rep['field'] = part(R, 'rest of the field')
    for lo, hi in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        rep[f'epoch_{lo}_{hi}'] = part(R[(R.year >= lo) & (R.year <= hi)], f'field {lo}-{hi}')

    D = R[(R.reach < 1) & R['close'].notna()]
    dec = D[D['close'] == '>half']
    rep['decision'] = dict(
        n=int(len(dec)), share_of_scenes=float(len(dec) / len(R)),
        available_minute_median=float(dec.minute.median()),
        available_minute_p90=float(dec.minute.quantile(.9)),
        far_first=float(dec.far_first.mean()),
        never_touched_either=float(dec.no_touch.mean()),
        median_wait_minutes=float(dec.wait[dec.wait < 99].median()),
        to_far_candles_median=float(dec.to_far.median()),
        to_near_candles_median=float(dec.to_near.median()),
        touch_first_by_epoch={f'{lo}-{hi}': float(dec[(dec.year >= lo) & (dec.year <= hi)].far_first.mean())
                              for lo, hi in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]})
    print('\n== decision at the close of the first Z, close past the midpoint')
    print(json.dumps(rep['decision'], ensure_ascii=False, indent=1))
    (OUT / 'decision_at_first_z.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
