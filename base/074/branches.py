"""074: retrospective branch of the whole film T0..+50 relative to one RIZ.

Minute state (+1..+50), exactly one per minute:
  north exit: E if low > zone_top;  F if high < zone_bottom;  else Z
  south exit: E if high < zone_bottom;  F if low > zone_top;  else Z
A touch of either border is Z.

Mutually exclusive classification, strictly in this order:
  CENSORED      a minute is missing, or the state jumps E<->F between adjacent
                minutes without a Z between them: minute OHLC does not show the
                way through the zone, so the path is not observed;
  OSCILLATION   after the first Z both E and F occur, in any order;
  TRAVERSE      after the first Z an F occurs, and after that first F never an E;
  EXIT-RECLAIM  after the first Z no F ever, and the state at +50 is E;
  ZONE-ABSORB   after the first Z no F ever, and the state at +50 is Z;
  EXIT-HOLD     all fifty minutes are E.

Retrospective labels of a finished film. Never used as prefix information.
Run: python -B base/074/branches.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from first_difference import scenes, membership          # noqa: E402

OUT = Path(__file__).resolve().parent


def validate(xl, s):
    """object validity -> 50 minutes present -> time continuity -> OHLC sanity.

    The window is cut by exact timestamps t0 + k*60s, so a missing, duplicated or
    out-of-order minute cannot silently shift the film: it lands as NaN. The
    monotonicity of the source spine itself is checked once, separately.
    """
    r = s.row.to_numpy()
    O, H, L, C = [xl[r, :, i] for i in range(4)]
    reason = np.array([''] * len(s), dtype=object)
    zone_ok = (np.isfinite(s.zone_top.to_numpy()) & np.isfinite(s.zone_bottom.to_numpy())
               & (s.zone_bottom.to_numpy() <= s.zone_top.to_numpy())
               & s.t0_exit_side.isin(['north', 'south']).to_numpy())
    reason[~zone_ok] = 'INDETERMINATE'
    present = np.isfinite(np.stack([O, H, L, C], 2)).all((1, 2))
    reason[(reason == '') & ~present] = 'CENSORED'
    body_lo = np.minimum(O, C)
    body_hi = np.maximum(O, C)
    sane = ((L <= body_lo) & (body_lo <= body_hi) & (body_hi <= H)).all(1)
    reason[(reason == '') & present & ~sane] = 'CENSORED'
    return reason


def spine_is_monotonic():
    ts = np.load(Path(__file__).resolve().parents[2] / 'data/market/NQ/close_ts_utc_ns.npy',
                 mmap_mode='r')
    step = np.diff(np.asarray(ts[:200000]))
    return bool((step > 0).all())


def minute_states(xl, s):
    r = s.row.to_numpy()
    H, L = xl[r, :, 1], xl[r, :, 2]
    top, bot = s.zone_top.to_numpy()[:, None], s.zone_bottom.to_numpy()[:, None]
    south = (s.t0_exit_side.to_numpy() == 'south')[:, None]
    kn = np.isfinite(H) & np.isfinite(L)
    E = np.where(south, H < bot, L > top)
    Fs = np.where(south, L > top, H < bot)
    out = np.where(~kn, '.', np.where(E, 'E', np.where(Fs, 'F', 'Z')))
    return [''.join(row[1:]) for row in out]


def classify(q):
    """returns label and the addresses that justify it"""
    info = dict(first_Z=None, first_E_after_Z=None, first_F_after_Z=None,
                first_E_after_F=None, state_50=q[-1] if q else None)
    if len(q) != 50 or '.' in q:
        return 'CENSORED', info
    for i in range(1, 50):
        if {q[i - 1], q[i]} == {'E', 'F'}:
            # minute OHLC does not show the way through the zone
            return 'INDETERMINATE', info
    if 'Z' not in q:
        return ('EXIT-HOLD' if set(q) == {'E'} else 'INDETERMINATE'), info
    z = q.index('Z')
    info['first_Z'] = z + 1
    tail = q[z:]
    if 'E' in tail:
        info['first_E_after_Z'] = z + tail.index('E') + 1
    if 'F' in tail:
        info['first_F_after_Z'] = z + tail.index('F') + 1
        f = tail.index('F')
        if 'E' in tail[f:]:
            info['first_E_after_F'] = z + f + tail[f:].index('E') + 1
    if 'E' in tail and 'F' in tail:
        return 'OSCILLATION', info
    if 'F' in tail:
        return 'TRAVERSE', info
    if q[-1] == 'E':
        return 'EXIT-RECLAIM', info
    if q[-1] == 'Z':
        return 'ZONE-ABSORB', info
    return 'INDETERMINATE', info


def label_all():
    t0, X, xl, d = scenes()
    prep = json.load(open(OUT / 'preparation.json', encoding='utf-8'))
    mem = np.zeros(len(X), bool)
    for p in prep:
        if p['seed'] != 6727:
            mem |= membership(X, p)
    s = d[mem[d.row.to_numpy()]].reset_index(drop=True)
    bad = validate(xl, s)
    q = minute_states(xl, s)
    lab, info = zip(*[classify(v) for v in q])
    lab = np.array(lab, dtype=object)
    lab[bad != ''] = bad[bad != '']
    return t0, X, xl, s, list(q), lab, list(info)


def main():
    t0, X, xl, s, q, lab, info = label_all()
    print('source spine strictly increasing:', spine_is_monotonic())
    print(pd.Series(lab).value_counts().to_string())
    rows = [dict(row=int(s.row[i]), t0=str(pd.Timestamp(s.t0_ts_ns[i])),
                 tf=int(s.tf_minutes[i]), exit=s.t0_exit_side[i],
                 zone=[float(s.zone_bottom[i]), float(s.zone_top[i])],
                 label=lab[i], seq=q[i], **info[i]) for i in range(len(s))]
    (OUT / 'branches.json').write_text(json.dumps(
        dict(counts={k: int(v) for k, v in pd.Series(lab).value_counts().items()},
             films=rows), ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nhand-check sample: two films of each class')
    rng = np.random.default_rng(74)
    for cls in ['OSCILLATION', 'TRAVERSE', 'EXIT-RECLAIM', 'ZONE-ABSORB', 'EXIT-HOLD',
                'INDETERMINATE', 'CENSORED']:
        idx = np.flatnonzero(lab == cls)
        if not len(idx):
            continue
        for i in rng.choice(idx, min(2, len(idx)), replace=False):
            print(f'{cls:13s} row={s.row[i]:6d} firstZ={info[i]["first_Z"]} '
                  f'firstF={info[i]["first_F_after_Z"]} firstE_after_F={info[i]["first_E_after_F"]} '
                  f'+50={info[i]["state_50"]}\n              {q[i]}')


if __name__ == '__main__':
    main()
