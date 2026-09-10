"""074: the transferable states of 073 followed as whole trajectories T0..+50.

For each state three groups are built from the SAME frozen predicate:
  core     - every relation of the transitively irreducible basis holds;
  variant  - the ladder (the separation relations) holds, the full basis does not;
  near     - exactly one basis relation is violated: the closest rejected films.

Each film is then read minute by minute in the language of its own RIZ zone:
  zone / exit-side / opposite. Nothing after the current minute is used to build
  a group; the trajectory is what is being looked at, not a filter.

Run from repository root: python -B base/074/trajectories.py
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'work/074'
OUT = Path(__file__).resolve().parent
F = 'OHLC'


def parse(r):
    m = re.match(r'(\w)\[(\d)\] ([<>]) (\w)\[(\d)\]', r)
    return 4 * int(m.group(2)) + F.index(m.group(1)), m.group(3), 4 * int(m.group(5)) + F.index(m.group(4))


def main():
    z8 = np.load(CACHE / 'NQ_prefix.npz')
    t0, x8 = z8['t0'], z8['x']
    X = x8.reshape(len(t0), 32)
    zl = np.load(CACHE / 'NQ_prefix51.npz')
    xl = zl['x']

    d = pd.read_parquet(CACHE / 'NQ_zones.parquet')
    d = d.drop_duplicates(['t0_ts_ns', 'zone_top', 'zone_bottom', 't0_exit_side'])
    row = pd.Series(np.arange(len(t0)), index=t0.astype('int64'))
    d = d[d.t0_ts_ns.isin(row.index)].copy()
    d['row'] = row.loc[d.t0_ts_ns].to_numpy()

    r = d.row.to_numpy()
    H, L = xl[r, :, 1], xl[r, :, 2]
    top, bot = d.zone_top.to_numpy()[:, None], d.zone_bottom.to_numpy()[:, None]
    south = (d.t0_exit_side.to_numpy() == 'south')[:, None]
    kn = np.isfinite(H) & np.isfinite(L)
    at_zone = (L <= top) & (H >= bot)
    on_exit = np.where(south, H < bot, L > top)
    state = np.where(~kn, 3, np.where(at_zone, 0, np.where(on_exit, 1, 2)))  # 0 zone 1 exit 2 opposite

    prep = json.load(open(OUT / 'preparation.json', encoding='utf-8'))
    rep = {}
    ok_all = state != 3
    base = [dict(k=k, n=int(ok_all[:, k].sum()),
                 **{lbl: float((state[ok_all[:, k], k] == v).mean())
                    for v, lbl in [(0, 'zone'), (1, 'exit'), (2, 'opposite')]})
            for k in range(51)]
    rep['field_baseline'] = base
    print('field baseline (all %d scenes):' % len(d))
    for k in [0, 1, 2, 3, 5, 7, 10, 20, 30, 40, 50]:
        q = base[k]
        print(f"   +{k:2d} {q['zone']:.2f}/{q['exit']:.2f}/{q['opposite']:.2f}", flush=True)
    for p in prep:
        rels = p['basis_relations']
        hit = np.ones((len(t0), len(rels)), bool)
        known = np.ones(len(t0), bool)
        for j, rl in enumerate(rels):
            i, op, k = parse(rl)
            v = X[:, i] - X[:, k]
            known &= np.isfinite(v)
            hit[:, j] = (v > 0) if op == '>' else (v < 0)
        miss = (~hit).sum(1)
        lad = np.ones(len(t0), bool)
        for rl in p['ladder_relations']:
            i, op, k = parse(rl)
            v = X[:, i] - X[:, k]
            lad &= (v > 0) if op == '>' else (v < 0)
        groups = {'core': known & (miss == 0),
                  'variant': known & lad & (miss > 0),
                  'near': known & (miss == 1)}
        block = {'seed': p['seed'], 't0': p['t0'], 'basis': len(rels)}
        share = {}
        for nm, film in groups.items():
            sel = film[d.row.to_numpy()]
            n = int(sel.sum())
            st = state[sel]
            ok = st != 3
            share[nm] = [
                dict(k=k, n=int(ok[:, k].sum()),
                     **{lbl: float((st[ok[:, k], k] == v).mean()) if ok[:, k].any() else None
                        for v, lbl in [(0, 'zone'), (1, 'exit'), (2, 'opposite')]})
                for k in range(51)]
            block[nm + '_n'] = n
        block['by_minute'] = share
        # where the closest rejected films start to differ from the core,
        # measured against the core's own sampling spread
        rng = np.random.default_rng(74)
        selc = groups['core'][d.row.to_numpy()]
        seln = groups['near'][d.row.to_numpy()]
        stc, stn = state[selc], state[seln]
        diverge = None
        rows = []
        for k in range(1, 51):
            a = stc[stc[:, k] != 3, k]
            b = stn[stn[:, k] != 3, k]
            if len(a) < 30 or len(b) < 30:
                continue
            pa = np.array([(a == v).mean() for v in range(3)])
            pb = np.array([(b == v).mean() for v in range(3)])
            gap = np.abs(pa - pb).sum() / 2
            half = [np.abs(np.array([(a[i] == v).mean() for v in range(3)]) -
                           np.array([(a[~np.isin(np.arange(len(a)), i)] == v).mean() for v in range(3)])).sum() / 2
                    for i in [rng.choice(len(a), len(a) // 2, replace=False) for _ in range(12)]]
            band = float(np.percentile(half, 90))
            rows.append(dict(k=k, gap=round(float(gap), 4), core_band=round(band, 4),
                             n_core=len(a), n_near=len(b)))
            if diverge is None and gap > band:
                diverge = k
        block['divergence'] = dict(first_minute=diverge, detail=rows)
        rep[str(p['seed'])] = block
        print(f"seed {p['seed']}: core={block['core_n']} variant={block['variant_n']} "
              f"near={block['near_n']} first divergence of near vs core: {diverge}", flush=True)
        for k in [0, 1, 2, 3, 5, 7, 10, 20, 30, 40, 50]:
            c = share['core'][k]
            v = share['variant'][k]
            nr = share['near'][k]
            f = lambda q: '-' if q['zone'] is None else f"{q['zone']:.2f}/{q['exit']:.2f}/{q['opposite']:.2f}"
            print(f'   +{k:2d} core {f(c)}   variant {f(v)}   near {f(nr)}', flush=True)
    (OUT / 'trajectories.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
