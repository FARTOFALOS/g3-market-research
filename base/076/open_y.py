"""Executes PROCEDURE_FREEZE.md exactly. First and only reading of Y.

Nothing here may be tuned after seeing output. Field is read-only; Volume is
never loaded.
"""
import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
OUT = Path(__file__).parent
MIN = 60_000_000_000
CURSOR_OFFSET, LOOK, HORIZON, K = 2, 60, 30, 20
SFEAT = ['side', 'kind', 'dur', 'position', 'bulk']
HFEAT = ['runs_rate', 'exit_share', 'far_share', 'inside_share']
RNG = np.random.default_rng(20260914)

_cache = {}


def market(inst):
    if inst not in _cache:
        m = ROOT / 'data/market' / inst
        A = {n: np.load(m / (n + '.npy'), mmap_mode='r')
             for n in ['open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id']}
        with np.load(m / 'sessions.npz') as z:
            se = {n: z[n].copy() for n in z.files}
        se['lookup'] = {int(v): i for i, v in enumerate(se['session_id'])}
        _cache[inst] = (A, se)
    return _cache[inst]


def build(inst, tf):
    """S, H, Y for every passport of (inst, tf) that survives the frozen mask."""
    A, se = market(inst)
    cell = ROOT / f'data/field/{inst}/cells/tf_{tf:04d}'
    rows = pq.read_table(cell / 'passports.parquet',
                         columns=['riz_id', 'zone_bottom', 'zone_top', 't0_spine_pos',
                                  't0_exit_side', 't0_kind']).to_pylist()
    rows.sort(key=lambda r: (r['t0_spine_pos'], r['riz_id']))
    ts_all = A['close_ts_utc_ns']
    n_all = len(A['close'])
    out, drop = [], dict(prefix=0, ruler=0, position=0, future=0)
    for r in rows:
        t0 = int(r['t0_spine_pos']); p = t0 + CURSOR_OFFSET
        if p + HORIZON >= n_all:
            drop['future'] += 1; continue
        si = se['lookup'].get(int(A['session_id'][t0]))
        if si is None:
            drop['prefix'] += 1; continue
        anchor = int(se['session_open_utc_ns'][si])
        key = ((int(ts_all[t0]) - anchor) // MIN - 1) // tf
        begin = anchor + key * tf * MIN
        start = int(np.searchsorted(ts_all[:p + 1], begin, side='right'))
        ts = np.asarray(ts_all[start:p + 1])
        if len(ts) < 3 or int(ts[0]) != begin + MIN or np.any(np.diff(ts) != MIN):
            drop['prefix'] += 1; continue
        a = start - LOOK
        if a < 0:
            drop['ruler'] += 1; continue
        tsp = np.asarray(ts_all[a:start])
        if len(tsp) < LOOK or np.any(np.diff(tsp) != MIN):
            drop['ruler'] += 1; continue
        hp = np.asarray(A['high'][a:start], float); lp = np.asarray(A['low'][a:start], float)
        if not (np.isfinite(hp).all() and np.isfinite(lp).all()):
            drop['ruler'] += 1; continue
        u = float(np.median(hp - lp))
        if u <= 0:
            drop['ruler'] += 1; continue
        raw = np.column_stack([np.asarray(A[n][start:p + 1], float)
                               for n in ['open', 'high', 'low', 'close']])
        if not np.isfinite(raw).all():
            drop['prefix'] += 1; continue
        w = r['zone_top'] - r['zone_bottom']
        if w <= 0:
            drop['prefix'] += 1; continue
        north = r['t0_exit_side'] == 'north'
        ex = r['zone_top'] if north else r['zone_bottom']
        b = raw - ex if north else (ex - raw)[:, [0, 2, 1, 3]]
        o, h, l, c = b.T
        position = float(c[-1]) / u
        if position <= 0:                       # frozen mask: outside at close of p
            drop['position'] += 1; continue
        # ---- H, from the same prefix
        e = (l <= 0) & (h >= 0); f = (l <= -w) & (h >= -w)
        loc = np.select([c < -w, c == -w, c < 0, c == 0], [-2, -1, 0, 1], default=2)
        reads = list(zip(e.astype(int), f.astype(int), loc.astype(int)))
        runs = sum(1 for i, v in enumerate(reads) if i == 0 or v != reads[i - 1])
        d = float(len(b))
        # ---- Y, strictly after p, never touching S or H
        fts = np.asarray(ts_all[p + 1:p + 1 + HORIZON])
        if len(fts) < HORIZON or np.any(np.diff(fts) != MIN):
            drop['future'] += 1; continue
        fh = np.asarray(A['high'][p + 1:p + 1 + HORIZON], float)
        fl = np.asarray(A['low'][p + 1:p + 1 + HORIZON], float)
        if not (np.isfinite(fh).all() and np.isfinite(fl).all()):
            drop['future'] += 1; continue
        flo = (fl - ex) if north else (ex - fh)   # oriented low of each future minute
        Y = int(np.any(flo <= 0))
        out.append(dict(side=1.0 if north else 0.0,
                        kind=1.0 if r['t0_kind'] == 'minute_ignition' else 0.0,
                        dur=d, position=position, bulk=float(w) / u,
                        runs_rate=runs / d, exit_share=int(e.sum()) / d,
                        far_share=int(f.sum()) / d, inside_share=int((loc == 0).sum()) / d,
                        Y=Y, day=int(A['session_id'][p]),
                        date=str(np.datetime64(int(ts_all[p]), 'ns'))[:10]))
    return out, drop


def mat(rows, feats):
    return np.array([[r[f] for f in feats] for r in rows], float)


def irls(X, y, maxit=100):
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        z = X @ beta
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        Wd = np.maximum(pr * (1 - pr), 1e-10)
        try:
            step = np.linalg.solve(X.T @ (X * Wd[:, None]) + 1e-12 * np.eye(X.shape[1]),
                                   X.T @ (y - pr))
        except np.linalg.LinAlgError:
            return None, 'singular'
        beta = beta + step
        if not np.isfinite(beta).all() or np.max(np.abs(beta)) > 50:
            return None, 'separation'
        if np.max(np.abs(step)) < 1e-8:
            return beta, 'ok'
    return None, 'no_convergence'


def ll_from_logit(z, y):
    """numerically stable, no probability clipping"""
    s = np.where(y == 1, -z, z)
    return np.logaddexp(0.0, s)


def auc(p, y):
    o = np.argsort(p, kind='stable')
    r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_freeze(train):
    """returns everything frozen on the training set"""
    out = {}
    y = np.array([r['Y'] for r in train], float)
    for name, feats in [('S', SFEAT), ('S+H', SFEAT + HFEAT)]:
        X = mat(train, feats)
        mu, sd = X.mean(0), X.std(0)
        zero = [f for f, v in zip(feats, sd) if v == 0]
        keep = sd > 0
        Xs = (X[:, keep] - mu[keep]) / sd[keep]
        beta, status = irls(Xs, y)
        out[name] = dict(feats=feats, mu=mu, sd=sd, keep=keep, beta=beta,
                         status=status, zero=zero)
    out['ecdf'] = {f: np.sort(np.array([r[f] for r in train], float))
                   for f in ['position', 'bulk'] + HFEAT}
    out['train'] = train
    return out


def score(fr, test, label):
    y = np.array([r['Y'] for r in test], float)
    day = np.array([r['day'] for r in test])
    res = dict(label=label, n=len(test), base_rate=float(y.mean()),
               days=int(len(set(day.tolist()))))
    lls = {}
    for name in ['S', 'S+H']:
        m = fr[name]
        if m['beta'] is None:
            res[name] = dict(status=m['status']); continue
        X = mat(test, m['feats'])
        Xs = (X[:, m['keep']] - m['mu'][m['keep']]) / m['sd'][m['keep']]
        z = np.column_stack([np.ones(len(Xs)), Xs]) @ m['beta']
        lls[name] = ll_from_logit(z, y)
        res[name] = dict(status=m['status'], ll=float(lls[name].mean()),
                         auc=auc(1 / (1 + np.exp(-np.clip(z, -500, 500))), y))
    if len(lls) == 2:
        dd = lls['S'] - lls['S+H']
        res['delta'] = float(dd.mean())
        ud = np.unique(day)
        by = {d: dd[day == d] for d in ud}
        bs = np.array([np.concatenate([by[d] for d in RNG.choice(ud, len(ud))]).mean()
                       for _ in range(2000)])
        res['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return res


def knn_delta(fr, test):
    """secondary estimator, exactly as frozen"""
    train = fr['train']
    ytr = np.array([r['Y'] for r in train], float)
    def ranks(rows, feats):
        M = np.zeros((len(rows), len(feats)))
        for j, f in enumerate(feats):
            ref = fr['ecdf'][f]
            v = np.array([r[f] for r in rows], float)
            M[:, j] = np.clip(np.searchsorted(ref, v, side='right') / len(ref), 0, 1)
        return M
    sd_tr = np.array([r['side'] for r in train]); kd_tr = np.array([r['kind'] for r in train])
    du_tr = np.array([r['dur'] for r in train]); dy_tr = np.array([r['day'] for r in train])
    out = {}
    for name, feats in [('S', ['position', 'bulk']),
                        ('S+H', ['position', 'bulk'] + HFEAT)]:
        Rtr = ranks(train, feats); Rte = ranks(test, feats)
        p, used = [], []
        for i, r in enumerate(test):
            c = np.where((sd_tr == r['side']) & (kd_tr == r['kind'])
                         & (np.abs(du_tr - r['dur']) <= 3) & (dy_tr != r['day']))[0]
            if len(c) < K:
                p.append(np.nan); used.append(False); continue
            nb = c[np.argsort(np.sum((Rtr[c] - Rte[i]) ** 2, axis=1), kind='stable')][:K]
            p.append((ytr[nb].sum() + 1) / (K + 2)); used.append(True)
        out[name] = (np.array(p), np.array(used))
    ok = out['S'][1] & out['S+H'][1]
    y = np.array([r['Y'] for r in test], float)[ok]
    d = np.array([-(y * np.log(out[n][0][ok]) + (1 - y) * np.log(1 - out[n][0][ok]))
                  for n in ['S', 'S+H']])
    return float((d[0] - d[1]).mean()), int(ok.sum()), int((~ok).sum())


if __name__ == '__main__':
    DATA = {}
    for inst, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
        rows, drop = build(inst, tf)
        DATA[(inst, tf)] = rows
        print('%s tf%-3d  n=%5d  Y-rate %.4f  dropped %s'
              % (inst, tf, len(rows), np.mean([r['Y'] for r in rows]) if rows else 0, drop),
              flush=True)

    def sel(k, lo, hi):
        return [r for r in DATA[k] if lo <= r['date'] <= hi]

    EARLY, LATE = ('2006-01-05', '2018-12-31'), ('2019-01-01', '2026-05-04')
    results = []

    print('\n' + '=' * 84)
    print('CONFIRMATORY: train NQ tf54 2006-2018  ->  ES/YM tf54 2019-2026')
    print('=' * 84)
    fr = fit_freeze(sel(('NQ', 54), *EARLY))
    print('training n=%d  S status=%s  S+H status=%s  zero-sd=%s'
          % (len(fr['train']), fr['S']['status'], fr['S+H']['status'],
             fr['S']['zero'] + fr['S+H']['zero']))
    for inst in ['ES', 'YM']:
        te = sel((inst, 54), *LATE)
        r = score(fr, te, 'CONFIRM %s tf54 2019-2026' % inst)
        r['knn'] = knn_delta(fr, te)
        results.append(r)
        print('\n%s  n=%d days=%d  Y-rate %.4f' % (r['label'], r['n'], r['days'], r['base_rate']))
        print('   LL(S)=%.5f  LL(S+H)=%.5f' % (r['S']['ll'], r['S+H']['ll']))
        print('   AUC(S)=%.4f AUC(S+H)=%.4f' % (r['S']['auc'], r['S+H']['auc']))
        print('   DELTA = %+.6f   95%% block CI [%+.6f, %+.6f]'
              % (r['delta'], r['ci'][0], r['ci'][1]))
        print('   kNN   delta = %+.6f  (used %d, skipped %d)' % r['knn'])

    print('\n' + '=' * 84)
    print('DIAGNOSTICS (reported, not part of the criterion)')
    print('=' * 84)
    frA = fit_freeze(DATA[('NQ', 54)])
    for lbl, fr_, key, per in [
            ('instrument only: NQ06-26 -> ES06-26', frA, ('ES', 54), ('2006-01-05', '2026-05-04')),
            ('instrument only: NQ06-26 -> YM06-26', frA, ('YM', 54), ('2006-01-05', '2026-05-04')),
            ('time only: NQ06-18 -> NQ19-26', fr, ('NQ', 54), LATE),
            ('projection: NQ06-18 -> NQtf30 19-26', fr, ('NQ', 30), LATE)]:
        te = sel(key, *per)
        r = score(fr_, te, lbl)
        results.append(r)
        print('%-38s n=%5d  D=%+.6f  CI [%+.6f, %+.6f]'
              % (lbl, r['n'], r['delta'], r['ci'][0], r['ci'][1]))

    json.dump(results, open(OUT / 'open_y_results.json', 'w'), indent=1, default=str)
    print('\nwritten: open_y_results.json')
