"""Does the equivalence survive when the ruler is known BEFORE the path starts?

The previous step used u = median(H-L) of the observed interval itself: the path
helped define its own ruler. Here u is recomputed from minutes strictly earlier
than the interval, so the scale exists before the path does.

Reads the frozen field read-only. No Y, no Volume, no model. Reports support and
road-variation without tuning toward either.
"""
import json, collections
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
OUT = Path(__file__).parent
MINUTE = 60_000_000_000
S = json.load(open(OUT / 'support_probe.json', encoding='utf-8'))['states']
N = len(S)

m = ROOT / 'data/market/NQ'
A = {n: np.load(m / (n + '.npy'), mmap_mode='r')
     for n in ['open', 'high', 'low', 'close', 'close_ts_utc_ns']}
print('market arrays loaded, %d minutes' % len(A['close']))

LOOKBACKS = [30, 60, 120]


def prior_u(start, k):
    """median (H-L) over the k contiguous minutes strictly before `start`."""
    a = start - k
    if a < 0:
        return None
    ts = np.asarray(A['close_ts_utc_ns'][a:start])
    if len(ts) < k or np.any(np.diff(ts) != MINUTE):
        return None
    h = np.asarray(A['high'][a:start], float)
    l = np.asarray(A['low'][a:start], float)
    if not np.isfinite(h).all() or not np.isfinite(l).all():
        return None
    v = float(np.median(h - l))
    return v if v > 0 else None


U_endo = np.array([float(np.median(np.asarray(s['raw_ohlc'], float)[:, 1] -
                                   np.asarray(s['raw_ohlc'], float)[:, 2]))
                   for s in S])
W = np.array([s['g0'][2] for s in S])
DUR = np.array([s['g0'][10] for s in S], float)
day = [s['session_id'] for s in S]

U_prior = {}
for k in LOOKBACKS:
    U_prior[k] = np.array([prior_u(s['start'], k) if prior_u(s['start'], k) else np.nan
                           for s in S])
    print('lookback %3d min: %d of %d states have a clean prior window'
          % (k, np.isfinite(U_prior[k]).sum(), N))

print('\n' + '-' * 84)
print('1. DOES THE ENDOGENOUS RULER ENCODE THE WINDOW ITSELF?')
print('-' * 84)
print('   corr(u_endo, duration)        = %+.3f' % np.corrcoef(U_endo, DUR)[0, 1])
print('   corr(log u_endo, log duration)= %+.3f'
      % np.corrcoef(np.log(U_endo), np.log(DUR))[0, 1])
# u_endo is the median range of exactly the minutes whose contacts form H
print('   u_endo is the median range of the SAME minutes whose boundary contacts')
print('   make up H, so part of H is inside the ruler by construction.')

print('\n' + '-' * 84)
print('2. HOW WELL DOES A PRIOR RULER STAND IN FOR THE ENDOGENOUS ONE?')
print('-' * 84)
for k in LOOKBACKS:
    u = U_prior[k]
    ok = np.isfinite(u)
    r = U_endo[ok] / u[ok]
    print('   %3d min: n=%3d  corr(log,log)=%+.3f  u_endo/u_prior p5=%.2f med=%.2f p95=%.2f'
          % (k, ok.sum(), np.corrcoef(np.log(U_endo[ok]), np.log(u[ok]))[0, 1],
             *np.percentile(r, [5, 50, 95])))
print('   agreement between prior rulers themselves:')
for i in range(len(LOOKBACKS) - 1):
    a, b = LOOKBACKS[i], LOOKBACKS[i + 1]
    ok = np.isfinite(U_prior[a]) & np.isfinite(U_prior[b])
    r = U_prior[a][ok] / U_prior[b][ok]
    print('     %3d vs %3d min: corr(log,log)=%+.3f  ratio p5=%.2f med=%.2f p95=%.2f'
          % (a, b, np.corrcoef(np.log(U_prior[a][ok]), np.log(U_prior[b][ok]))[0, 1],
             *np.percentile(r, [5, 50, 95])))


def state_blocks(u, cut=None, with_bulk=True):
    ok = np.isfinite(u)
    ratio = np.where(ok, W / u, np.nan)
    if cut is None:
        cut = np.nanpercentile(ratio, [100 / 3, 200 / 3])
    g = collections.defaultdict(list)
    for i, s in enumerate(S):
        if not ok[i]:
            continue
        key = [s['g0'][0], s['g0'][1], s['g0'][10], int(np.floor(s['g0'][9] / u[i]))]
        if with_bulk:
            key.append(int(np.searchsorted(cut, ratio[i])))
        g[tuple(key)].append(i)
    bl = []
    for mem in g.values():
        seen, keep = set(), []
        for i in mem:
            if day[i] in seen:
                continue
            seen.add(day[i]); keep.append(i)
        if len(keep) > 1:
            bl.append(keep)
    return bl, cut


print('\n' + '-' * 84)
print('3. DOES THE EQUIVALENCE SURVIVE A RULER KNOWN BEFORE THE PATH?')
print('-' * 84)
print('%-26s %5s %7s %7s %5s %5s %8s %14s'
      % ('ruler', 'n ok', 'blocks', 'states', 'days', 'yrs', 'largest', 'blocks 2+ roads'))


def report(name, u):
    bl, cut = state_blocks(u)
    if not bl:
        print('%-26s %5d %7d' % (name, np.isfinite(u).sum(), 0))
        return bl
    idx = [i for b in bl for i in b]
    multi = sum(1 for b in bl
                if len({tuple(map(tuple, S[i]['h_order'])) for i in b}) > 1)
    print('%-26s %5d %7d %7d %5d %5d %8d %10d/%d'
          % (name, int(np.isfinite(u).sum()), len(bl), len(idx),
             len({day[i] for i in idx}), len({S[i]['p_utc'][:4] for i in idx}),
             max(len(b) for b in bl), multi, len(bl)))
    return bl


bl_endo = report('endogenous (interval)', U_endo)
bl_prior = {k: report('prior %d min' % k, U_prior[k]) for k in LOOKBACKS}

# restrict the endogenous ruler to the same states a prior ruler can serve, so
# the comparison is not confounded by different populations
k0 = 60
ok = np.isfinite(U_prior[k0])
u_endo_sub = np.where(ok, U_endo, np.nan)
print()
report('endogenous, same states', u_endo_sub)

print('\n' + '-' * 84)
print('4. DO THE TWO RULERS PUT THE SAME SCENES TOGETHER?')
print('-' * 84)
for k in LOOKBACKS:
    a, _ = state_blocks(np.where(np.isfinite(U_prior[k]), U_endo, np.nan))
    b, _ = state_blocks(U_prior[k])
    pa = {frozenset(x) for x in a}
    pb = {frozenset(x) for x in b}
    # pairwise agreement: of all co-grouped pairs under one, how many co-group under other
    def pairs(bl):
        out = set()
        for g in bl:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    out.add((min(g[i], g[j]), max(g[i], g[j])))
        return out
    A_, B_ = pairs(a), pairs(b)
    inter = len(A_ & B_)
    print('   %3d min: endo pairs %4d, prior pairs %4d, shared %3d '
          '(Jaccard %.3f)' % (k, len(A_), len(B_), inter,
                              inter / len(A_ | B_) if (A_ | B_) else 0))

print('\n' + '-' * 84)
print('5. RECONSTRUCTION CHECK ON THE PRIOR RULER (60 min)')
print('-' * 84)
bl, cut = state_blocks(U_prior[60])
bl.sort(key=len, reverse=True)
u = U_prior[60]
for b in bl[:3]:
    s0 = S[b[0]]
    print('  %s dur=%d pos=%d bulk=%d  (%d scenes)'
          % (s0['g0'][0], s0['g0'][10], int(np.floor(s0['g0'][9] / u[b[0]])),
             int(np.searchsorted(cut, W[b[0]] / u[b[0]])), len(b)))
    for i in b:
        s = S[i]
        print('    %s  w=%6.2f u_prior=%5.2f u_endo=%5.2f  w/u=%5.2f  pos=%5.2f u  runs=%2d'
              % (s['p_utc'], s['g0'][2], u[i], U_endo[i], W[i] / u[i],
                 s['g0'][9] / u[i], len(s['h_order'])))
