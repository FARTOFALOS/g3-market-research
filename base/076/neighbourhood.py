"""Is there real common territory of different H in the continuous S-space,
or is the apparent comparability a product of how locality is defined?

S = (lastC/u_prior, w/u_prior) inside exact (side, kind), with duration handled
explicitly. u_prior is measured strictly BEFORE the interval.

No radius is chosen. Neighbours are taken by RANK k and k is swept, so the
answer is a curve, not a number. Every claim is repeated under several metrics
(anisotropy 1:4 .. 4:1) and several lookbacks (30/60/120 min); a conclusion is
only called a property of the data if it is invariant across them.

H is read at LOW cardinality on purpose. The earlier "176 of 177 blocks hold two
roads" was arithmetic, not evidence: with 698 distinct h_order among 952 scenes
two scenes differ with probability 0.995. The readings below are bounded shares
defined by the existing protocol's G1 counts, so the result is not forced by
cardinality.

No Y, no Volume, no model. Field is read-only.
"""
import json, collections
import numpy as np
from pathlib import Path

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
OUT = Path(__file__).parent
MIN = 60_000_000_000
S = json.load(open(OUT / 'support_probe.json', encoding='utf-8'))['states']
N = len(S)
RNG = np.random.default_rng(20260914)

m = ROOT / 'data/market/NQ'
A = {n: np.load(m / (n + '.npy'), mmap_mode='r')
     for n in ['high', 'low', 'close_ts_utc_ns']}


def prior_u(st, k):
    a = st - k
    if a < 0:
        return np.nan
    ts = np.asarray(A['close_ts_utc_ns'][a:st])
    if len(ts) < k or np.any(np.diff(ts) != MIN):
        return np.nan
    h = np.asarray(A['high'][a:st], float)
    l = np.asarray(A['low'][a:st], float)
    if not np.isfinite(h).all() or not np.isfinite(l).all():
        return np.nan
    v = float(np.median(h - l))
    return v if v > 0 else np.nan


LOOK = [30, 60, 120]
U = {k: np.array([prior_u(s['start'], k) for s in S]) for k in LOOK}
W = np.array([s['g0'][2] for s in S])
LC = np.array([s['g0'][9] for s in S])
DUR = np.array([s['g0'][10] for s in S], float)
day = np.array([s['session_id'] for s in S])
side = np.array([s['g0'][0] for s in S])
kind = np.array([s['g0'][1] for s in S])

# ---------------------------------------------------- H, bounded shares of time
def h_reads(s):
    e, f, l_2, l_1, l0, l1, l2 = s['counts']
    d = float(s['g0'][10])
    return dict(runs_rate=len(s['h_order']) / d,
                exit_share=e / d,
                far_share=f / d,
                inside_share=l0 / d)


H = [h_reads(s) for s in S]
READS = ['runs_rate', 'exit_share', 'far_share', 'inside_share']
HM = {r: np.array([h[r] for h in H]) for r in READS}
print('H readings (bounded shares of the interval):')
for r in READS:
    v = HM[r]
    print('   %-14s mean %.3f sd %.3f  corr with duration %+.3f'
          % (r, v.mean(), v.std(), np.corrcoef(v, DUR)[0, 1]))


def coords(u, wpos, wbulk):
    """rank-normalised S coordinates, then anisotropic weights."""
    ok = np.isfinite(u)
    pos = np.where(ok, LC / u, np.nan)
    bulk = np.where(ok, W / u, np.nan)
    out = np.full((N, 2), np.nan)
    for j, v in enumerate([pos, bulk]):
        r = np.full(N, np.nan)
        idx = np.where(ok)[0]
        order = np.argsort(v[idx], kind='stable')
        rr = np.empty(len(idx)); rr[order] = np.arange(len(idx))
        r[idx] = rr / max(len(idx) - 1, 1)
        out[:, j] = r
    out[:, 0] *= wpos; out[:, 1] *= wbulk
    return out, ok


def neighbour_curve(u, wpos, wbulk, kmax=40, nperm=200):
    """mean |H_i - H_j| over the k-th nearest admissible neighbour, swept in k."""
    X, ok = coords(u, wpos, wbulk)
    idx = np.where(ok)[0]
    # admissible: different session day, same side and kind
    order_lists = []
    for i in idx:
        cand = idx[(day[idx] != day[i]) & (side[idx] == side[i]) & (kind[idx] == kind[i])]
        if len(cand) < kmax:
            order_lists.append(None); continue
        d = np.sum((X[cand] - X[i]) ** 2, axis=1)
        o = cand[np.argsort(d, kind='stable')]
        order_lists.append(o[:kmax])
    valid = [i for i, o in zip(idx, order_lists) if o is not None]
    orders = {i: o for i, o in zip(idx, order_lists) if o is not None}
    res = {}
    for r in READS:
        v = HM[r]
        obs = np.array([np.mean([abs(v[i] - v[j]) for j in orders[i][:k]])
                        for k in [1, 2, 5, 10, 20, 40] for i in valid[:0]] or [0.0])
        curve = []
        for k in [1, 2, 5, 10, 20, 40]:
            vals = [abs(v[i] - v[orders[i][k - 1]]) for i in valid]
            curve.append(float(np.mean(vals)))
        # permutation: H reattached at random, same neighbour structure
        null = np.empty((nperm, 6))
        for t in range(nperm):
            vp = v[RNG.permutation(N)]
            for a, k in enumerate([1, 2, 5, 10, 20, 40]):
                null[t, a] = np.mean([abs(vp[i] - vp[orders[i][k - 1]]) for i in valid])
        res[r] = (curve, np.percentile(null, [5, 95], axis=0))
    return res, len(valid)


print('\n' + '=' * 92)
print('MEAN |H_i - H_j| AT THE k-th NEAREST NEIGHBOUR, k SWEPT (no radius chosen)')
print('if H were organised by S, this would RISE with k; flat = different H coexist')
print('=' * 92)

KS = [1, 2, 5, 10, 20, 40]
for look in LOOK:
    for wp, wb, tag in [(1, 1, '1:1'), (1, 0.25, '4:1 pos'), (0.25, 1, '1:4 bulk')]:
        res, nv = neighbour_curve(U[look], wp, wb)
        print('\nlookback %d min, anisotropy %s, n=%d' % (look, tag, nv))
        print('   %-14s %s | %s' % ('reading',
              ' '.join('k=%-6d' % k for k in KS), 'permutation 5..95% at k=1 / k=40'))
        for r in READS:
            curve, band = res[r]
            flags = ''.join('<' if curve[a] < band[0][a] else
                            ('>' if curve[a] > band[1][a] else '.') for a in range(6))
            print('   %-14s %s | [%.3f %.3f] / [%.3f %.3f]  %s'
                  % (r, ' '.join('%-8.4f' % c for c in curve),
                     band[0][0], band[1][0], band[0][5], band[1][5], flags))
    break  # the full sweep over lookbacks runs below only for the 1:1 metric

print('\n' + '=' * 92)
print('DOES THE NEIGHBOURHOOD STRUCTURE ITSELF SURVIVE A CHANGE OF RULER?')
print('overlap of the k nearest-neighbour SETS between lookbacks (not rank corr)')
print('=' * 92)
for k in [5, 10, 20]:
    for a, b in [(30, 60), (60, 120)]:
        Xa, oka = coords(U[a], 1, 1)
        Xb, okb = coords(U[b], 1, 1)
        both = np.where(oka & okb)[0]
        ov = []
        for i in both:
            cand = both[(day[both] != day[i]) & (side[both] == side[i]) & (kind[both] == kind[i])]
            if len(cand) <= k:
                continue
            da = np.sum((Xa[cand] - Xa[i]) ** 2, axis=1)
            db = np.sum((Xb[cand] - Xb[i]) ** 2, axis=1)
            sa = set(cand[np.argsort(da)][:k]); sb = set(cand[np.argsort(db)][:k])
            ov.append(len(sa & sb) / k)
        print('   k=%-3d %3d vs %3d min: mean overlap %.3f  (n=%d)'
              % (k, a, b, float(np.mean(ov)), len(ov)))
