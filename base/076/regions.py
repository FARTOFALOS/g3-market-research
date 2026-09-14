"""Where in S is there both enough neighbour density and real residual H?

Fixes the null: the real comparison pairs are admissible (same side, same kind,
|d duration| <= 3 min, different session day). The null must respect exactly the
same admissibility and destroy ONLY proximity in S. So instead of permuting H,
the null draws a RANDOM ADMISSIBLE PARTNER for each state. Anything left is
attributable to S-proximity and to nothing else.

Then: how far is the k-th neighbour in different parts of S, where is H nearly
determined by S, and where do different H genuinely coexist.

No Y, no Volume, no model. Field read-only.
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


def pu(st, k=60):
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


U = np.array([pu(s['start']) for s in S])
W = np.array([s['g0'][2] for s in S])
LC = np.array([s['g0'][9] for s in S])
DUR = np.array([s['g0'][10] for s in S], float)
day = np.array([s['session_id'] for s in S])
sd = np.array([s['g0'][0] for s in S])
kd = np.array([s['g0'][1] for s in S])


def hr(s):
    e, f, l_2, l_1, l0, l1, l2 = s['counts']
    d = float(s['g0'][10])
    return [len(s['h_order']) / d, e / d, f / d, l0 / d]


HM = np.array([hr(s) for s in S])
READS = ['runs_rate', 'exit_share', 'far_share', 'inside_share']

ok = np.isfinite(U)
idx = np.where(ok)[0]
pos_raw = np.where(ok, LC / U, np.nan)
bulk_raw = np.where(ok, W / U, np.nan)
X = np.zeros((N, 2))
for j, v in enumerate([pos_raw, bulk_raw]):
    o = np.argsort(v[idx], kind='stable')
    r = np.empty(len(idx)); r[o] = np.arange(len(idx))
    X[idx, j] = r / (len(idx) - 1)

# admissible pools, identical for the observed comparison and for the null
POOL, ORD = {}, {}
KMAX = 20
for i in idx:
    c = idx[(day[idx] != day[i]) & (sd[idx] == sd[i]) & (kd[idx] == kd[i])
            & (np.abs(DUR[idx] - DUR[i]) <= 3)]
    if len(c) >= KMAX:
        POOL[i] = c
        ORD[i] = c[np.argsort(np.sum((X[c] - X[i]) ** 2, axis=1), kind='stable')]
V = sorted(POOL)
print('states with an admissible pool of >= %d: %d' % (KMAX, len(V)))
print('pool size: median %d, p5 %d, p95 %d'
      % (np.median([len(POOL[i]) for i in V]),
         np.percentile([len(POOL[i]) for i in V], 5),
         np.percentile([len(POOL[i]) for i in V], 95)))

KS = [1, 2, 5, 10, 20]
print('\n' + '=' * 90)
print('NULL REBUILT: random ADMISSIBLE partner, same side/kind/duration window')
print('only proximity in S is destroyed; the matching constraint is preserved')
print('=' * 90)
print('%-14s %s | %s' % ('reading', ' '.join('k=%-7d' % k for k in KS),
                         'admissible null 5..95%'))
for j, r in enumerate(READS):
    v = HM[:, j]
    curve = [float(np.mean([abs(v[i] - v[ORD[i][k - 1]]) for i in V])) for k in KS]
    null = np.empty(400)
    for t in range(400):
        null[t] = np.mean([abs(v[i] - v[POOL[i][RNG.integers(len(POOL[i]))]])
                           for i in V])
    p5, p95 = np.percentile(null, [5, 95])
    fl = ''.join('<' if c < p5 else ('>' if c > p95 else '.') for c in curve)
    print('%-14s %s | [%.4f %.4f]  %s'
          % (r, ' '.join('%-9.4f' % c for c in curve), p5, p95, fl))

print('\n' + '=' * 90)
print('HOW FAR IS THE k-th NEIGHBOUR, IN THE RAW UNITS, ACROSS S')
print('=' * 90)
print('distance reported as |d position| and |d bulk| in units of u_prior')
for k in [5, 10, 20]:
    dp = [abs(pos_raw[i] - pos_raw[ORD[i][k - 1]]) for i in V]
    db = [abs(bulk_raw[i] - bulk_raw[ORD[i][k - 1]]) for i in V]
    print('   k=%-3d  |d pos| p50 %.3f p90 %.3f max %.2f   |d bulk| p50 %.3f p90 %.3f max %.2f'
          % (k, np.median(dp), np.percentile(dp, 90), max(dp),
             np.median(db), np.percentile(db, 90), max(db)))

print('\n' + '=' * 90)
print('BY REGION OF S: density, how far the 20th neighbour is, and residual H')
print('=' * 90)
G = 3
cell = {}
for i in V:
    a = min(int(X[i, 0] * G), G - 1)
    b = min(int(X[i, 1] * G), G - 1)
    cell.setdefault((a, b), []).append(i)
qp = np.nanpercentile(pos_raw[idx], [100 / 3, 200 / 3])
qb = np.nanpercentile(bulk_raw[idx], [100 / 3, 200 / 3])
print('position terciles (lastC/u): < %.2f, %.2f..%.2f, > %.2f'
      % (qp[0], qp[0], qp[1], qp[1]))
print('bulk     terciles (w/u)    : < %.2f, %.2f..%.2f, > %.2f\n'
      % (qb[0], qb[0], qb[1], qb[1]))
hdr = '%-16s %5s %9s |' % ('region (pos,bulk)', 'n', 'd20 pos')
for r in READS:
    hdr += ' %-18s' % r
print(hdr)
print('%-16s %5s %9s | %s' % ('', '', '', 'obs |dH| at k=20 / admissible null'))
rows = []
for a in range(G):
    for b in range(G):
        mem = cell.get((a, b), [])
        if len(mem) < 15:
            continue
        d20 = np.median([abs(pos_raw[i] - pos_raw[ORD[i][19]]) for i in mem])
        line = '%-16s %5d %9.3f |' % ('p%d b%d' % (a, b), len(mem), d20)
        row = dict(region='p%d b%d' % (a, b), n=len(mem), d20=float(d20))
        for j, r in enumerate(READS):
            v = HM[:, j]
            obs = float(np.mean([abs(v[i] - v[ORD[i][19]]) for i in mem]))
            nl = np.mean([[abs(v[i] - v[POOL[i][RNG.integers(len(POOL[i]))]])
                           for i in mem] for _ in range(200)])
            line += ' %6.4f/%-6.4f %s' % (obs, nl, '<' if obs < nl * 0.9 else
                                          ('~' if obs < nl * 1.1 else '>'))
            row[r] = dict(obs=obs, null=float(nl))
        print(line)
        rows.append(row)

json.dump(rows, open(OUT / 'regions.json', 'w'), indent=1, default=str)
print('\nlegend: "<" neighbours clearly more alike than an admissible random')
print('partner (S determines H here); "~" no gain from proximity (different H')
print('coexist at comparable S); ">" neighbours less alike than random.')
