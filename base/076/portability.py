"""Point 4: is the comparable territory and its residual H an artifact of one
ruler, one epoch, or one object?

No synthetic null anywhere in this file. The question is not "is S better than
chance" but "does the same construction, applied to other projections and other
instruments and inside each epoch, still give a dense comparable territory with
real residual variation of the road".

S is built only from what is available at the cursor:
  side, T0 kind, duration           (passport and prefix)
  position = lastC / u_prior        (u_prior: 60 minutes strictly BEFORE the interval)
  bulk     = zone width / u_prior
H is read as bounded shares of the interval, from the same prefix.
Y is not defined, loaded or computed. Volume is not loaded. Field is read-only.
"""
import json, collections, sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
OUT = Path(__file__).parent
MIN = 60_000_000_000
KMAX = 20


def load_instrument(inst):
    m = ROOT / 'data/market' / inst
    A = {n: np.load(m / (n + '.npy'), mmap_mode='r')
         for n in ['open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id']}
    with np.load(m / 'sessions.npz') as z:
        sess = {n: z[n].copy() for n in z.files}
    return A, sess, {int(v): i for i, v in enumerate(sess['session_id'])}


def states(inst, tf, A, sess, look):
    cell = ROOT / f'data/field/{inst}/cells/tf_{tf:04d}'
    if not (cell / 'passports.parquet').exists():
        return []
    cols = ['riz_id', 'zone_bottom', 'zone_top', 't0_spine_pos', 't0_exit_side',
            't0_kind']
    rows = pq.read_table(cell / 'passports.parquet', columns=cols).to_pylist()
    rows.sort(key=lambda r: (r['t0_spine_pos'], r['riz_id']))
    out = []
    ts_all = A['close_ts_utc_ns']
    for r in rows:
        t0 = int(r['t0_spine_pos']); p = t0 + 2
        if p >= len(A['close']):
            continue
        si = sess['lookup'].get(int(A['session_id'][t0]))
        if si is None:
            continue
        anchor = int(sess['session_open_utc_ns'][si])
        t = int(ts_all[t0])
        key = ((t - anchor) // MIN - 1) // tf
        begin = anchor + key * tf * MIN
        start = int(np.searchsorted(ts_all[:p + 1], begin, side='right'))
        ts = np.asarray(ts_all[start:p + 1])
        if len(ts) < 3 or int(ts[0]) != begin + MIN or np.any(np.diff(ts) != MIN):
            continue
        # ruler strictly before the interval
        a = start - look
        if a < 0:
            continue
        tsp = np.asarray(ts_all[a:start])
        if len(tsp) < look or np.any(np.diff(tsp) != MIN):
            continue
        hp = np.asarray(A['high'][a:start], float); lp = np.asarray(A['low'][a:start], float)
        if not np.isfinite(hp).all() or not np.isfinite(lp).all():
            continue
        u = float(np.median(hp - lp))
        if u <= 0:
            continue
        raw = np.column_stack([np.asarray(A[n][start:p + 1], float)
                               for n in ['open', 'high', 'low', 'close']])
        if not np.isfinite(raw).all():
            continue
        w = r['zone_top'] - r['zone_bottom']
        if w <= 0:
            continue
        side = r['t0_exit_side']
        ex = r['zone_top'] if side == 'north' else r['zone_bottom']
        b = raw - ex if side == 'north' else (ex - raw)[:, [0, 2, 1, 3]]
        o, h, l, c = b.T
        e = (l <= 0) & (h >= 0); f = (l <= -w) & (h >= -w)
        loc = np.select([c < -w, c == -w, c < 0, c == 0], [-2, -1, 0, 1], default=2)
        reads = list(zip(e.astype(int), f.astype(int), loc.astype(int)))
        runs = sum(1 for i, v in enumerate(reads) if i == 0 or v != reads[i - 1])
        d = float(len(b))
        out.append(dict(side=side, kind=r['t0_kind'], dur=d,
                        pos=float(c[-1]) / u, bulk=float(w) / u,
                        day=int(A['session_id'][p]),
                        year=int(str(np.datetime64(int(ts[-1]), 'ns'))[:4]),
                        H=[runs / d, int(e.sum()) / d, int(f.sum()) / d,
                           int((loc == 0).sum()) / d]))
    return out


READS = ['runs_rate', 'exit_share', 'far_share', 'inside_share']


def territory(st, label):
    """density of the comparable territory and the residual H inside it."""
    if len(st) < 80:
        return None
    n = len(st)
    pos = np.array([s['pos'] for s in st]); bulk = np.array([s['bulk'] for s in st])
    dur = np.array([s['dur'] for s in st]); day = np.array([s['day'] for s in st])
    side = np.array([s['side'] for s in st]); kind = np.array([s['kind'] for s in st])
    H = np.array([s['H'] for s in st])
    X = np.zeros((n, 2))
    for j, v in enumerate([pos, bulk]):
        o = np.argsort(v, kind='stable'); r = np.empty(n); r[o] = np.arange(n)
        X[:, j] = r / max(n - 1, 1)
    d20, resid, used = [], [], 0
    allidx = np.arange(n)
    for i in range(n):
        c = allidx[(day != day[i]) & (side == side[i]) & (kind == kind[i])
                   & (np.abs(dur - dur[i]) <= 3)]
        if len(c) < KMAX:
            continue
        o = c[np.argsort(np.sum((X[c] - X[i]) ** 2, axis=1), kind='stable')]
        j20 = o[KMAX - 1]
        d20.append(abs(pos[i] - pos[j20]))
        resid.append(np.abs(H[i] - H[j20]))
        used += 1
    if used < 40:
        return None
    resid = np.array(resid)
    sd = H.std(axis=0)
    return dict(label=label, n=n, used=used, d20=float(np.median(d20)),
                resid=resid.mean(axis=0), sd=sd,
                ratio=resid.mean(axis=0) / np.where(sd > 0, sd, np.nan))


def show(rows, title):
    print('\n' + '=' * 96)
    print(title)
    print('=' * 96)
    print('%-26s %6s %6s %8s | %s' % ('', 'n', 'used', 'd20 pos',
          '  '.join('%-14s' % r for r in READS)))
    print('%-26s %6s %6s %8s | %s' % ('', '', '', '',
          'mean |dH| at k=20  (as share of that column sd)'))
    for r in rows:
        if r is None:
            continue
        cells = '  '.join('%.4f (%.2f)' % (a, b) for a, b in zip(r['resid'], r['ratio']))
        print('%-26s %6d %6d %8.3f | %s' % (r['label'], r['n'], r['used'], r['d20'], cells))


CACHE = {}
def inst_data(inst):
    if inst not in CACHE:
        A, sess, lookup = load_instrument(inst)
        sess['lookup'] = lookup
        CACHE[inst] = (A, sess)
    return CACHE[inst]


if __name__ == '__main__':
    # --- different objects: other TF projections of NQ, and other instruments
    rows = []
    for inst, tf in [('NQ', 54), ('NQ', 30), ('NQ', 90), ('NQ', 120),
                     ('ES', 54), ('YM', 54)]:
        try:
            A, sess = inst_data(inst)
            st = states(inst, tf, A, sess, 60)
            print('%s tf%-4d -> %d usable states' % (inst, tf, len(st)), flush=True)
            rows.append(territory(st, '%s tf%d' % (inst, tf)))
        except Exception as ex:
            print('%s tf%d failed: %s' % (inst, tf, ex))
    show(rows, 'SAME CONSTRUCTION ON OTHER PROJECTIONS AND OTHER INSTRUMENTS')

    # --- epochs, on the original object
    A, sess = inst_data('NQ')
    st = states('NQ', 54, A, sess, 60)
    erows = []
    for lo, hi in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        sub = [s for s in st if lo <= s['year'] <= hi]
        erows.append(territory(sub, 'NQ tf54 %d-%d' % (lo, hi)))
    show(erows, 'INSIDE EACH EPOCH (NQ tf54)')

    # --- ruler depth, on the original object
    lrows = []
    for look in [30, 60, 120]:
        sub = states('NQ', 54, A, sess, look)
        lrows.append(territory(sub, 'NQ tf54 ruler %d min' % look))
    show(lrows, 'RULER DEPTH')

    json.dump([{k: (list(map(float, v)) if isinstance(v, np.ndarray) else v)
                for k, v in r.items()} for r in rows + erows + lrows if r],
              open(OUT / 'portability.json', 'w'), indent=1)
    print('\nwritten: portability.json')
