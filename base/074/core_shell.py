"""074: prefix-only structural ablation; no future, Volume or trading labels.
Run from repository root: python -B base/074/core_shell.py prepare|fit|validate
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'work/074'
SEEDS = [20071, 20517, 75248, 6727]
EPOCHS = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]
MINUTE = 60_000_000_000

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load(inst):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{inst}_prefix.npz'
    if not path.exists():
        files = sorted((ROOT / f'data/field/{inst}/cells').glob('*/passports.parquet'))
        ts0 = np.unique(np.concatenate([pd.read_parquet(f, columns=['t0_ts_ns']).t0_ts_ns.to_numpy() for f in files]))
        m = ROOT / f'data/market/{inst}'
        ts = np.load(m/'close_ts_utc_ns.npy', mmap_mode='r')
        wanted = ts0[:, None] + np.arange(8)[None, :]*MINUTE
        pos = np.searchsorted(ts, wanted)
        safe = np.minimum(pos, len(ts)-1)
        hit = (pos < len(ts)) & (ts[safe] == wanted)
        x = np.stack([np.where(hit, np.load(m/f'{f}.npy', mmap_mode='r')[safe], np.nan)
                      for f in ['open', 'high', 'low', 'close']], axis=2)
        np.savez_compressed(path, t0=ts0, x=x)
    z = np.load(path)
    t0, x = z['t0'], z['x']
    years = t0.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
    return t0, x.reshape(len(t0), 32), years

def pairs():
    a, b = np.triu_indices(32, 1)
    # Exactly the pair vocabulary used by 073; its forced H>L is audited separately.
    intra = {(0,3), (1,0), (1,3), (1,2), (2,0), (2,3)}
    keep = np.array([i//4 != j//4 or (i%4,j%4) in intra for i,j in zip(a,b)])
    return a[keep], b[keep]

def codes(x, a, b):
    v = np.sign(x[:, a]-x[:, b]).astype('float32')
    return v

def name(a, b, v):
    fields = 'OHLC'
    return f'{fields[a%4]}[{a//4}] {">" if v == 1 else "<"} {fields[b%4]}[{b//4}]'

def extract(x, years, seed, floor=200):
    a,b = pairs()
    q = codes(x[seed:seed+1], a,b)[0]
    use = np.isfinite(q) & (q != 0)
    a,b,q = a[use], b[use], q[use]
    c = codes(x,a,b)
    match = c == q
    known = np.isfinite(c)
    tr = years%2 == 0
    freq = match[tr].sum(0)/known[tr].sum(0)
    order = np.argsort(freq, kind='stable')
    alive = tr.copy()
    Q = []
    for j in order:
        nex = alive & match[:,j]
        if nex.sum() < floor:
            break
        alive = nex
        Q.append(int(j))
    bb = np.flatnonzero((match[alive].sum(0)/known[alive].sum(0) >= .90) & (known[alive].sum(0)>alive.sum()//2))
    return a,b,q,match,known,np.array(Q),bb,alive

def closure(edges):
    r = np.zeros((32,32), np.int8)
    for k in range(8):
        o,h,l,c = np.arange(4)+4*k
        for i,j in [(l,o),(l,c),(o,h),(c,h),(l,h)]:
            r[i,j] = 1  # weak anatomy: does not invent a strict relation
    for i,j in edges:
        r[i,j] = 2
    for k in range(32):
        reach = (r[:,k,None]>0) & (r[None,k,:]>0)
        strength = np.maximum(r[:,k,None],r[None,k,:])
        r = np.maximum(r, np.where(reach,strength,0))
    return r

def reduce_edges(a,b,q,ids):
    ids = list(map(int,ids))
    edges = {j:(int(b[j]),int(a[j])) if q[j]>0 else (int(a[j]),int(b[j])) for j in ids}
    kept = ids.copy()
    redundant = []
    for j in ids:
        reach = closure([edges[t] for t in kept if t != j])
        if reach[edges[j]] == 2:
            kept.remove(j)
            redundant.append(j)
    return kept, redundant

def prepare():
    t,x,y = load('NQ')
    tr = y%2 == 0
    rows = []
    for seed in SEEDS:
        a,b,q,m,k,Q,B,F = extract(x,y,seed)
        R,D = reduce_edges(a,b,q,B)
        assert np.array_equal(m[:,B].all(1),m[:,R].all(1))
        qr,qd = reduce_edges(a,b,q,Q)
        exact = m[:,B].all(1)
        valid = k[:,B].all(1)
        eras = []
        for lo,hi in EPOCHS:
            f = F & (y>=lo)&(y<=hi)
            eras.append(dict(epoch=[lo,hi], n=int(f.sum()), support=m[f][:,R].mean(0).tolist()))
        row = dict(seed=seed,t0=str(t[seed].astype('datetime64[ns]')),backbone=len(B),basis=len(R),
                   ladder=len(Q),ladder_basis=len(qr),family=int(F.sum()),
                   exact_train=int((exact&tr).sum()),exact_test=int((exact&~tr).sum()),
                   ratio=float(exact[~tr&valid].mean()/exact[tr&valid].mean()),
                   basis_relations=[name(a[j],b[j],q[j]) for j in R],
                   ladder_relations=[name(a[j],b[j],q[j]) for j in qr],epochs=eras,
                   seed_ohlc=x[seed].reshape(8,4).tolist())
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['epochs','basis_relations','seed_ohlc']}, ensure_ascii=False),flush=True)
    dump(OUT/'preparation.json',rows)

if __name__ == '__main__':
    globals()[sys.argv[1]]()
