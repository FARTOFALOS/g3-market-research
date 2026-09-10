"""Recheck the fixed 200/2000/5000 floors with identical frozen predicates.
No floor search. Reproduces seed selection and family deduplication in 073.
Run: python -B base/073/transfer_recheck.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'base/074'))
from core_shell import load, extract, dump, sha

def main():
    t,x,y = load('NQ')
    ts = np.load(ROOT/'data/market/NQ/close_ts_utc_ns.npy',mmap_mode='r')
    # Only for exact reproduction of the OLD seed pool, not membership.
    wanted = t[:,None]+np.arange(51)[None,:]*60_000_000_000
    pos = np.searchsorted(ts,wanted)
    full = ((pos<len(ts)) & (ts[np.minimum(pos,len(ts)-1)]==wanted)).all(1)
    tr,te = y%2==0,y%2==1
    scenes = np.random.default_rng(73).choice(np.flatnonzero(tr&full),10,replace=False)
    all_out = []
    for floor in [200,2000,5000]:
        families,states = [],[]
        for seed in scenes:
            a,b,q,m,k,Q,B,F = extract(x,y,int(seed),floor)
            if len(B)<5 or any((F&prev).sum()/(F|prev).sum()>.5 for prev in families):
                continue
            families.append(F)
            pred = m[:,B].all(1)
            known = k[:,B].all(1)
            nA,nB = int((tr&known).sum()),int((te&known).sum())
            eA,eB = int((pred&tr).sum()),int((pred&te).sum())
            sA,sB = eA/nA,eB/nB
            inter = int((F&pred).sum())
            row = dict(floor=floor,seed=int(seed),relations=len(B),
                       retrieval_family_size_A=int(F.sum()),predicate_extent_A=eA,predicate_extent_B=eB,
                       known_A=nA,known_B=nB,unknown_A=int(tr.sum()-nA),unknown_B=int(te.sum()-nB),
                       sA=sA,sB=sB,transfer=sB/sA,
                       old_mixed_ratio=(eB/te.sum())/(F.sum()/tr.sum()),
                       family_predicate_intersection_A=inter,
                       family_only_A=int(F.sum())-inter,predicate_only_A=eA-inter,
                       family_predicate_jaccard_A=inter/int((F|pred&tr).sum()),
                       frozen_predicate=[dict(a=int(a[j]),b=int(b[j]),v=int(q[j])) for j in B])
            states.append(row)
        coverage = np.zeros(len(x),bool)
        for s in states:
            atoms = s['frozen_predicate']
            aa=np.array([r['a'] for r in atoms]);bb=np.array([r['b'] for r in atoms]);qq=np.array([r['v'] for r in atoms])
            coverage |= (np.sign(x[:,aa]-x[:,bb])==qq).all(1)
        summary = dict(floor=floor,states=len(states),coverage_B=float(coverage[te].mean()),
                       median_transfer=float(np.median([s['transfer'] for s in states])),
                       range_transfer=[float(min(s['transfer'] for s in states)),float(max(s['transfer'] for s in states))],
                       median_old_mixed_ratio=float(np.median([s['old_mixed_ratio'] for s in states])),
                       median_predicate_to_family_A=float(np.median([s['predicate_extent_A']/s['retrieval_family_size_A'] for s in states])))
        print(summary,flush=True)
        all_out.append(dict(summary=summary,states=states))
    dump(Path(__file__).with_name('transfer_recheck.json'),dict(
        operator='same-frozen-predicate-prevalence/1',train='even calendar years',test='odd calendar years',
        population=len(t),train_n=int(tr.sum()),test_n=int(te.sum()),seed_pool='original 073 full-51-minute pool; membership uses only 0..7',
        seeds=scenes.tolist(),source_data=__import__('json').loads((ROOT/'SOURCE_DATA.json').read_text())['instruments']['NQ'],
        input_sha256=sha(ROOT/'work/074/NQ_prefix.npz'),results=all_out))

if __name__=='__main__': main()
