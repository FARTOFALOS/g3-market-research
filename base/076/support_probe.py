"""Execute support_probe_protocol.md; source field is read-only; no Y or Volume."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
OUT = Path(__file__).parent
MINUTE = 60_000_000_000

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stamp(ns):
    return str(np.datetime64(int(ns), 'ns'))[:16]

def describe(bars, width, side, kind):
    # Bars are already oriented (O,H,L,C), with H>=L after reflection.
    o,h,l,c = bars.T
    e = (l <= 0) & (h >= 0)
    f = (l <= -width) & (h >= -width)
    location = np.select([c < -width, c == -width, c < 0, c == 0],
                         [-2, -1, 0, 1], default=2).astype(int)
    readings = [(int(a), int(b), int(d)) for a,b,d in zip(e,f,location)]
    runs = [(i, v) for i,v in enumerate(readings) if i == 0 or v != readings[i-1]]
    g0 = (side, kind, float(width), float(o[0]), float(l.min()), float(h.max()),
          *[float(v) for v in bars[-1]], len(bars))
    counts = (int(e.sum()), int(f.sum()), *[int((location==k).sum()) for k in [-2,-1,0,1,2]])
    return dict(g0=g0, g1=g0+counts, h_order=tuple(v for _,v in runs),
                h_timed=tuple(runs), readings=readings, counts=counts)

def synthetic(path):
    o=np.asarray(path[:-1],float)-110; c=np.asarray(path[1:],float)-110
    return np.column_stack([o,np.maximum(o,c),np.minimum(o,c),c])

a = describe(synthetic([105,111,105,99,105,106]),10,'north','synthetic')
b = describe(synthetic([105,99,105,111,105,106]),10,'north','synthetic')
assert a['g0']==b['g0'] and a['g1']==b['g1'] and a['h_order']!=b['h_order']

m=ROOT/'data/market/NQ'; cell=ROOT/'data/field/NQ/cells/tf_0054'
names=['open','high','low','close','close_ts_utc_ns','session_id']
arrays={n:np.load(m/(n+'.npy'),mmap_mode='r') for n in names}
with np.load(m/'sessions.npz') as z: sessions={n:z[n].copy() for n in z.files}
session_lookup={int(v):i for i,v in enumerate(sessions['session_id'])}
columns=['riz_id','corpus_id','zone_bottom','zone_top','t0_spine_pos','t0_ts_ns',
         't0_exit_side','t0_kind','t0_span_count']
rows=pq.read_table(cell/'passports.parquet',columns=columns).to_pylist()
rows.sort(key=lambda r:(r['t0_spine_pos'],r['riz_id']))
market_manifest=json.loads((m/'manifest.json').read_text())
cell_manifest=json.loads((cell/'manifest.json').read_text())
source=json.loads((ROOT/'SOURCE_DATA.json').read_text())
assert source['instruments']['NQ']['corpus_id']==market_manifest['corpus_id']==cell_manifest['market_corpus_id']
assert cell_manifest['status']=='complete'
for name in ['passports','events']:
    assert digest(cell/(name+'.parquet'))==cell_manifest['output_sha256'][name+'.parquet']

def extract(row, source_arrays):
    t0=int(row['t0_spine_pos']); p=t0+2
    if p>=len(source_arrays['close']): return None,'cursor_not_observed'
    si=session_lookup[int(source_arrays['session_id'][t0])]
    anchor=int(sessions['session_open_utc_ns'][si]); t=int(source_arrays['close_ts_utc_ns'][t0])
    key=((t-anchor)//MINUTE-1)//54
    begin_ns=anchor+key*54*MINUTE
    start=int(np.searchsorted(source_arrays['close_ts_utc_ns'][:p+1],begin_ns,side='right'))
    ts=np.asarray(source_arrays['close_ts_utc_ns'][start:p+1])
    if int(ts[0])!=begin_ns+MINUTE or np.any(np.diff(ts)!=MINUTE):
        return None,'native_prefix_gap'
    raw=np.column_stack([source_arrays[n][start:p+1] for n in ['open','high','low','close']])
    if not np.isfinite(raw).all() or np.any(raw[:,1]<raw[:,2]) or np.any(raw[:,0]>raw[:,1]) or np.any(raw[:,0]<raw[:,2]) or np.any(raw[:,3]>raw[:,1]) or np.any(raw[:,3]<raw[:,2]):
        return None,'invalid_ohlc'
    side=row['t0_exit_side']; exit_price=row['zone_top'] if side=='north' else row['zone_bottom']
    width=row['zone_top']-row['zone_bottom']
    if width<=0: return None,'invalid_zone'
    if side=='north': bars=raw-exit_price
    else: bars=(exit_price-raw)[:,[0,2,1,3]]
    state=describe(bars,width,side,row['t0_kind'])
    state.update(riz_id=row['riz_id'],scene=f'NQ:{t0}:{p}',t0=t0,p=p,start=start,
                 p_utc=stamp(ts[-1]),t0_utc=stamp(source_arrays['close_ts_utc_ns'][t0]),
                 session_id=int(source_arrays['session_id'][p]),zone=[row['zone_bottom'],row['zone_top']],
                 raw_ohlc=raw.tolist(),utc_minutes=[stamp(t) for t in ts])
    return state,None

states=[]; missing=[]; checked=0
for row in rows:
    state,reason=extract(row,arrays)
    if reason:
        missing.append(dict(riz_id=row['riz_id'],t0=row['t0_spine_pos'],reason=reason)); continue
    cut={n:v[:state['p']+1] for n,v in arrays.items()}
    assert extract(row,cut)==(state,None)
    checked+=1; states.append(state)

support={}
for name in ['g0','g1']:
    groups=defaultdict(list)
    for state in states: groups[state[name]].append(state)
    report=dict(groups=len(groups),repeated_groups=0,repeated_anchors=0,
                cross_scene_groups=0,h_order_contrast_groups=0,h_timed_contrast_groups=0)
    contrast_states=[]
    for key, members in groups.items():
        if len(members)<2: continue
        report['repeated_groups']+=1; report['repeated_anchors']+=len(members)
        if len({x['scene'] for x in members})<2: continue
        report['cross_scene_groups']+=1
        for h in ['h_order','h_timed']:
            if len({x[h] for x in members})>1:
                report[h+'_contrast_groups']+=1
                if h=='h_order': contrast_states.extend(members)
    report['order_support_anchors']=len(contrast_states)
    report['order_support_scenes']=len({x['scene'] for x in contrast_states})
    report['order_support_session_days']=len({x['session_id'] for x in contrast_states})
    support[name]=report

# The first three distinct H_order readings among 2025 prefixes, then the first
# repeated H_order from a different scene, are illustrations only. No Y or G
# distance participates in selecting them.
examples=[]; seen=set()
for state in states:
    if not state['p_utc'].startswith('2025-'): continue
    if state['h_order'] not in seen:
        examples.append(state); seen.add(state['h_order'])
    if len(examples)==3: break
repeat=None
for state in states:
    if state['h_order']==examples[0]['h_order'] and state['scene']!=examples[0]['scene']:
        repeat=state; break
if repeat: examples.append(repeat)

summary=dict(protocol_sha256=digest(OUT/'support_probe_protocol.md'),
             code_sha256=digest(Path(__file__)),corpus_id=market_manifest['corpus_id'],
             cell_passport_sha256=cell_manifest['output_sha256']['passports.parquet'],
             anchor_rows=len(rows),eligible_anchors=len(states),
             physical_scenes=len({s['scene'] for s in states}),
             session_days=len({s['session_id'] for s in states}),
             first_p=states[0]['p_utc'],last_p=states[-1]['p_utc'],
             unresolved=Counter(x['reason'] for x in missing),
             distinct_h_order=len({s['h_order'] for s in states}),
             distinct_h_timed=len({s['h_timed'] for s in states}),
             checked_truncated_prefixes=checked,synthetic_equal_geometry_contrast=True,
             support=support,example_riz_ids=[s['riz_id'] for s in examples],
             future_target_evaluated=False,volume_loaded=False,
             exposure_note='Each state excludes its own post-p prices. The union of overlapping prefixes may expose prices later than another state cursor. This is not an independent forecast validation.')
result=dict(summary=summary,states=states,unresolved=missing,examples=examples)
(OUT/'support_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
for state in examples:
    print('EXAMPLE',state['riz_id'],state['p_utc'],'zone',state['zone'])
    print('G0',state['g0'],'COUNTS',state['counts'])
    print('H_TIMED',state['h_timed'])
    print('TAIL',list(zip(state['utc_minutes'][-8:],state['raw_ohlc'][-8:])))
