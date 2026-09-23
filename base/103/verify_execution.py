"""Discriminating tests of closed-bar exits, next-open fills and unknown gaps."""
import json
import numpy as np
from replay import position
from census import ROOT,MINUTE

def tape(O,H,L,C):
    return {'ts':np.arange(1,len(C)+1,dtype=np.int64)*MINUTE,
            **{n:np.array(v,dtype=float) for n,v in zip(('open','high','low','close'),(O,H,L,C))}}

checks=[]
# A wick crosses both levels. Neither triggers before a closed signal.
a=tape([100,101,103,107],[120,121,110,108],[80,81,99,106],[101,102,106,107])
r=position(a,0,4*MINUTE,1,105,95,True)
assert r['event_pos']==2 and r['exit_pos']==3 and r['exit_price']==107 and r['gross']==7
checks.append('intrabar extremes do not trigger; target close fills next open')
# Zone exit followed by recovery: same initial position, same target.
a=tape([100,101,90,106],[102,102,107,110],[99,92,89,105],[101,93,106,108])
x=position(a,0,4*MINUTE,1,105,95,True)
y=position(a,0,4*MINUTE,1,105,95,False)
assert x['event_pos']==1 and x['exit_price']==90 and x['net']==-11
assert y['exit_price']==106 and y['net']==5
assert x['net']-y['net']==-16
checks.append('gap price honored; recovered winner remains in paired comparison')
# Mirroring preserves arithmetic.
b={**a,'open':-a['open'],'high':-a['low'],'low':-a['high'],'close':-a['close']}
z=position(b,0,4*MINUTE,-1,-105,-95,True)
assert z['net']==x['net'] and z['reason']==x['reason']
checks.append('long/short symmetry')
# Missing minutes invalidate known execution rather than giving it a zero.
c={**a,'ts':np.array([1,2,4,5],dtype=np.int64)*MINUTE}
u=position(c,0,5*MINUTE,1,105,95,True)
assert u['status']=='unknown' and u['reason']=='gap_at_exit' and u['gross'] is None
checks.append('missing execution retained unknown')
r=position(a,0,2*MINUTE,1,105,95,True)
assert r['reason']=='session_close' and r['exit_price']==93
checks.append('announced session close precedes later next-open exit')
r=position(a,0,5*MINUTE,1,105,95,True)
assert r['status']=='known' and r['exit_price']==90
checks.append('missing later session close cannot invalidate an already completed trade')
(ROOT/'base/103/execution_verification.json').write_text(json.dumps({'status':'pass','checks':checks,
    'scope':'synthetic execution counterexamples; real-trade audit still required'},indent=2),encoding='utf-8')
print('PASS',len(checks),'execution checks')
