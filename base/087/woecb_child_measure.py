#!/usr/bin/env python3
"""Child layer of frozen WOECB'. Freeze ONLY the positive observable event:
  MB-resume(t): first bar after WOECB' with B_out(t) > MB_reco (body-front advances again).
Two SEPARATE time fields (do not mix levels):
  event_time  = tau of the FIRST MB-resume (observable at its close).
  recog_time  = tau of a CONFIRMED extend: the resume after which a SECOND MB-resume occurs
                (hold / re-advance). A single resume that then returns to b is NOT recognition.
consumed adverse and remaining movement are measured FROM recog_time (and, for contrast, from
event_time). No QUICK/LONG labels. No fixed <=3 window; whole tau curve. Absence of resume is
NOT an event (kept as a separate 'no_resume' bucket, terminal recorded). Dedup physical acts.
sigma = mean(high-low) T0..reco. NQ discovery. WOECB' frozen = joint->M-only+inward close.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0, str(ROOT / 'base/084'))
import race084
TERR = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
CAP = 400
market, kind = race084.load_market('NQ'); high, low, close = market
opn = np.load(ROOT / 'data/market/NQ/open.npy'); last = close.size - 1
films = pd.read_parquet(ROOT / ('work/081a/paths/films_NQ_'+TERR+'.parquet'),
                        columns=['side', 't0_spine_pos', 'exit_boundary'])


def measure(t0, up, e):
    sgn = 1 if up else -1; Eo = e if up else -e
    M = high[t0] if up else -low[t0]; MB = max(sgn*opn[t0], sgn*close[t0]); last_up = None; parent = -1
    for k in range(1, CAP + 1):
        j = t0 + k
        if j > last or kind[j] != 0: return None
        hi = high[j]; lo = low[j]
        if lo <= e <= hi: return None
        Ho = hi if up else -lo
        if Ho < Eo: return None
        Oo = sgn*opn[j]; Co = sgn*close[j]; Bo = max(Oo, Co)
        cur = 'j' if (Ho > M and Bo > MB) else ('m' if Ho > M else ('b' if Bo > MB else None))
        if parent >= 0 and cur == 'm' and Co < Oo and last_up == 'j':
            reco = j; MBr = MB; c0o = Co; sig = (high[t0:j+1] - low[t0:j+1]).mean()
            if sig <= 0: return None
            # walk forward: find resumes (B_out>MBr), track outward extreme, terminal
            outmax = Ho if Ho > -np.inf else -np.inf   # include reco bar outward extreme
            outmax = Ho
            resumes = []                                # (tau, adverse_consumed_sigma, out_at)
            cause = None; jj = reco; nb = 0
            while True:
                jj += 1; nb += 1
                if nb > CAP: cause = 'cap'; break
                if jj > last: cause = 'archive'; break
                if kind[jj] != 0: cause = 'gap'; break
                h2 = high[jj]; l2 = low[jj]
                if l2 <= e <= h2: cause = 'B'; break
                H2 = h2 if up else -l2
                if H2 < Eo: cause = 'X'; break
                O2 = sgn*opn[jj]; C2 = sgn*close[jj]; B2 = max(O2, C2)
                outmax = max(outmax, H2)
                if B2 > MBr:
                    resumes.append((nb, (outmax - c0o) / sig, outmax, C2))
                    MBr = B2                             # advance the resume reference (next resume must beat this)
            final_out = outmax
            row = dict(reco=reco, up=up, sig=float(sig), cause=cause, bars_after=nb,
                       to_b0=(c0o - Eo) / sig, n_resume=len(resumes))
            if resumes:
                t1, adv1, o1, c1 = resumes[0]
                row.update(event_tau=t1, adv_at_event=adv1,
                           rem_out_from_event=(final_out - o1) / sig,
                           rem_to_b_at_event=(c1 - Eo) / sig)
                if len(resumes) >= 2:
                    t2, adv2, o2, c2 = resumes[1]
                    row.update(recog_tau=t2, adv_at_recog=adv2,
                               rem_out_from_recog=(final_out - o2) / sig,
                               rem_to_b_at_recog=(c2 - Eo) / sig)
                else:
                    row.update(recog_tau=np.nan)
            else:
                row.update(event_tau=np.nan, recog_tau=np.nan)
            return row
        if parent < 0 and cur == 'j': parent = k
        if cur is not None: last_up = cur
        if Ho > M: M = Ho
        if Bo > MB: MB = Bo
    return None


rows = []
t0a = films.t0_spine_pos.to_numpy(); up_a = (films.side.values == 'north'); ea = films.exit_boundary.to_numpy()
for i in range(len(films)):
    r = measure(int(t0a[i]), bool(up_a[i]), float(ea[i]))
    if r: rows.append(r)
R = pd.DataFrame(rows).drop_duplicates(subset=['reco', 'up'])
n = len(R); has = R.event_tau.notna()
print(f"WOECB' acts (deduped): {n}")
print(f"  MB-resume EVER (event): {has.mean():.3f}   no-resume before terminal: {(~has).mean():.3f}")
print(f"  of resumers, a CONFIRMED second resume (recog): {R.recog_tau.notna().sum()/has.sum():.3f}")
print(f"  no-resume bucket terminal mix: " +
      R[~has].cause.value_counts(normalize=True).round(2).to_dict().__repr__())

print('\ntau curve of FIRST MB-resume (event_time), cumulative share of all acts:')
for t in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
    print(f"   by +{t:>2}: {(R.event_tau <= t).mean():.3f}", end='')
print()

def q(s): return [round(float(np.nanpercentile(s, p)), 2) for p in (25, 50, 75)]
print('\nMAP: time-to-resume -> adverse consumed -> remaining, at EVENT (first resume):')
print(' tau bin | n | adverse consumed@event (sd) | remaining outward (sd) | remaining to b (sd) | reached_b | cens')
for lab, lo, hi in [('1', 1, 2), ('2', 2, 3), ('3-4', 3, 5), ('5-8', 5, 9), ('9+', 9, 10**9)]:
    g = R[(R.event_tau >= lo) & (R.event_tau < hi)]
    if len(g) < 20: continue
    print(f"   {lab:5s} | {len(g):5d} | {q(g.adv_at_event)} | {q(g.rem_out_from_event)} | "
          f"{q(g.rem_to_b_at_event)} | {(g.cause=='B').mean():.2f} | {g.cause.isin(['gap','cap','archive']).mean():.2f}")

print('\nEVENT vs RECOGNITION (confirmed 2nd resume), adverse consumed & remaining, overall:')
ev = R[R.event_tau.notna()]; rc = R[R.recog_tau.notna()]
print(f"  EVENT (n={len(ev)}): tau {q(ev.event_tau)}  adverse consumed {q(ev.adv_at_event)}  "
      f"remaining outward {q(ev.rem_out_from_event)}  remaining to b {q(ev.rem_to_b_at_event)}")
print(f"  RECOG (n={len(rc)}): tau {q(rc.recog_tau)}  adverse consumed {q(rc.adv_at_recog)}  "
      f"remaining outward {q(rc.rem_out_from_recog)}  remaining to b {q(rc.rem_to_b_at_recog)}")
print(f"\n  single-resume-then-no-confirm (event but no recog): {(has & R.recog_tau.isna()).sum()} "
      f"({(has & R.recog_tau.isna()).mean():.3f} of all) -> first resume alone is NOT reliable extend recognition")
