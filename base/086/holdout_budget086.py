"""086 — resolution budget of the selected region on the NQ holdout (FREEZE_086 §7: "не подтверждено" -> budget names the distinguishable size).

Added at card time, after the holdout was opened: holdout() in fork086.py did not report the region budget.
Frozen §5 formula (fork086.budget_unit), NQ evaluation clustering and unknown-share proxy from budget086.json.
Uses geometry only (p0, d, s, t0_day); no fork outcome is read. Cannot change any class.

    python -B holdout_budget086.py
"""
from __future__ import annotations
import json
import numpy as np
import fork086 as F


def main():
    B = json.loads((F.OUT / 'budget086.json').read_text(encoding='utf-8'))
    cells = json.loads((F.OUT / 'cells086.json').read_text(encoding='utf-8'))
    sp = F.OUT / 'selected086.json'
    S = json.loads(sp.read_text(encoding='utf-8'))
    R = B['clustering_084']['NQ evaluation']
    g = F.geometry('NQ_holdout')
    ok = np.isfinite(g.ttc.to_numpy())
    cid = np.full(len(g), -1)
    cid[ok] = F.cell_ids(g[ok], cells['medians'])
    res = {'freeze_sha256': F.freeze(), 'selected_sha256': F.sha(sp), 'budget_sha256': F.sha(F.OUT / 'budget086.json'),
           'clustering_084': R, 'regions': {},
           'note': 'computed after the holdout was opened; frozen FREEZE_086 §5 formula; geometry only, no fork outcomes'}
    for rname, cl in S['regions'].items():
        sub = g[np.isin(cid, cl)]
        u = F.budget_unit(sub, R['rho'], R['unknown_share_084'], F.COST['NQ'])
        m = sub.groupby('t0_day').size()
        u['K'] = round(float((m.to_numpy().astype(float) ** 2).sum() / len(sub)), 2)
        u['largest_day'] = {'day': str(np.datetime64(int(m.idxmax()), 'D')), 'films': int(m.max()),
                            'share': round(float(m.max() / len(sub)), 4)}
        res['regions'][rname] = {'cells': [F.cell_label(c) for c in cl], **u}
        print(rname, json.dumps(res['regions'][rname], ensure_ascii=False))
    (F.OUT / 'holdout_budget086.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
