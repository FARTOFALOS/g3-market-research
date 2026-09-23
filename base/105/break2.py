"""Вопрос 2: противоположный пробой ночного диапазона после первого. Код вилки — break1."""
import numpy as np, pandas as pd
from tape import load_minutes
from break1 import KS, COST, tlayer, summary

def scenes2(inst):
    df = load_minutes('2012-12-01', '2026-07-11', inst).reset_index(drop=True)
    o, h, l, c = (df[k].to_numpy() for k in 'ohlc')
    date = df['date'].to_numpy(); mod = df['mod'].to_numpy()
    u_all = pd.Series(h - l).rolling(30).median().to_numpy()
    first = pd.read_csv(f'break1_{inst}.csv'); first = first[first.status == 'ok']
    idx_of = {}
    for i, d in enumerate(date):
        idx_of.setdefault(d, []).append(i)
    out = []
    for _, r in first.iterrows():
        D = int(r.date); di = np.array(idx_of[D]); H, L = r.H, r.L; s1 = int(r.side)
        wi = di[(mod[di] > r['mod']) & (mod[di] < 960)]
        brk = wi[(c[wi] < L)] if s1 > 0 else wi[(c[wi] > H)]
        rec = dict(date=D, H=H, L=L, first_mod=r['mod'])
        if len(brk) == 0:
            rec['status'] = 'no_opp'; out.append(rec); continue
        i = brk[0]; side = -s1; u = u_all[i]
        rec.update(side=side, mod=mod[i], u=u)
        if i + 1 > di[-1] or mod[i + 1] != mod[i] + 1 or mod[i + 1] >= 960:
            rec['status'] = 'no_entry'; out.append(rec); continue
        e = o[i + 1]; rec.update(status='ok', entry=e)
        path = [i + 1]; j = i + 1
        while len(path) < 120 and j + 1 <= di[-1] and mod[j + 1] == mod[j] + 1 and mod[j + 1] < 960:
            j += 1; path.append(j)
        path = np.array(path)
        fav = (h[path] - e) if side > 0 else (e - l[path]); adv = (e - l[path]) if side > 0 else (h[path] - e)
        rec['mfe_u'] = fav.max() / u; rec['mae_u'] = adv.max() / u
        for k in KS:
            a = np.nonzero(fav >= k * u)[0]; b = np.nonzero(adv >= k * u)[0]
            ta = a[0] if len(a) else 10**6; tb = b[0] if len(b) else 10**6
            if ta == tb == 10**6: y = 'cens'; pnl = side * (c[path[-1]] - e)
            elif ta == tb: y = 'both'; pnl = np.nan
            elif ta < tb: y = 'cont'; pnl = k * u
            else: y = 'ret'; pnl = -k * u
            rec[f'y{k}'] = y; rec[f'p{k}'] = pnl
        out.append(rec)
    return pd.DataFrame(out)

rows = []
for inst in ['NQ', 'ES']:
    s = scenes2(inst); s.to_csv(f'break2_{inst}.csv', index=False)
    s['ep'] = np.where(s.date >= 20200101, '2020-26', '2013-19')
    print(inst, s.groupby('ep').status.value_counts().unstack(fill_value=0).to_string())
    for ep, g in s.groupby('ep'):
        rows += summary(g, f'{inst} {ep} all')
        g2 = g[g.status == 'ok'].copy(); g2['tl'] = g2['mod'].map(tlayer)
        print(inst, ep, g2.tl.value_counts().sort_index().to_dict())
        for tl, g3 in g2.groupby('tl'):
            if len(g3) >= 30: rows += summary(g3, f'{inst} {ep} {tl}')
r = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(r[['layer', 'k', 'n', 'resolved', 'Y', 'Y_minus_p0', 'net_pt_lo', 'net_pt_hi', 'net_u']].to_string(index=False))
