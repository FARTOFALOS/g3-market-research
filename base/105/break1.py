"""Вопрос 1 (LOG.md): первый пробой ночного диапазона закрытием минуты, 02:00–16:00 ET.
Вилка ±k·u от open следующей минуты; u = медиана (h−l) 30 последних закрытых минут на узнавании."""
import sys, numpy as np, pandas as pd
from tape import load_minutes

KS = [1, 2, 3, 4, 6]
COST = 0.75


def scenes(inst):
    df = load_minutes('2012-12-01', '2026-07-11', inst).reset_index(drop=True)
    o, h, l, c = (df[k].to_numpy() for k in 'ohlc')
    date = df['date'].to_numpy(); mod = df['mod'].to_numpy(); ts = df['ts'].to_numpy()
    u_all = pd.Series(h - l).rolling(30).median().to_numpy()
    starts = {}
    for idx, d in enumerate(date):
        if d not in starts:
            starts[d] = idx
    ends = {}
    for idx in range(len(date) - 1, -1, -1):
        if date[idx] not in ends:
            ends[date[idx]] = idx
    dates = sorted(starts)
    prev_of = {dates[j]: dates[j - 1] for j in range(1, len(dates))}
    out = []
    for D in dates:
        if D < 20130101 or D not in prev_of:
            continue
        P = prev_of[D]
        # ночь: бары календарной даты P с 18:00 и бары D до 02:00
        pi = np.arange(starts[P], ends[P] + 1); pi = pi[mod[pi] >= 1080]
        di = np.arange(starts[D], ends[D] + 1)
        ni = np.concatenate([pi, di[mod[di] < 120]])
        # ночь должна начинаться в 18:00 вчерашней календарной даты
        if len(ni) < 470 or mod[ni[0]] != 1080:
            continue
        H, L = h[ni].max(), l[ni].min()
        hi_pos = ni[np.argmax(h[ni])]; lo_pos = ni[np.argmin(l[ni])]
        night_dir = 1 if hi_pos > lo_pos else -1
        wi = di[(mod[di] >= 120) & (mod[di] < 960)]
        if len(wi) == 0:
            continue
        brk = wi[(c[wi] > H) | (c[wi] < L)]
        rec = dict(date=D, H=H, L=L, width=H - L, night_dir=night_dir, nbars=len(ni))
        if len(brk) == 0:
            rec['status'] = 'no_break'; out.append(rec); continue
        i = brk[0]
        side = 1 if c[i] > H else -1
        rec.update(side=side, mod=mod[i], u=u_all[i], with_night=int(side == night_dir),
                   beyond_u=(c[i] - (H if side > 0 else L)) * side / u_all[i], bar_u=(h[i] - l[i]) / u_all[i])
        lvl = H if side > 0 else L
        pos = hi_pos if side > 0 else lo_pos
        prev60 = np.arange(max(i - 60, 0), i)
        rec.update(run15_u=side * (c[i] - c[i - 15]) / u_all[i],
                   touches60=int(((h[prev60] >= lvl) if side > 0 else (l[prev60] <= lvl)).sum()),
                   age=(ts[i] - ts[pos]) / 6e10, width_u=(H - L) / u_all[i], expand=u_all[i] / u_all[i - 30])
        if i + 1 > ends[D] or mod[i + 1] != mod[i] + 1 or mod[i + 1] >= 960:
            rec['status'] = 'no_entry'; out.append(rec); continue
        e = o[i + 1]; u = u_all[i]
        rec.update(status='ok', entry=e)
        # путь: непрерывные минуты до 120 и до 15:59
        path = [i + 1]
        j = i + 1
        while len(path) < 120 and j + 1 <= ends[D] and mod[j + 1] == mod[j] + 1 and mod[j + 1] < 960:
            j += 1; path.append(j)
        path = np.array(path)
        full_len = min(120, 960 - mod[i + 1])
        rec['gap_cut'] = int(len(path) < full_len)
        fav = (h[path] - e) if side > 0 else (e - l[path])
        adv = (e - l[path]) if side > 0 else (h[path] - e)
        rec['mfe_u'] = fav.max() / u; rec['mae_u'] = adv.max() / u
        rec['end_u'] = side * (c[path[-1]] - e) / u
        for k in KS:
            a = np.nonzero(fav >= k * u)[0]; b = np.nonzero(adv >= k * u)[0]
            ta = a[0] if len(a) else 10**6; tb = b[0] if len(b) else 10**6
            if ta == tb == 10**6:
                y = 'cens_gap' if rec['gap_cut'] else 'cens'
                pnl = side * (c[path[-1]] - e)
            elif ta == tb:
                y = 'both'; pnl = np.nan
            elif ta < tb:
                y = 'cont'; pnl = k * u
            else:
                y = 'ret'; pnl = -k * u
            rec[f'y{k}'] = y; rec[f'p{k}'] = pnl; rec[f't{k}'] = min(ta, tb)
        out.append(rec)
    return pd.DataFrame(out)


def tlayer(m):
    if m < 510: return 'A 02:00-08:29'
    if m < 570: return 'B 08:30-09:29'
    if m == 570: return 'C 09:30'
    return 'D 09:31-15:59'


def summary(s, label):
    ok = s[s.status == 'ok']
    rows = []
    for k in KS:
        y = ok[f'y{k}']; p = ok[f'p{k}']
        res = y.isin(['cont', 'ret'])
        Y = (y[res] == 'cont').mean()
        both = (y == 'both').sum()
        net_lo = p.fillna(-k * ok.u) - COST; net_hi = p.fillna(k * ok.u) - COST
        netu = (p - COST) / ok.u
        rows.append(dict(layer=label, k=k, n=len(ok), resolved=res.mean().round(3), both=both,
                         Y=round(Y, 3), Y_minus_p0=round(Y - 0.5, 3),
                         net_pt_lo=round(net_lo.mean(), 2), net_pt_hi=round(net_hi.mean(), 2),
                         t_lo=round(net_lo.mean() / net_lo.std() * np.sqrt(len(net_lo)), 2),
                         net_u=round(netu.mean(), 3)))
    return rows


if __name__ == '__main__':
    allrows = []
    for inst in ['NQ', 'ES']:
        s = scenes(inst)
        s.to_csv(f'break1_{inst}.csv', index=False)
        s['ep'] = np.where(s.date >= 20200101, '2020-26', '2013-19')
        st = s.groupby('ep').status.value_counts().unstack(fill_value=0)
        print(inst, 'status by epoch\n', st.to_string())
        ok = s[s.status == 'ok'].copy()
        ok['tl'] = ok['mod'].map(tlayer)
        print(inst, 'first-break time layer counts\n', ok.groupby(['ep', 'tl']).size().unstack(fill_value=0).to_string())
        for ep, g in s.groupby('ep'):
            allrows += summary(g, f'{inst} {ep} all')
            g2 = g[g.status == 'ok'].copy(); g2['tl'] = g2['mod'].map(tlayer)
            for tl, g3 in g2.groupby('tl'):
                allrows += summary(g3.assign(status='ok'), f'{inst} {ep} {tl}')
            for w, g3 in g2.groupby('with_night'):
                allrows += summary(g3.assign(status='ok'), f'{inst} {ep} with_night={w}')
            for sd, g3 in g2.groupby('side'):
                allrows += summary(g3.assign(status='ok'), f'{inst} {ep} side={sd}')
    r = pd.DataFrame(allrows)
    r.to_csv('break1_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(r.to_string(index=False))
