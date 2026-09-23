"""Исторический календарь NQ для окна 02:00–16:00 ET (линия nq-manual), 2006-01 … 2026-07.

Уровни происхождения (записываются в каждую дату):
  RULE+TAPE  — тип дня задан правилом дат (праздники США / соседние дни), время закрытия = модальная минута
               обрыва ленты у ВСЕХ дней этого типа в эту эпоху, и дата обрывается ровно в неё.
               Официального документа по годам нет; секундарный источник (pandas_market_calendars CME_Equity)
               согласуется с 12:00 CT после 2014, для ранних лет расходится (у ленты 10:30 CT) — отмечено.
  TAPE_ONLY  — особые дни, объяснённые событием, но без найденного уведомления CME (указано в note).
  UNKNOWN    — лента не согласуется с правилом или обрывается вне расписания.
Статусы: regular (ожидается 02:00–15:59), short(HH:MM) (ожидается 02:00–HH:MM), closed, unknown.
Полнота объекта: все минуты [02:00, закрытие) наблюдены.
"""
from datetime import date, timedelta
import numpy as np, pandas as pd
from tape import load_minutes


def nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def last_weekday(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def observed(d):
    if d.weekday() == 5: return d - timedelta(days=1)
    if d.weekday() == 6: return d + timedelta(days=1)
    return d


def easter(y):
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; da = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, mo, da)


def day_types(y):
    t = {}
    ny = date(y, 1, 1)
    if ny.weekday() == 6: t[date(y, 1, 2)] = 'NEWYEAR'
    elif ny.weekday() < 5: t[ny] = 'NEWYEAR'
    t[nth_weekday(y, 1, 0, 3)] = 'HOL'          # MLK
    t[nth_weekday(y, 2, 0, 3)] = 'HOL'          # Presidents
    t[easter(y) - timedelta(days=2)] = 'GOODFRI'
    t[last_weekday(y, 5, 0)] = 'HOL'            # Memorial
    if y >= 2022: t[observed(date(y, 6, 19))] = 'HOL'
    j4 = observed(date(y, 7, 4)); t[j4] = 'HOL'
    j3 = date(y, 7, 3)
    if j3.weekday() < 5 and j3 != j4 and date(y, 7, 4).weekday() < 5: t[j3] = 'EVE'
    t[nth_weekday(y, 9, 0, 1)] = 'HOL'          # Labor
    th = nth_weekday(y, 11, 3, 4); t[th] = 'HOL'; t[th + timedelta(days=1)] = 'EVE'   # Thanksgiving, Black Friday
    xm = observed(date(y, 12, 25)); t[xm] = 'XMAS'
    ce = date(y, 12, 24)
    if ce.weekday() < 5 and ce != xm: t[ce] = 'EVE'
    return t


SPECIAL = {  # события, объясняющие ленту; уведомление CME для них в этой линии не найдено → TAPE_ONLY
    date(2007, 1, 2): 'national day of mourning (Ford)',
    date(2012, 10, 29): 'Hurricane Sandy (NYSE closed)', date(2012, 10, 30): 'Hurricane Sandy (NYSE closed)',
    date(2018, 12, 5): 'national day of mourning (Bush)',
    date(2025, 1, 9): 'national day of mourning (Carter)',
    date(2025, 11, 28): 'CME Globex outage until ~08:30 ET (memory; not sourced)',
}


def era(d):
    return 'pre2014' if d < date(2014, 3, 1) else 'post2014'


def build(start='2006-01-01', end='2026-07-11'):
    df = load_minutes(start, end)
    w = df[(df['mod'] >= 120) & (df['mod'] < 960)]
    info = w.groupby('date')['mod'].agg(['min', 'max', 'size'])
    types = {}
    for y in range(int(start[:4]), int(end[:4]) + 1):
        types.update(day_types(y))
    rows = []
    d = pd.Timestamp(start).date()
    while d <= pd.Timestamp(end).date():
        if d.weekday() < 5:
            k = d.year * 10000 + d.month * 100 + d.day
            ty = types.get(d, 'REG')
            if k in info.index:
                mn, mx, sz = info.loc[k]
                last = int(mx) + 1; holes = (int(mx) - 120 + 1) - int(sz); first = int(mn)
            else:
                last = None; holes = None; first = None
            rows.append(dict(date=k, dow=d.weekday(), type=ty, era=era(d), first=first, last=last, holes=holes,
                             special=SPECIAL.get(d)))
        d += timedelta(days=1)
    c = pd.DataFrame(rows)
    # модальная минута обрыва по типу и эпохе
    modal = c[c['type'].isin(['HOL', 'EVE']) & c['last'].notna()].groupby(['type', 'era'])['last'].agg(lambda x: x.mode().iloc[0])
    st, close, prov, note = [], [], [], []
    for _, r in c.iterrows():
        if isinstance(r.special, str):
            st.append('special'); close.append(r['last']); prov.append('TAPE_ONLY'); note.append(r.special); continue
        if r['type'] in ('NEWYEAR', 'XMAS', 'GOODFRI'):
            if r['last'] is None or pd.isna(r['last']):
                st.append('closed'); close.append(None); prov.append('RULE+TAPE'); note.append('')
            else:
                st.append('short'); close.append(r['last']); prov.append('TAPE_ONLY')
                note.append(f'{r["type"]} with abbreviated session on tape (e.g. Good Friday NFP)')
            continue
        if r['type'] in ('HOL', 'EVE'):
            m = modal.get((r['type'], r['era']))
            if r['last'] == m:
                st.append('short'); close.append(m); prov.append('RULE+TAPE')
                note.append('' if r['holes'] == 0 else f'{int(r["holes"])} missing minutes before close')
            else:
                st.append('unknown'); close.append(r['last']); prov.append('UNKNOWN'); note.append(f'{r["type"]}: tape stop {r["last"]} vs modal {m}')
            continue
        # обычный день
        if r['last'] is None or pd.isna(r['last']):
            st.append('unknown'); close.append(None); prov.append('UNKNOWN'); note.append('no bars in window')
        elif r['last'] == 960:
            st.append('regular'); close.append(960); prov.append('RULE+TAPE'); note.append('' if r['holes'] == 0 else f'{int(r["holes"])} missing minutes')
        else:
            st.append('unknown'); close.append(960); prov.append('UNKNOWN'); note.append(f'tape stops at {r["last"]} on a regular-type day')
    c['status'] = st; c['close_mod'] = close; c['provenance'] = prov; c['note'] = note
    c['complete'] = c.apply(lambda r: bool(r['status'] in ('regular', 'short') and r['holes'] == 0), axis=1)
    return c, modal


if __name__ == '__main__':
    c, modal = build()
    c.to_csv('calendar_nq.csv', index=False)
    print('modal stop (ET minute+1) by type/era:', {k: f'{int(v)//60:02d}:{int(v)%60:02d}' for k, v in modal.items()})
    c['yr'] = c.date // 10000
    print(pd.crosstab(c.yr, [c.status]).to_string())
    print('\ncomplete objects by year (all minutes to official close):')
    print(c.groupby('yr').apply(lambda g: pd.Series({'regular_complete': int(((g.status == 'regular') & g.complete).sum()),
                                                     'short_complete': int(((g.status == 'short') & g.complete).sum())}), include_groups=False).T.to_string())
    print('\nunknown / special days 2018+:')
    print(c[(c.yr >= 2018) & c.status.isin(['unknown', 'special'])][['date', 'type', 'last', 'holes', 'status', 'note']].to_string(index=False))
