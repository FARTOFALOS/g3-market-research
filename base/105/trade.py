"""Прибор действия (линия nq-manual, режим «торговая конструкция», слово трейдера 2026-09-23).

Сигнал = (минута узнавания i_rec, сторона, цена естественной отмены, цель или NaN, последний допустимый бар).
Вход — open минуты i_rec+1 (должна идти подряд в той же дате). Отмена — касание low/high уровня отмены;
исполнение min(open, уровень) − 0,5 пт (для long). Цель — лимит, нужен проход на тик; открытие за целью — по open.
Стоп и цель в одной минуте — считается стоп. Время — close последнего допустимого бара (≤ 120 минут от входа,
не позже официального закрытия дня по calendar_nq.csv). Пропуск минут внутри сделки — выход по последнему close,
помечается (исход неизвестен, сделка не удаляется). Расход 0,75 пт за оборот.
Допуск к деньгам: риск идеи (вход − отмена) + 0,75 + 0,5 ≤ 20 пт; остальные — «широкие», считаются отдельно.
"""
import numpy as np, pandas as pd
from numba import njit
from tape import load_minutes

COST, SLIP, TICK, LIMIT, DAY_LIMIT = 0.75, 0.5, 0.25, 20.0, 60.0
EPOCHS = [(20060101, 20121231, '2006-12'), (20130101, 20191231, '2013-19'), (20200101, 20251231, '2020-25'),
          (20260101, 20261231, '2026')]


def epoch(d):
    for a, b, n in EPOCHS:
        if a <= d <= b:
            return n
    return 'other'


class Tape:
    def __init__(self, start='2005-12-01', end='2026-07-11', inst='NQ'):
        df = load_minutes(start, end, inst).reset_index(drop=True)
        self.o, self.h, self.l, self.c = (df[k].to_numpy() for k in 'ohlc')
        self.date = df['date'].to_numpy().astype(np.int64); self.mod = df['mod'].to_numpy().astype(np.int64)
        self.ts = df['ts'].to_numpy().astype(np.int64)
        self.u = pd.Series(self.h - self.l).rolling(30).median().to_numpy()
        cal = pd.read_csv('calendar_nq.csv')
        self.cal = cal.set_index('date')
        close = dict(zip(cal.date, cal.close_mod.fillna(960).astype(int)))
        # индексы окна 02:00–официальное закрытие по датам
        d = self.date; m = self.mod
        cm = np.array([close.get(x, 960) for x in d])
        inwin = (m >= 120) & (m < cm)
        self.close_mod = cm
        idx = np.nonzero(inwin)[0]
        dd = d[idx]
        cut = np.nonzero(np.diff(dd))[0] + 1
        self.days = {}
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)]):
            self.days[int(dd[a])] = idx[a:b]
        # первый бар каждой даты (для ночи) и бары до 02:00
        self.date_start = {}
        u, first = np.unique(d, return_index=True)
        for x, f in zip(u, first):
            self.date_start[int(x)] = int(f)

    def at(self, D, minute):
        """индекс бара даты D с открытием в minute (ET) или −1"""
        ix = self.days.get(D)
        if ix is None:
            return -1
        k = np.searchsorted(self.mod[ix], minute)
        if k < len(ix) and self.mod[ix[k]] == minute:
            return int(ix[k])
        return -1


@njit(cache=True)
def _sim(o, h, l, c, mod, dt, irec, side, stop, target, ilast, slip, tick):
    n = len(irec)
    ent = np.full(n, np.nan); px = np.full(n, np.nan); typ = np.zeros(n, np.int8); jx = np.full(n, -1)
    mae = np.full(n, np.nan); mfe = np.full(n, np.nan)
    for k in range(n):
        i = irec[k]
        if i + 1 >= len(o) or dt[i + 1] != dt[i] or mod[i + 1] != mod[i] + 1 or i + 1 > ilast[k]:
            typ[k] = -1          # нет исполнимого входа
            continue
        j = i + 1; s = side[k]; e = o[j]; st = stop[k]; tg = target[k]
        ent[k] = e
        if s * (e - st) <= 0:
            typ[k] = -2          # вход уже за уровнем отмены — идея мертва на входе
            continue
        if tg == tg and s * (tg - e) <= 0:
            typ[k] = -3          # остаток уже пройден к входу
            continue
        mf = 0.0; ma = 0.0
        while True:
            if s > 0:
                if l[j] <= st:
                    ma = max(ma, e - st); px[k] = min(o[j], st) - slip; typ[k] = 1; break
                if tg == tg and h[j] >= tg + tick:
                    mf = max(mf, tg - e); px[k] = max(o[j], tg); typ[k] = 2; break
                mf = max(mf, h[j] - e); ma = max(ma, e - l[j])
            else:
                if h[j] >= st:
                    ma = max(ma, st - e); px[k] = max(o[j], st) + slip; typ[k] = 1; break
                if tg == tg and l[j] <= tg - tick:
                    mf = max(mf, e - tg); px[k] = min(o[j], tg); typ[k] = 2; break
                mf = max(mf, e - l[j]); ma = max(ma, h[j] - e)
            if j >= ilast[k]:
                px[k] = c[j]; typ[k] = 3; break
            if j + 1 >= len(o) or dt[j + 1] != dt[j] or mod[j + 1] != mod[j] + 1:
                px[k] = c[j]; typ[k] = 4; break
            j += 1
        jx[k] = j; mae[k] = ma; mfe[k] = mf
    return ent, px, typ, jx, mae, mfe


def simulate(T, sig, tmax=120):
    """sig: DataFrame c колонками date, irec, side, stop, target (NaN — без цели), iend (последний допустимый бар
    по логике сцены или −1). Возвращает sig с исходом."""
    s = sig.copy().reset_index(drop=True)
    if len(s) == 0:
        return s
    dayend = np.array([T.days[d][-1] for d in s.date])
    iend = s.get('iend', pd.Series(-1, index=s.index)).to_numpy()
    iend = np.where(iend < 0, np.iinfo(np.int64).max, iend)
    ilast = np.minimum.reduce([s.irec.to_numpy() + tmax, dayend, iend]).astype(np.int64)
    ent, px, typ, jx, mae, mfe = _sim(T.o, T.h, T.l, T.c, T.mod, T.date, s.irec.to_numpy().astype(np.int64),
                                      s.side.to_numpy().astype(np.int64), s.stop.to_numpy().astype(float),
                                      s.target.to_numpy().astype(float), ilast, SLIP, TICK)
    s['entry'] = ent; s['exit'] = px; s['xtype'] = typ; s['jexit'] = jx; s['mae'] = mae; s['mfe'] = mfe
    s['risk'] = s.side * (s.entry - s.stop)
    s['plan_loss'] = s.risk + COST + SLIP
    s['net'] = s.side * (s.exit - s.entry) - COST
    s['hold'] = s.jexit - s.irec
    s['epoch'] = s.date.map(epoch)
    s['fits'] = (s.xtype > 0) & (s.plan_loss <= LIMIT)
    return s


XT = {1: 'stop', 2: 'target', 3: 'time', 4: 'gap'}


def report(s, label, extra=None):
    rows = []
    for ep, g in s.groupby('epoch'):
        live = g[g.xtype > 0]
        f = live[live.fits]
        if len(f) == 0:
            rows.append(dict(v=label, epoch=ep, signals=len(g), entered=len(live))); continue
        net = f.net
        r = dict(v=label, epoch=ep, signals=len(g), entered=len(live), fits=len(f),
                 fits_share=round(len(f) / max(len(live), 1), 2), risk_med=round(live.risk.median(), 1),
                 net_pt=round(net.mean(), 2), t=round(net.mean() / net.std(ddof=1) * np.sqrt(len(net)), 2) if len(net) > 2 else np.nan,
                 win=round((net > 0).mean(), 2), avg_w=round(net[net > 0].mean(), 1), avg_l=round(net[net <= 0].mean(), 1),
                 hold=round(f.hold.mean(), 0), days=f.date.nunique(), pt_per_day=round(net.sum() / f.date.nunique(), 2),
                 wide_net=round(live[~live.fits].net.mean(), 2) if (~live.fits).any() else np.nan)
        for k, nm in XT.items():
            r[nm] = round((f.xtype == k).mean(), 2)
        if extra:
            r.update(extra(f))
        rows.append(r)
    return rows
