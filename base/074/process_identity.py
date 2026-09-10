"""074: identity of the market process at a RIZ scene, read from the zone it is
anchored to. Prefix only (T0..+7, closed minutes), no outcome, no Volume, no PnL.

Anchor: the RIZ zone itself (zone_bottom, zone_top) and the side E through which
the T0 close left it. Both come from the frozen field, not from the examples.

Events, all observable at the close of their minute:
  RETURN   m>=1, the candle range intersects [zone_bottom, zone_top] again
  CONFIRM  m>=1, the candle lies wholly beyond the exit border E
The first of the two decides; k is that minute.

  RETURN first  -> same      (the exit was refused)
  CONFIRM first -> different (the exit held)
  neither by +7 -> indeterminate

Run from repository root: python -B base/074/process_identity.py <command>
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / 'work/074'
SEEDS = [20071, 20517, 75248, 6727]


def load():
    z = np.load(CACHE / 'NQ_prefix.npz')
    t0, x = z['t0'], z['x']
    d = pd.read_parquet(CACHE / 'NQ_zones.parquet')
    # one RIZ = one scene; copies of the same zone across timeframes are dropped
    d = d.drop_duplicates(['t0_ts_ns', 'zone_top', 'zone_bottom', 't0_exit_side'])
    row = pd.Series(np.arange(len(t0)), index=t0.astype('int64'))
    d = d[d.t0_ts_ns.isin(row.index)].copy()
    d['row'] = row.loc[d.t0_ts_ns].to_numpy()
    d['year'] = pd.to_datetime(d.t0_ts_ns).dt.year.to_numpy()
    return t0, x, d.reset_index(drop=True)


def events(x, d, top=None, bot=None):
    """RETURN / CONFIRM minute per scene. top/bot override the anchor (control)."""
    r = d.row.to_numpy()
    H, L = x[r, :, 1], x[r, :, 2]
    top = d.zone_top.to_numpy() if top is None else top
    bot = d.zone_bottom.to_numpy() if bot is None else bot
    south = (d.t0_exit_side.to_numpy() == 'south')
    back = (L <= top[:, None]) & (H >= bot[:, None])
    away = np.where(south[:, None], H < bot[:, None], L > top[:, None])
    known = np.isfinite(H) & np.isfinite(L)
    back &= known
    away &= known
    first = lambda m: np.where(m[:, 1:].any(1), m[:, 1:].argmax(1) + 1, 99)
    return first(back), first(away)


def label(ret, con):
    out = np.full(len(ret), 'indeterminate', dtype=object)
    out[ret < con] = 'same'
    out[con < ret] = 'different'
    return out, np.minimum(ret, con)


def profile(d, lab, name):
    n = len(lab)
    share = {k: float((lab == k).mean()) for k in ['same', 'different', 'indeterminate']}
    ev, od = d.year % 2 == 0, d.year % 2 == 1
    tr = float((lab[od] == 'same').mean() / (lab[ev] == 'same').mean())
    eras = {f'{lo}-{hi}': float((lab[(d.year >= lo) & (d.year <= hi)] == 'same').mean())
            for lo, hi in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]}
    return dict(name=name, n=int(n), share=share, transfer_even_to_odd=tr, epochs=eras)


def read_scenes():
    t0, x, d = load()
    sel = d[d.row.isin(SEEDS)]
    ret, con = events(x, sel)
    lab, k = label(ret, con)
    for i, (_, r) in enumerate(sel.iterrows()):
        print(f'row={r.row} {pd.Timestamp(r.t0_ts_ns)} tf{r.tf_minutes} '
              f'zone {r.zone_bottom}-{r.zone_top} exit {r.t0_exit_side} '
              f'-> {lab[i]} at +{k[i]} (return +{ret[i]}, confirm +{con[i]})')
        for m, (O, H, L, C) in enumerate(x[r.row]):
            inside = L <= r.zone_top and H >= r.zone_bottom
            beyond = H < r.zone_bottom if r.t0_exit_side == 'south' else L > r.zone_top
            tag = 'ZONE' if inside else ('EXIT' if beyond else 'OPPO')
            print(f'    +{m} [{tag}] O{O:.2f} H{H:.2f} L{L:.2f} C{C:.2f}')


def measure():
    t0, x, d = load()
    rep = {}
    ret, con = events(x, d)
    lab, k = label(ret, con)
    rep['base'] = profile(d, lab, 'zone anchor, range-return, whole-candle-confirm')
    rep['base']['k_minute'] = {str(m): int((k == m).sum()) for m in range(1, 8)}
    rep['base']['k_median_same'] = float(np.median(k[lab == 'same']))
    rep['base']['k_median_different'] = float(np.median(k[lab == 'different']))

    rng = np.random.default_rng(74)
    # control 1: the same zone, the exit side flipped
    d2 = d.copy()
    d2['t0_exit_side'] = np.where(d.t0_exit_side == 'south', 'north', 'south')
    r2, c2 = events(x, d2)
    l2, _ = label(r2, c2)
    rep['control_flipped_side'] = profile(d2, l2, 'exit side flipped')

    # control 2: sham zone of the same width displaced by +-2 widths, anchor kept
    w = np.maximum(d.zone_width.to_numpy() if 'zone_width' in d else
                   d.zone_top.to_numpy() - d.zone_bottom.to_numpy(), 0.25)
    off = w * 2 * rng.choice([-1.0, 1.0], len(d))
    top, bot = d.zone_top.to_numpy() + off, d.zone_bottom.to_numpy() + off
    t0c = d.t0_close.to_numpy()
    still = np.where(d.t0_exit_side == 'south', t0c < bot, t0c > top)
    r3, c3 = events(x, d, top, bot)
    l3, _ = label(r3, c3)
    rep['control_shifted_zone'] = profile(d[still], l3[still.nonzero()[0]], 'zone displaced by 2 widths')

    # variants that should not change the reading
    r = d.row.to_numpy()
    H, L, C = x[r, :, 1], x[r, :, 2], x[r, :, 3]
    top_, bot_ = d.zone_top.to_numpy()[:, None], d.zone_bottom.to_numpy()[:, None]
    south = (d.t0_exit_side.to_numpy() == 'south')[:, None]
    kn = np.isfinite(H) & np.isfinite(L)
    first = lambda m: np.where(m[:, 1:].any(1), m[:, 1:].argmax(1) + 1, 99)
    v_ret_close = first(((C <= top_) & (C >= bot_)) & kn)
    v_con_close = first(np.where(south, C < bot_, C > top_) & kn)
    for nm, rr, cc in [('return_by_close', v_ret_close, con),
                       ('confirm_by_close', ret, v_con_close),
                       ('both_by_close', v_ret_close, v_con_close)]:
        lv, _ = label(rr, cc)
        p = profile(d, lv, nm)
        p['agreement_with_base'] = float((lv == lab).mean())
        rep['variant_' + nm] = p

    # k-invariance: everything after k replaced by noise must not move the answer
    xs = x.copy()
    for m in range(1, 8):
        pick = (k == m)
        rows = d.row.to_numpy()[pick]
        if len(rows) == 0:
            continue
        noise = rng.normal(0, 50, size=(len(rows), 8 - m - 1, 4)) + xs[rows, m + 1:, :]
        xs[np.repeat(rows, 1)[:, None], np.arange(m + 1, 8)[None, :], :] = np.sort(noise, axis=2)[:, :, [1, 3, 0, 2]]
    rn, cn = events(xs, d)
    ln, kn2 = label(rn, cn)
    rep['k_invariance'] = dict(label_unchanged=float((ln == lab).mean()),
                               k_unchanged=float((kn2 == k).mean()))
    # k-1 must still be indeterminate
    dec = k < 99
    prior = np.zeros(len(d), bool)
    for m in range(2, 8):
        pick = dec & (k == m)
        xt = x[d.row.to_numpy()[pick]][:, :m].copy()
        pad = np.full((pick.sum(), 8 - m, 4), np.nan)
        rt, ct = events(np.concatenate([xt, pad], 1), d[pick].assign(row=np.arange(pick.sum())))
        lt, _ = label(rt, ct)
        prior[np.flatnonzero(pick)] = (lt == 'indeterminate')
    rep['prior_minute_indeterminate'] = float(prior[dec & (k > 1)].mean())
    (OUT / 'identity_measurement.json').write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def anchor_controls():
    """Does the reading belong to the zone, or only to the distance to it?

    Three controls, all keeping the prefix untouched:
      mirror  - a region of the same width at the same distance on the other
                side of the T0 close; geometry matched, history not;
      far     - the near border kept, the far border taken from another scene
                (width shuffled inside width deciles);
      near    - the near border moved to another scene's distance, shuffled
                inside distance deciles.
    """
    t0, x, d = load()
    r = d.row.to_numpy()
    H, L, C = x[r, :, 1], x[r, :, 2], x[r, :, 3]
    top, bot = d.zone_top.to_numpy(), d.zone_bottom.to_numpy()
    w = np.maximum(top - bot, .25)
    south = d.t0_exit_side.to_numpy() == 'south'
    t0c = d.t0_close.to_numpy()
    dist = np.where(south, bot - t0c, t0c - top)
    ok = np.isfinite(H).all(1) & np.isfinite(dist) & (dist > 0)
    rng = np.random.default_rng(74)
    rep = {'n': int(ok.sum())}

    # state at +7 relative to the region, split by the operator's label
    def gap(top_, bot_, south_, sel):
        ret, con = events(x, d.assign(t0_exit_side=np.where(south_, 'south', 'north')), top_, bot_)
        lab, _ = label(ret, con)
        beyond = np.where(south_, C[:, 7] < bot_, C[:, 7] > top_)
        a = beyond[sel & (lab == 'same')].mean()
        b = beyond[sel & (lab == 'different')].mean()
        return float(a), float(b), float(b - a)

    rep['plus7_gap_real'] = gap(top, bot, south, ok)
    mtop = np.where(south, t0c - dist, t0c + dist + w)
    rep['plus7_gap_mirror'] = gap(mtop, mtop - w, ~south, ok)

    push = np.where(south[:, None], H[:, 1:] - bot[:, None], top[:, None] - L[:, 1:]).max(1)
    reached = ok & (push >= 0)
    dec = lambda v, n: pd.qcut(pd.Series(v).rank(method='first'), n, labels=False).to_numpy()
    wd = dec(w[reached], 10)
    pr, wr = push[reached], w[reached]
    rows = []
    for g in range(10):
        i = np.flatnonzero(wd == g)
        sh = wr[i][rng.permutation(len(i))]
        rows.append(dict(width_decile=g, median_width=float(np.median(wr[i])), n=len(i),
                         stop_inside_real=float((pr[i] < wr[i]).mean()),
                         stop_inside_shuffled=float((pr[i] < sh).mean())))
    rep['far_border'] = dict(overall_real=float((pr < wr).mean()),
                             overall_shuffled=float(np.average([r['stop_inside_shuffled'] for r in rows],
                                                               weights=[r['n'] for r in rows])),
                             by_width_decile=rows)

    back = np.where(south[:, None], H[:, 1:], -L[:, 1:]).max(1)
    lvl = np.where(south, t0c, -t0c)
    dd, bb, cc = dist[ok], back[ok], lvl[ok]
    dq = dec(dd, 10)
    real = float((bb >= cc + dd).mean())
    sh = float(np.mean([(bb[np.flatnonzero(dq == g)] >=
                         cc[np.flatnonzero(dq == g)] +
                         dd[np.flatnonzero(dq == g)][rng.permutation((dq == g).sum())]).mean()
                        for g in range(10)]))
    rep['near_border'] = dict(touch_real=real, touch_shuffled=sh)
    (OUT / 'anchor_controls.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:1200])


def cold_set():
    """The 18 unseen prefixes handed to the cold reader (odd years only).

    Writes work/074/cold_cases.json (definition + candles, no labels) and
    base/074/cold_key.json (the operator's own answers, withheld from the reader).
    """
    t0, x, d = load()
    ret, con = events(x, d)
    lab, k = label(ret, con)
    odd = (d.year % 2 == 1).to_numpy()
    rng = np.random.default_rng(1074)
    pick = []
    for m, n in [(lab == 'same', 6), (lab == 'different', 6),
                 (lab == 'indeterminate', 2), ((k > 1) & (k < 99), 4)]:
        pick += rng.choice(np.flatnonzero(m & odd), n, replace=False).tolist()
    pick = list(dict.fromkeys(pick))
    rng.shuffle(pick)
    cases, key = [], []
    for n, i in enumerate(pick, 1):
        r = d.iloc[i]
        cases.append(dict(case=n, zone_bottom=float(r.zone_bottom), zone_top=float(r.zone_top),
                          exit_side=r.t0_exit_side,
                          candles=[dict(m=m, O=float(a), H=float(b), L=float(c), C=float(e))
                                   for m, (a, b, c, e) in enumerate(x[r.row])]))
        key.append(dict(case=n, row=int(r.row), t0=str(pd.Timestamp(r.t0_ts_ns)),
                        tf=int(r.tf_minutes), label=lab[i], k=int(k[i])))
    (CACHE / 'cold_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'cold_key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(cases), 'cases')


def reversibility():
    """Which event of the scene cannot be taken back by the rest of the prefix?

    For every candidate boundary event the scene is followed to +7 and asked
    whether the later minutes cancel the reading the event announced.
    """
    t0, x, d = load()
    r = d.row.to_numpy()
    H, L = x[r, :, 1], x[r, :, 2]
    top, bot = d.zone_top.to_numpy()[:, None], d.zone_bottom.to_numpy()[:, None]
    south = (d.t0_exit_side.to_numpy() == 'south')[:, None]
    kn = np.isfinite(H) & np.isfinite(L)
    back = (L <= top) & (H >= bot) & kn                     # price at the zone again
    away = np.where(south, H < bot, L > top) & kn           # whole minute on the exit side
    over = np.where(south, L > top, H < bot) & kn           # whole minute past the far side
    rep = {'n': int(kn.all(1).sum())}

    def follow(event, cancel, name):
        rows = []
        for m in range(1, 8):
            first = event[:, m] & ~event[:, 1:m].any(1) if m > 1 else event[:, 1]
            first &= kn.all(1)
            later = cancel[:, m + 1:].any(1) if m < 7 else np.zeros(len(d), bool)
            n = int(first.sum())
            rows.append(dict(minute=m, n=n,
                             cancelled_later=float(later[first].mean()) if n else None))
        tot = sum(r_['n'] for r_ in rows)
        can = sum((r_['cancelled_later'] or 0) * r_['n'] for r_ in rows)
        rep[name] = dict(by_minute=rows, n=tot, cancelled_share=can / tot if tot else None)
        print(name, 'n=%d cancelled=%.4f' % (tot, can / tot if tot else float('nan')),
              [(r_['minute'], r_['n'], None if r_['cancelled_later'] is None
                else round(r_['cancelled_later'], 3)) for r_ in rows], flush=True)

    # RETURN says "the exit was refused"; only a scene that never comes back could deny it
    follow(back, np.zeros_like(back), 'RETURN_cancelled_by_nothing')
    # CONFIRM says "the exit held"; a later return to the zone denies it
    follow(away, back, 'CONFIRM_cancelled_by_return')
    # traverse says "the scene turned into the mirror scene"; a later return denies it
    follow(over, back, 'TRAVERSE_cancelled_by_return')
    # is the decay of the cancellation rate a property, or just the prefix ending?
    full = kn.all(1)
    for nm, ev in [('CONFIRM', away), ('TRAVERSE', over)]:
        flat = []
        for m in range(1, 7):
            first = (ev[:, m] & ~ev[:, 1:m].any(1)) if m > 1 else ev[:, 1]
            first &= full
            row = dict(minute=m, n=int(first.sum()))
            for j in (1, 2, 3):
                e = min(m + j, 7) + 1
                # only scenes that still have j minutes ahead take part
                row[f'within_{j}'] = (float(back[first, m + 1:e].any(1).mean())
                                      if m + j <= 7 else None)
            flat.append(row)
            print('%s +%d n=%d' % (nm, m, row['n']),
                  {k: None if v is None else round(v, 3)
                   for k, v in row.items() if k.startswith('within')}, flush=True)
        rep['hazard_at_fixed_lookahead_' + nm] = flat
    (OUT / 'reversibility.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                            encoding='utf-8')


WINDOW = 51


def load_long():
    """T0..+50 closed minutes: the instrument's observation window, not a claim
    that the process must end inside it."""
    path = CACHE / f'NQ_prefix{WINDOW}.npz'
    if not path.exists():
        z = np.load(CACHE / 'NQ_prefix.npz')
        ts0 = z['t0']
        m = ROOT / 'data/market/NQ'
        ts = np.load(m / 'close_ts_utc_ns.npy', mmap_mode='r')
        want = ts0[:, None] + np.arange(WINDOW)[None, :] * 60_000_000_000
        pos = np.searchsorted(ts, want)
        safe = np.minimum(pos, len(ts) - 1)
        hit = (pos < len(ts)) & (ts[safe] == want)
        x = np.stack([np.where(hit, np.load(m / f'{f}.npy', mmap_mode='r')[safe], np.nan)
                      for f in ['open', 'high', 'low', 'close']], axis=2).astype('float32')
        np.savez_compressed(path, t0=ts0, x=x)
    z = np.load(path)
    return z['t0'], z['x']


def long_line():
    """The state sequence over k = 0..50 and the cancellation hazard at every k."""
    t0, x8, d = load()
    _, x = load_long()
    r = d.row.to_numpy()
    H, L = x[r, :, 1], x[r, :, 2]
    top, bot = d.zone_top.to_numpy()[:, None], d.zone_bottom.to_numpy()[:, None]
    south = (d.t0_exit_side.to_numpy() == 'south')[:, None]
    kn = np.isfinite(H) & np.isfinite(L)
    back = (L <= top) & (H >= bot) & kn
    away = np.where(south, H < bot, L > top) & kn
    over = np.where(south, L > top, H < bot) & kn
    live = kn[:, :2].all(1)
    print('scenes', int(live.sum()), 'minutes', WINDOW)

    # minute 0 is excluded: the T0 candle straddles the zone by construction
    b0, a0 = back.copy(), away.copy()
    b0[:, 0] = False
    a0[:, 0] = False
    cum_b = np.maximum.accumulate(b0, 1)
    cum_a = np.maximum.accumulate(a0, 1)
    seq = []
    for k in range(1, WINDOW):
        sel = live & kn[:, k]
        n = int(sel.sum())
        b, a = cum_b[sel, k], cum_a[sel, k]
        seq.append(dict(k=k, n=n,
                        returned=float(b.mean()),
                        confirmed_never_returned=float((a & ~b).mean()),
                        neither=float((~a & ~b).mean())))
    rep = dict(n_scenes=int(live.sum()), window=WINDOW, state_by_k=seq)
    for row in seq[:8] + seq[9::10]:
        print('k=%2d n=%6d returned=%.3f exit-held-so-far=%.3f neither=%.3f'
              % (row['k'], row['n'], row['returned'],
                 row['confirmed_never_returned'], row['neither']), flush=True)

    # a reading announced at k: is it taken back later, at equal remaining future?
    for nm, ev in [('CONFIRM', away), ('TRAVERSE', over)]:
        rows = []
        firstev = ev & ~np.maximum.accumulate(np.concatenate(
            [np.zeros((len(d), 1), bool), ev[:, :-1]], 1), 1)
        for k in range(1, WINDOW - 10):
            f = firstev[:, k] & live & kn[:, k:k + 11].all(1)
            n = int(f.sum())
            if n < 200:
                continue
            rows.append(dict(k=k, n=n,
                             **{f'within_{j}': float(back[f, k + 1:k + 1 + j].any(1).mean())
                                for j in (1, 3, 10)}))
        rep['hazard_' + nm] = rows
        print(nm, 'cancellation at equal remaining future:')
        for row in rows[:6] + rows[6::8]:
            print('   k=%2d n=%6d j1=%.3f j3=%.3f j10=%.3f'
                  % (row['k'], row['n'], row['within_1'], row['within_3'], row['within_10']),
                  flush=True)
    (OUT / 'long_line.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                        encoding='utf-8')


if __name__ == '__main__':
    globals()[sys.argv[1]]()
