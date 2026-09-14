"""075: the zone word of every scene, and its two readings.

state of minute k (k = 1..120), oriented by the exit side of T0:
    E  the whole minute is on the exit side of the near border
    Z  the minute's range meets the zone (a touch of a border counts)
    F  the whole minute is beyond the far border
    .  the minute is absent from the spine (session break, gap, archive edge)

reading 1 (exact time):  the word indexed by k
reading 2 (order):       run-length compression -> letters + durations
"""
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

CHARS = np.array(list('.EZF'))


def states(d, pos, hi, lo, chunk=40000):
    """(n_scenes, 120) uint8 codes: 0 '.', 1 E, 2 Z, 3 F"""
    out = np.zeros((len(d), common.HOR), np.uint8)
    r = d.row.to_numpy()
    top = d.zone_top.to_numpy(np.float32)
    bot = d.zone_bottom.to_numpy(np.float32)
    south = (d.t0_exit_side.to_numpy() == 'south')
    for a in range(0, len(d), chunk):
        b = min(a + chunk, len(d))
        rr = r[a:b]
        H = common.window(hi, pos[rr])[:, 1:]
        L = common.window(lo, pos[rr])[:, 1:]
        t, o, s = top[a:b, None], bot[a:b, None], south[a:b, None]
        kn = np.isfinite(H) & np.isfinite(L)
        E = np.where(s, H < o, L > t)
        F = np.where(s, L > t, H < o)
        out[a:b] = np.where(~kn, 0, np.where(E, 1, np.where(F, 3, 2)))
    return out


def runs(code):
    """for one scene: list of (letter_code, start_minute, length) up to the first gap"""
    n = len(code)
    stop = n
    g = np.flatnonzero(code == 0)
    if len(g):
        stop = g[0]
    if stop == 0:
        return []
    c = code[:stop]
    brk = np.flatnonzero(np.diff(c)) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [stop]))
    return [(int(c[s]), int(s) + 1, int(e - s)) for s, e in zip(starts, ends)]


def order_word(rr):
    return ''.join('.EZF'[k] for k, _, _ in rr)


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    d = common.scene_table()
    st = states(d, pos, h, l)
    np.save(common.CACHE / 'states.npy', st)
    obs = (st != 0).argmin(1)
    obs = np.where((st != 0).all(1), common.HOR, obs)
    print('scenes', len(d))
    print('observed length of the word: median', int(np.median(obs)),
          ' share with full 120:', round(float((obs == common.HOR).mean()), 3),
          ' share with <50:', round(float((obs < 50).mean()), 3))
    import collections
    cnt = collections.Counter()
    cnt50 = collections.Counter()
    for i in range(len(d)):
        rr = runs(st[i])
        if not rr:
            continue
        cnt[order_word(rr)] += 1
        rr50 = runs(st[i][:50])
        cnt50[order_word(rr50)] += 1
    print('\n--- reading 2 (order of events), window 120, distinct sequences:', len(cnt))
    for w, n in cnt.most_common(25):
        print(f'{n:7d}  {n/len(d):6.3f}  {w}')
    print('\n--- same, window 50, distinct:', len(cnt50))
    for w, n in cnt50.most_common(15):
        print(f'{n:7d}  {n/len(d):6.3f}  {w}')


if __name__ == '__main__':
    main()
