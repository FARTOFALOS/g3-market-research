"""074: the manner of the first return, read before the branches part.

The film's branch (branches.py) is decided late. The first minute that meets the
zone again is early. This asks what that minute already shows: how deep into the
zone it reaches and where it closes, both measured in the zone's own width, and
how the later branch is distributed over that.

Read first on the three rich states of 073, then repeated on every scene of the
field. No PnL, no Volume, no label from the future used as an input.

Run: python -B base/074/first_return.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branches import classify, minute_states, validate      # noqa: E402
from first_difference import scenes, membership             # noqa: E402

OUT = Path(__file__).resolve().parent
BUCKET = ['touch', '<half', '>half', 'whole zone', 'past far border']


def table(xl, s):
    bad = validate(xl, s)
    q = minute_states(xl, s)
    lab, info = zip(*[classify(v) for v in q])
    lab = np.array(lab, dtype=object)
    lab[bad != ''] = bad[bad != '']
    r = s.row.to_numpy()
    H, L, C = xl[r, :, 1], xl[r, :, 2], xl[r, :, 3]
    top, bot = s.zone_top.to_numpy(), s.zone_bottom.to_numpy()
    south = s.t0_exit_side.to_numpy() == 'south'
    w = np.maximum(top - bot, .25)
    near = np.where(south, bot, top)
    z = np.array([i['first_Z'] or 0 for i in info])
    use = (z > 0) & np.isin(lab, ['TRAVERSE', 'EXIT-RECLAIM', 'ZONE-ABSORB', 'OSCILLATION'])
    i = np.flatnonzero(use)
    sgn = np.where(south[i], 1.0, -1.0)
    reach = np.where(south[i], H[i, z[i]] - near[i], near[i] - L[i, z[i]]) / w[i]
    close = np.where(south[i], C[i, z[i]] - near[i], near[i] - C[i, z[i]]) / w[i]
    cut = lambda v: pd.cut(v, [-9, 0, .5, 1, 1.5, 9], labels=BUCKET)
    # the same close measured against a shuffled zone width: is the midpoint of
    # the zone doing the work, or only the distance travelled?
    rng = np.random.default_rng(74)
    ws = w[i][rng.permutation(len(i))]
    close_sh = np.where(south[i], C[i, z[i]] - near[i], near[i] - C[i, z[i]]) / ws
    return pd.DataFrame(dict(branch=lab[i], reach=cut(reach), close=cut(close),
                             close_shuffled_width=cut(close_sh), minute=z[i],
                             reach_raw=reach))


def show(df, title):
    print('\n===', title, f'(n={len(df)})')
    for col in [c for c in ['reach', 'close', 'close_shuffled_width'] if c in df]:
        p = pd.crosstab(df[col], df.branch, normalize='index').round(3)
        n = df[col].value_counts().reindex(BUCKET)
        p['n'] = n
        print(f'-- first returning minute, {col} into the zone (zone widths)')
        print(p.reindex(BUCKET).to_string())
    return {c: pd.crosstab(df[c], df.branch, normalize='index').round(4).to_dict()
            for c in df.columns if c not in ('branch', 'minute', 'reach_raw')}


def main():
    t0, X, xl, d = scenes()
    prep = json.load(open(OUT / 'preparation.json', encoding='utf-8'))
    mem = np.zeros(len(X), bool)
    for p in prep:
        if p['seed'] != 6727:
            mem |= membership(X, p)
    rich = d[mem[d.row.to_numpy()]].reset_index(drop=True)
    rep = {'rich_states': show(table(xl, rich), 'three rich states of 073')}
    rest = d[~mem[d.row.to_numpy()]].reset_index(drop=True)
    parts = []
    for k in range(0, len(rest), 40000):
        parts.append(table(xl, rest.iloc[k:k + 40000].reset_index(drop=True)))
    rest_df = pd.concat(parts, ignore_index=True)
    rep['rest_of_field'] = show(rest_df, 'every other scene of the field')
    # the risk: a first Z that already closed past the far border is TRAVERSE
    # beginning, not an early sign. Drop every film whose first Z reached the
    # far border at all.
    rich_df = table(xl, rich)
    for nm, df in [('rich states', rich_df), ('rest of field', rest_df)]:
        sub = df[df.reach_raw < 1].drop(columns=['reach', 'close_shuffled_width'])
        rep['short_of_far_border_' + nm.split()[0]] = show(
            sub, f'{nm}: first Z stayed short of the far border')
        sub2 = df[df.reach_raw < 1].drop(columns=['reach', 'close'])
        rep['short_shuffled_' + nm.split()[0]] = show(
            sub2, f'{nm}: the same, width of the zone shuffled')
    (OUT / 'first_return.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
