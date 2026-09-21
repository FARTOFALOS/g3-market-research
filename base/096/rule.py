#!/usr/bin/env python3
"""096 rule — детерминированное правило по свечам против reference labels.

Правило выписано прямо из текста критерия FREEZE_096_INSTRUMENT §3 и не содержит
свободных параметров. Вопрос, на который оно отвечает: нужен ли для этого чтения
внешний измерительный канал, или оно выражается неравенством по OHLC.

    python -B rule.py <labels.json>

Печатает: распределение классов, срез по слоям k, k-only baseline, пять правил
(полное v1 в двух вариантах и четыре вырожденных) с κ, точностью и recall.
Рыночные исходы не читаются: файл gold60.json содержит только префикс T0..q.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'

ISO, PRO, MIX = 'isolated_interruption', 'progressive_deterioration', 'mixed_or_unclear'
CLASSES = (ISO, PRO, MIX)
STRATA = ('1-2', '3-4', '5-6', '7-12', '13-50')


def parts(s, inward='red_body'):
    """Разбор сцены: где стоит M и что было после него.

    renews[i]  — бар i поставил новый наружный экстремум.
    last_renew — последний такой бар строго до q; на нём стоит M. Всякое
                 нарушение ДО него восстановлено по определению: позже экстремум
                 был превзойдён. Не восстановлено то, что после него.
    """
    n, h, o, c = s['n'], s['h'], s['o'], s['c']
    run, renews = -1e18, []
    for i in range(n):
        renews.append(h[i] > run)
        run = max(run, h[i])
    q = n - 1
    last_renew = max([i for i in range(q) if renews[i]], default=-1)
    inw = (lambda i: c[i] < o[i]) if inward == 'red_body' else (lambda i: c[i] < c[i - 1])
    unrecovered = sum(1 for i in range(last_renew + 1, q) if i > 0 and inw(i))
    m_bar_inward = last_renew > 0 and inw(last_renew)
    return last_renew, q, unrecovered, m_bar_inward


def rule_v1(s, inward='red_body'):
    _, _, unrec, m_in = parts(s, inward)
    return ISO if (unrec == 0 and not m_in) else PRO


RULES = {
    'полное v1, inward = красное тело': lambda s: rule_v1(s, 'red_body'),
    'полное v1, inward = закрытие ниже прежнего': lambda s: rule_v1(s, 'prev_close'),
    'только «бар перед q зелёный»': lambda s: ISO if s['c'][-2] >= s['o'][-2] else PRO,
    'только «M на баре прямо перед q»': lambda s: ISO if parts(s)[0] == parts(s)[1] - 1 else PRO,
    'только «нет невосстановленных откатов»': lambda s: ISO if parts(s)[2] == 0 else PRO,
    'только «бар M не закрылся внутрь»': lambda s: ISO if not parts(s)[3] else PRO,
}


def kappa(true, pred):
    n = len(true)
    po = sum(t == p for t, p in zip(true, pred)) / n
    ct, cp = collections.Counter(true), collections.Counter(pred)
    pe = sum(ct[c] * cp[c] for c in CLASSES) / n ** 2
    return (po - pe) / (1 - pe), po


def recall(true, pred, cls):
    d = sum(1 for t in true if t == cls)
    return (sum(1 for t, p in zip(true, pred) if t == cls == p) / d, d) if d else (float('nan'), 0)


def main(labels_path):
    scenes = {s['id']: s for s in json.loads((OUT / 'gold60.json').read_text(encoding='utf-8'))['scenes']}
    key = {k['id']: k for k in json.loads((OUT / 'gold60_key.json').read_text(encoding='utf-8'))}
    lab = json.loads(Path(labels_path).read_text(encoding='utf-8'))
    L = {x['id']: x['label'] for x in lab['labels'] if x['label']}
    assert set(L) == set(scenes), 'набор id не совпал с манифестом'
    ids = sorted(scenes)
    true = [L[i] for i in ids]

    print(f"разметчик: {lab.get('annotator', '?')} · размечено {len(L)}\n")
    for c, v in collections.Counter(true).most_common():
        print(f'  {c:<26} {v:>3}  ({v / len(ids):.2f})')

    print('\nметка по слоям k:')
    hit, pred_base = 0, {}
    for s in STRATA:
        sub = [i for i in ids if key[i]['stratum'] == s]
        cc = collections.Counter(L[i] for i in sub)
        maj, cnt = cc.most_common(1)[0]
        hit += cnt
        for i in sub:
            pred_base[i] = maj
        print(f"  {s:>6} | iso {cc.get(ISO, 0):>2}  prog {cc.get(PRO, 0):>2}  mix {cc.get(MIX, 0):>2} "
              f"| мажоритарный {maj} ({cnt}/{len(sub)})")
    kb, pb = kappa(true, [pred_base[i] for i in ids])
    print(f'\nk-only baseline: точность {pb:.3f}  κ = {kb:.3f}')

    print(f"\n{'правило':<46}{'κ':>7}{'точн.':>8}{'rec_iso':>9}")
    for name, f in RULES.items():
        pred = [f(scenes[i]) for i in ids]
        k, po = kappa(true, pred)
        r, d = recall(true, pred, ISO)
        print(f'  {name:<44}{k:>7.3f}{po:>8.3f}{r:>9.3f}  ({int(r * d)}/{d})')

    print('\nматрица полного v1 (строки — метка, столбцы — правило):')
    pred = {i: rule_v1(scenes[i]) for i in ids}
    print(f"  {'':<28}{'iso':>5}{'prog':>6}")
    for t in CLASSES:
        r = collections.Counter(pred[i] for i in ids if L[i] == t)
        if sum(r.values()):
            print(f'  {t:<28}{r.get(ISO, 0):>5}{r.get(PRO, 0):>6}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('нужен путь к файлу меток: python -B rule.py <labels.json>')
    main(sys.argv[1])
