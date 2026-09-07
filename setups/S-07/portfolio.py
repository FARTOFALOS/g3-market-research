"""S-07: совпадают ли убыточные дни с прежними кандидатами.

Первый из трёх открытых шагов карточки. Вопрос узкий: две линии — годовой
выбор по семейству S-04/S-05/S-06 (576 правил) и S-07 — попадают в минус в
одни и те же дни или в разные? Ответ решает, имеет ли смысл держать их вместе.

Что берётся и чего здесь нет:
  * S-07 — базовый вариант NQ, минута 3, горизонт 120, без стопа, ОДИН контракт
    и БЕЗ ведения и катастрофического лимита. Ведение меняет величину дня, но
    не дату; вопрос здесь о датах.
  * прежняя линия — сохранённая дневная кривая совместного годового выбора из
    `data/research/S-06/selection_after_v3_combined_daily.parquet`, та самая,
    что даёт −$612,50.
  * день без сделки — это ноль, а не пропуск: позиции в этот день нет.
  * общий знаменатель — дни, известные обеим линиям.

Ни оптимизации, ни выбора весов здесь нет: один контракт на линию.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
S07 = ROOT / 'data/research/S-07/trades_v1.parquet'
PRIOR = ROOT / 'data/research/S-06/selection_after_v3_combined_daily.parquet'


def s07_daily(instrument='NQ', minute=3, horizon=120, stop=0.0) -> pd.DataFrame:
    t = pd.read_parquet(S07)
    m = ((t.instrument == instrument) & (t.minute == minute)
         & (t.horizon == horizon) & (t.stop == stop) & (t.status > 0))
    d = t.loc[m, ['date', 'net_dollars']].copy()
    return d.groupby('date', as_index=False).net_dollars.sum().rename(columns={'net_dollars': 's07'})


def prior_daily() -> pd.DataFrame:
    p = pd.read_parquet(PRIOR)
    p = p.loc[p.known_all, ['date', 'selection_net']].copy()
    return p.rename(columns={'selection_net': 'prior'})


def drawdown(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def line_stats(x: np.ndarray) -> dict:
    return dict(days=int(len(x)), total=round(float(x.sum()), 1),
                mean=round(float(x.mean()), 2) if len(x) else 0.0,
                dd=round(drawdown(x), 1), worst=round(float(x.min()), 1) if len(x) else 0.0,
                loss_days=int((x < 0).sum()),
                loss_share=round(float((x < 0).mean()), 3) if len(x) else 0.0)


def report(frame: pd.DataFrame) -> dict:
    a, b = frame.s07.to_numpy(), frame.prior.to_numpy()
    both = a + b
    la, lb = a < 0, b < 0
    observed = float((la & lb).mean())
    independent = float(la.mean() * lb.mean())
    return {
        'S-07': line_stats(a),
        'prior_selection': line_stats(b),
        'together_one_contract_each': line_stats(both),
        'loss_overlap': {
            'both_negative_share': round(observed, 3),
            'if_independent': round(independent, 3),
            'ratio': round(observed / independent, 2) if independent else None,
            'correlation': round(float(np.corrcoef(a, b)[0, 1]), 3),
        },
    }


def main() -> None:
    frame = s07_daily().merge(prior_daily(), on='date', how='inner')
    # A day either line skipped is a real day with no position, not a gap.
    frame[['s07', 'prior']] = frame[['s07', 'prior']].fillna(0.0)
    out = {'denominator': 'дни, известные обеим линиям',
           'variant_S07': 'NQ m3 h120 без стопа, один контракт, без ведения и лимита',
           'variant_prior': 'совместный годовой выбор S-04/S-05/S-06, 576 правил',
           'all_years': report(frame),
           'since_2020': report(frame.loc[frame.date >= '2020-01-01'])}
    (HERE / 'portfolio_v1.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    frame.assign(together=frame.s07 + frame.prior).to_csv(
        HERE / 'portfolio_daily_v1.csv', index=False, lineterminator='\n')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
