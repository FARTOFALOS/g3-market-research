"""S-07: заморозка версии v1 — хеши кода и входов, по которым результат восстановим."""
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    code = {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}
    code['S07_opening_drift.pine'] = sha(HERE / 'S07_opening_drift.pine')
    inputs = {'calendar': sha(ROOT / 'setups/S-04/calendar.parquet'),
              'SOURCE_DATA.json': sha(ROOT / 'SOURCE_DATA.json')}
    market = {}
    for ins in ['ES', 'NQ', 'YM']:
        for name in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']:
            p = ROOT / 'data/market' / ins / (name + '.npy')
            a = np.load(p, mmap_mode='r')
            market[f'{ins}/{name}'] = dict(sha256=sha(p), minutes=int(len(a)))
    trades = ROOT / 'data/research/S-07/trades_v1.parquet'
    summary = pd.read_csv(HERE / 'summary_v1.csv')
    frozen = dict(
        version='v1', frozen='2026-09-07',
        rule=dict(instrument='NQ', decision_minute=3, window=30, entry='open следующей минуты',
                  hold_minutes=120, exit_if_negative_at=15, catastrophic_limit_candles=16,
                  breakeven='never', trailing='never', cost_per_turn_usd=15,
                  unit='median true range of last 30 closed minutes'),
        code=code, inputs=inputs, market=market,
        results=dict(trades_file=str(trades.relative_to(ROOT)),
                     configurations=int(len(summary)),
                     nq_m3_h120_nostop_2020_total=273050.0,
                     nq_m3_h120_managed_2020_total=236265.0),
        environment=dict(python=sys.version.split()[0], platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__))
    (HERE / 'FROZEN_v1.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in frozen.items() if k != 'market'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
