# 091-TEMPORAL — прогон замороженного действия на продолжении ленты

Объявление, входная проверка и результат — [`../FREEZE_091_TEMPORAL.md`](../FREEZE_091_TEMPORAL.md)
§7.5–§7.7; итог для читателя — в карточке
[091](../../091-ekonomicheskoe-zavershenie-086-ispolnimyy-istoricheskiy-kandidat.md), раздел
«Проверка во времени».

Порядок повтора из корня репозитория (нужны `data/market/NQ`, `data/field/NQ`,
`data/forward/market/NQ`, `data/forward/_incoming/lynx1231_equity_index_minute.parquet`,
выходы `work/084/xray_NQ_evaluation.parquet`, `work/081a/paths/films_NQ_evaluation.parquet`,
`work/091/exec091.json`; numpy, pandas, pyarrow, numba):

```powershell
python -B base/091/temporal/gate_lens_vs_field.py   # линза против поля: 14 356 / 14 356
python -B base/091/temporal/gate_091_level.py       # регрессия обвязки, линза, бары lynx
python -B base/091/temporal/run_temporal.py         # единственный прогон по новой ленте
```

| Файл | Что это |
|---|---|
| `lens_riz.py` | RIZ с минутным T0 на ленте без поля — дословная копия правил замороженного сборщика (git `dd278b6`) |
| `harness091.py` | Замороженные функции 084/086/091 на поданных массивах и одиночный трейдер §1 |
| `gate_lens_vs_field.json`, `gate_091_level.json` | Входная проверка §7.4, записана до прогона |
| `temporal_result.json`, `temporal_signals.csv`, `temporal_daily.csv` | Результат прогона |

Поле `data/field` ни одним из скриптов не пишется.
