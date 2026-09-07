# Повторение S-04

[Карточка](../S-04-vozvrat-vstrechnymi-svechami.md): возврат встречными свечами
после T0. Цикл v1 завершён, все 216 правил после расходов отрицательны;
в торговлю кандидат не предлагается. Здесь лежит то, чем это посчитано.

Из корня `g3-market-research`. Среда как у S-05: Python 3.14.4,
[зависимости](../S-05/requirements.txt).

```powershell
python -B setups/S-04/run.py
python -B setups/S-04/reversal.py signals
python -B setups/S-04/check.py
python -B setups/S-04/reversal.py backtest
python -B setups/S-04/reversal.py selection
python -B setups/S-04/scenes.py
```

`reversal.py` принимает ровно три фазы: `signals`, `backtest`, `selection`.

| Что | Где |
|---|---|
| Правило, издержки, знаменатели | `SEARCH.md` |
| Все 216 вариантов и их итоги | `summary_v1.csv` |
| Последовательный годовой выбор по годам | `selection_v1.csv` |
| Сколько дней вообще давали сигнал | `recognition_v1.csv` |
| Семейство проверенных правил | `family_v1.json` |
| Проверки, названные до просмотра исходов | `checks_before_outcomes.json` |
| Прочитанные сцены на свечах | `ES_read_1.png`, `ES_read_2.png`, `NQ_read_1.png`, `NQ_read_2.png` |
| Хеши кода и входов | `run_hashes_v1.json`, `run_hashes_at_v1.json` |
| Дневные деньги, решения, исходы | `data/research/S-04/` |

## Календарь живёт здесь, и им пользуются соседи

`calendar.parquet` — торговые дни XNYS с наносекундными границами сессии
(`open_ns` — закрытие последней минуты перед открытием акций, `close_ns` —
закрытие сессии), происхождение и версия — в `calendar_source.json`.

**Этот файл читают S-05, S-06 и S-07**, а не только S-04:
`pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')`. Он остаётся здесь
потому, что здесь был построен, и переносить его — значит одновременно править
код четырёх кандидатов. Если календарь понадобится пересобрать, это отдельная
работа с пересчётом всех четырёх, а не правка одного каталога.

Скрипты S-04 читают минутные массивы `data/market/<ins>/` и **замороженное поле**
`data/field/<ins>/`: `run.py` берёт паспорта ризов и хеширует их вместе с
источниками, от них считаются сигналы. Ничего в `data/market` и `data/field` не
пишется. Тяжёлый вывод уходит в `data/research/S-04/` и в git не едет.
