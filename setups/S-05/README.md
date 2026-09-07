# Повторение S-04/S-05 и просмотр свечей

Текущий результат — [карточка S-05, v2](../S-05-otkat-vstrechnymi-svechami.md).
Исторический кандидат положителен после принятых расходов; устойчивое
прибыльное преимущество не установлено. Здесь воспроизводится именно этот
результат, включая проигравшие варианты и оценку способа выбора.

Нужен корень `g3-market-research` с существующими `data/market` и `data/field`.
Большие данные локальные и в git не входят. Их отсутствие в новом клоне
не разрешает пересборку. Расчёт читает только OHLC/время и готовые RIZ;
рыночный объём не используется.

Проверенная среда: Windows, Python 3.14.4. В текущем рабочем каталоге есть
`data/tmp/s04-venv/Scripts/python.exe`; это удобство, не обязательный путь.
Для другой среды установите пакет проекта и зависимости:

```powershell
python -m pip install -e .
python -m pip install -r setups/S-05/requirements.txt
```

## Быстро проверить сохранённую версию

При уже существующих результатах в `data/research/S-04` и `S-05`:

```powershell
python -B setups/S-04/check.py
python -B setups/S-05/verify.py
python -B setups/S-05/present.py --date 2025-06-16
```

`verify.py` заново узнаёт сигналы из полной копии паспортов, сверяет журнал
v2 и независимо проверяет цены/часы каждой сделки; затем повторяет заявленные
проверки расходов, задержки и неопределённости. `check.py` физически обрезает
свечные массивы для проверки доступности узнавания и проверяет исполнение
на различающих коротких примерах. Это не независимая рыночная выборка.

`present.py --date YYYY-MM-DD` показывает сделки выбранных дат, сохраняет
свечи и все участвовавшие в минуте RIZ. Без аргумента воспроизводит две
иллюстрации карточки. Времена на рисунках — закрытия свечей, America/New_York.
Возможность просматривать любую дату сокращает путь от итога к реальному
объекту; в строке без сделки нельзя придумывать сигнал по её будущему исходу.

## Полностью повторить расчёт

Команды последовательны, из корня. Они перезаписывают только собственные
результаты S-04/S-05 и читают замороженное поле:

```powershell
python -B setups/S-04/run.py
python -B setups/S-04/scenes.py ES
python -B setups/S-04/scenes.py NQ
python -B setups/S-04/reversal.py signals
python -B setups/S-04/check.py
python -B setups/S-04/reversal.py backtest
python -B setups/S-04/reversal.py selection
python -B setups/S-05/follow.py backtest
python -B setups/S-05/follow.py selection
python -B setups/S-05/verify.py
python -B setups/S-05/timing.py --version v1
python -B setups/S-05/timing.py --version v2
python -B setups/S-05/present.py
```

Семейство содержит 216 правил S-04 и 216 S-05; `selected_v1.json` и
`selected_v2.json` сохраняют два последующих выбора. Автоматического поиска
нового победителя в `verify.py` нет. Исходные хеши прогонов сохранены отдельно
как `run_hashes_at_v1.json` в обеих папках. В S-05 исходное объявление —
`SEARCH_at_v1.md`; изменяемая `SEARCH.md` дополнена реальной историей выбора.
Повторный расчёт создаёт свежую `run_hashes_v1.json` по текущим файлам и не
создаёт задним числом предварительной регистрации.

## Где искать доказательства

| Что | Файл относительно корня |
|---|---|
| Входы, SHA-256 всех 4 320 паспортов/манифестов и OHLC | `data/research/S-04/input_hashes.json` |
| Происхождение и неизвестное о фиде | `SOURCE_DATA.json` |
| Зафиксированный календарь и его источник | `setups/S-04/calendar.parquet`, `calendar_source.json` |
| Отбор первых просмотренных сцен | `data/research/S-04/ES_visual_selection.json`, `NQ_visual_selection.json` |
| Узнавание по каждому RIZ, включая отсутствие/неизвестность | `data/research/S-04/{ES,NQ,YM}_signals_k{2,3,4}.parquet` |
| Результаты всех вариантов | `setups/S-04/summary_v1.csv`, `setups/S-05/summary_v1.csv` |
| Все решения, дни и числа сделок | `data/research/S-0{4,5}/{all_decisions,daily,counts}_v1.parquet` |
| Повтор и чувствительности v2 | `setups/S-05/verification_v2.json`, `sensitivities_v2.csv`, `yearly_v2.csv` |
| Исполненные/отменённые/неизвестные решения v2 | `data/research/S-05/selected_v2_replay.parquet` |
| Жизнь исходных RIZ к решению | `data/research/S-05/selected_v2_lifecycle.parquet` |
| Годовой выбор по прошлому, всё семейство | `setups/S-05/selection_combined_v1.csv` |
| Его дневная кривая | `data/research/S-05/selection_daily_combined_v1.parquet` |
| Все дополнительно просмотренные горизонты | `setups/S-05/timing_exploration_v{1,2}.csv` |
| Точные пути без ограничения шкалы графика | `data/research/S-05/selected_v{1,2}_aligned_points.npy` |
| Свечи и предыстория иллюстраций | `data/research/S-05/illustrations_v2.json`, файлы дат |
| Снимок файлов завершённого прохода | `setups/S-05/completion.json` |

В таблице фигурные скобки обозначают несколько конкретных файлов. В журнале
исполненная сделка имеет `day_known == True` и `status > 0`; отменённая цена
входа — `status == -1`, недоступный вход — `-2`, неизвестный дальнейший путь —
`-3`, конфликт — `-4`. Известный день без сделки имеет нулевой дневной результат;
неизвестный день — NaN. Весь такой день исключён ретроспективно по покрытию;
это не фильтр, который можно знать на открытии рынка.

Проверки общего слоя уже выполнены: 64 passed командой
`python -B -m pytest tests -q -p no:cacheprovider`. Они не доказывают рыночное
преимущество. Ограничения первичного фида и склейки остаются неизвестными
свойствами исходного архива; пересчёт той же ленты их не устраняет.
