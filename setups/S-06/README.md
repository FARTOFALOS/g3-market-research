# Повторение S-06

[Карточка](../S-06-svechnaya-pauza-u-riz.md): две отрицательные торговые
трактовки частой свечной паузы у действующего RIZ, повторённые затем на суженном
определении паузы (v3, закрытия внутри общей области). Все 144 правила
сохранены; общая оценка выбора включает 576 правил S-04/S-05/S-06.

Из корня `g3-market-research` с существующими `data/market` и `data/field`.
Среда как у S-05: Python 3.14.4, [зависимости](../S-05/requirements.txt).
Локально подготовлен `data/tmp/s04-venv/Scripts/python.exe`.

```powershell
python -B setups/S-06/observe.py
python -B setups/S-06/run.py prepare
python -B setups/S-06/check.py
python -B setups/S-06/run.py backtest
python -B setups/S-06/run.py selection
python -B setups/S-06/check_reject.py
python -B setups/S-06/reject.py backtest
python -B setups/S-06/reject.py selection
python -B setups/S-06/verify.py
python -B setups/S-06/confined.py prepare_and_check
python -B setups/S-06/confined.py backtest
python -B setups/S-06/confined.py selection
python -B setups/S-06/verify_v3.py
python -B setups/S-06/show.py
```

Календарь, покрытие и исходная копия паспортов для первой картинки — из
S-04: `setups/S-04/calendar.parquet`, `data/research/S-04/*_coverage.parquet`,
`*_objects.parquet`. Подготовка — `python -B setups/S-04/run.py`; поле только
читается. Общий выбор требует дневных денег/чисел сделок S-04/S-05;
их повтор описан в [S-05](../S-05/README.md). Новые данные и пересборка не нужны.

Быстро перепроверить существующие результаты — `verify.py`. Просмотр первой
исполненной сделки иллюстративного NQ v2 на произвольную дату:

```powershell
python -B setups/S-06/show.py --date 2025-11-13
```

Рисунок, свечи и ID появятся в `data/research/S-06/scene_YYYY-MM-DD*`.
Без даты скрипт воспроизводит иллюстрации карточки. По ID и позициям из JSON
можно читать всю предысторию через существующий `Field`.

| Что | Где |
|---|---|
| Развитие догадки и раскрытый поиск | `SEARCH.md` |
| Исходные объявления и хеши версий | `SEARCH_at_v{1,2}.md`, `run_hashes_at_v{1,2}.json`, `run_hashes_v3.json` |
| Переданный трейдером материал, числа не перепроверены | `LENSES_FROM_USER.txt` |
| Все конфигурации и итоги | `family_v{1,2,3}.json`, `summary_v{1,2,3}.csv` |
| Префиксы и случаи исполнения | `checks_before_outcomes.json`, `checks_v{2,3}_before_outcomes.json` |
| Повтор, цены 117 163 сделок, эпохи | `verification.json` |
| Узнавание v3 и его повтор, 33 012 сверенных сделок | `recognition_v3.csv`, `verification_v3.json` |
| Происхождение и неизменность входов | `inputs.json`, `integrity.json` |
| Выбор по прошлому после версий | `selection_after_v{2,3}_*.csv` |
| Все per-RIZ паузы | `data/research/S-06/*_pauses_k*.parquet`, суженные v3 — `*_confined_k*.parquet` |
| Дневные деньги, числа позиций, все решения | `data/research/S-06/{daily,counts,all_decisions}_v{1,2,3}.parquet` |
| Дневные кривые выбора | `data/research/S-06/selection_after_v2_*_daily.parquet` |

Неизвестный день — NaN. В решениях `status > 0` — исполнение, 0 — не сработавший
пробой v1, −1 — отменённый вход, −2 — неизвестный вход, −3 — неизвестный
дальнейший путь, −4 — занятость предыдущей позицией. Отчётная сделка требует
также `day_known == True`; полнота дня оценивается ретроспективно. Heat/MFE —
границы по всей свече, для внутриминутных исполнений они могут включать путь
вне фактической позиции.

Повтор перезаписывает рассчитанные результаты, но не исходные снимки
`*_at_v*`. Свежая `run_hashes_v*.json` описывает текущие байты и не создаёт
предварительную регистрацию задним числом. Большие результаты локальны и
в git не входят. Ограничения фида сохраняются; живого исполнения здесь нет.
