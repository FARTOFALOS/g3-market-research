# FREEZE 090 — контракт ветки «additional prefix information about immediate continuation beyond declared representation/context»

Заморожено ДО split и X-ray. Вопрос узкий: несёт ли доступная **не позднее q**
величина `C` дополнительную информацию о немедленном продолжении сверх
объявленного `R₀+K`. НЕ тест достаточности `R₀` относительно всего remaining path.
Внутренняя диагностика на ранее экспонированной NQ discovery — не внешняя
репликация. `Volume` не используется. Замороженное поле не пересобирается.

## 1. Объект, cursor, population (prefix-honest)

- Объект — object-scene: тройка `(t0_spine_pos, side, exit_boundary)`. Наружная
  живая сцена читается дословно 084/087/088 (`fullfilm.get_bars/annotate`):
  фронты `M/MB`, терминалы `contact_b` / `gap` / `archive` (+ end_pos-терминал),
  горизонта нет.
- Терминал детерминирован тройкой (проверено: `end_pos` уникален на тройке).
  Поэтому представитель эпизода по `life` из 089 **отменён** — это future-selection.
- Канонизация **только по префиксу**: схлопываются лишь prefix-identical копии
  (одинаковая тройка; TF/зон-дубли). Разные `exit_boundary` при том же `(t0,side)`
  — разные сцены, сохраняются; их зависимость несёт `E`.
- Cursor `q` — живой фронт-событийный бар `j` (`newM ∨ newMB`, бит-идентично
  `fullfilm.annotate`), минута закрыта, вся информация до `j` доступна. **q
  фиксирован закрытым front-event cursor'ом; все компоненты `C` доступны не позднее
  q.** Оператор, требующий информацию после q, — другой recognition cursor и другой
  вопрос (заново `R₀,K,population,future-from-recognition`).
- Территория — только NQ discovery (2006-01-05 … 2018-12-24). ES/YM и evaluation не
  открываются. Референс-числа 089 (106 603 курсора / 17 190 актов) — не размер этой
  population; фактические counts печатаются до результата.

## 2. Outcome CONT* и observability (censoring — не K)

- `CONT*(q)=1` — следующее квалифицирующее событие наблюдается до физического
  терминала;
- `CONT*(q)=0` — физический терминал (`contact_b`/`retreat_in`) наблюдается до
  следующего события;
- `CONT*(q)=censored` — наблюдаемость потеряна (`gap`/`archive`) до разрешения.

Старый `CONT` 089 (последнее событие=0, цензура неявно=0) остаётся prior
descriptive result, не переиспользуется.

**Censoring (фиксация 1):** все исходные `q` сохраняются в описании population.
Анализ resolved-only, если применяется, оценивает `P(CONT*|R₀,K,C,resolved)` и НЕ
переносится автоматически на всю population. Claim для всей population требует
отдельно обоснованной обработки цензуры или sensitivity/bounds по неизвестным
исходам; иначе общий статус — UNRESOLVED. Обязательный диагностик: зависит ли доля
цензуры от `C` (informative censoring) — если да, claim ограничивается.

## 3. R₀ — tested compact representation

Дословно FREEZE_089 §3: `dpos=sgn·(close[j]−exit_boundary)`;
`psig=mean_{t=t0..j}(high−low)` (`prefix_sigma _sig`); `nz=dpos/psig` —
детерминированная комбинация, не отдельная ось.

## 4. K — только prefix-context, доступный на recognition (≤ q)

`side`; `ord` (порядковый номер события = зрелость); час сессии только если объявлен
в уровень до счёта, иначе unused. `ord` — в K, не в C: его вклад ⇒
REPRESENTATION/CONTEXT REFINED, не канал.

## 5. E — evidence / provenance / dependence (не K, не outcome)

Единица независимости — не курсор. Зависимость: курсоры одной сцены; сцены,
делящие `(t0,side)`/перекрывающиеся во времени пути. Обрабатывается кластерным
ресэмплом по времени-блокам и purge/embargo (§7). В признак не входит.

## 6. Вложенные уровни и разведение исходов

Сравниваются три вложенных уровня out-of-sample: `R₀` → `R₀+K` → `R₀+K+C`.
- прирост `R₀→R₀+K` ⇒ **REPRESENTATION / CONTEXT REFINED**;
- прирост `R₀+K→R₀+K+C`, устоявший novelty+adequacy+censoring ⇒ кандидат в
  **ADDITIONAL INFORMATION**;
- прирост снят более сильным чтением R₀ ⇒ baseline-reading issue (не доказывает сам
  по себе семантическую сводимость, см. §9);
- опоры/разрешения не хватает ⇒ UNRESOLVED.

Baseline `R₀`/`R₀+K` — гибкие непрерывные estimand'ы (сплайн/GAM/монотонно-гибкая
поверхность), cross-fit; НЕ децильные бины 089. **Монотонность baseline как
установленный закон не навязывается.**

## 7. Split: time-blocked NQ, leakage-safe (до любого X-ray) — фиксация 2

- Test membership определяется **временем q и prefix-доступной информацией**, не
  будущей длиной Film-1. Test НЕ очищается от длинных сцен по их future life.
- Защищается test и необходимая outcome-follow-up лента: purge/embargo применяется к
  **обучающему** материалу, который мог бы показать эту ленту. Train-сцена
  исключается, если её лента `[t0, end_pos]` достаёт до `[T_split − embargo, T_end]`.
  Сцена, чьи q straddle границу, целиком выбывает из train (в обоих фолдах одна и та
  же сцена не появляется).
- Любая техническая future-dependent фильтрация test называется отдельной
  ограниченной population.
- X-ray и весь выбор (divergence, `C`, recognition, support, normalization, baseline)
  — только на train. Test трогается один раз без ретюнинга.

## 8. Discovery wording — фиксация 5

Контраст `CONT*=0/1` среди похожих сцен на train — **discovery contrast, не
доказанный residual**. Дополнительная информация возникает только после
замороженной проверки prefix-`C` сверх адекватного `R₀+K` на test.

## 9. Novelty C и baseline adequacy — раздельно — фиксация 4

- **Novelty:** `C=f(R₀)` — re-encoding, отбраковывается **по определению и
  сохраняемой информации**, а не по эмпирической восстанавливаемости на конечной
  таблице. `C` не должен быть также переименованием `K`.
- **Adequacy:** исчезновение прироста под более сильным estimator ⇒ baseline-reading
  issue, но само по себе НЕ доказывает семантическую сводимость `C` к R₀.
- Обе докладываются отдельно.

## 10. Measurement contract — фиксация 4

- До test заморожены **один primary effect/score** и (если нужен) **один** заранее
  объявленный robustness check.
- Три вещи разведены: effect metric; meaningful ε; statistical resolution/uncertainty.
- Primary effect metric: out-of-sample различие proper scoring (объявляется числом в
  §11 до test).
- Uncertainty: обоснована для непрерывного baseline и зависимых сцен — кластерный
  ресэмпл по времени-блокам сцен. Conditional-randomization заранее НЕ предписывается;
  naive permutation внутри приблизительных `R₀+K` страт НЕ используется.
- Negative result допустим только если uncertainty достаточно узка, чтобы исключить
  эффект ≥ ε. Иначе — UNRESOLVED (не negative).

## 11. Resolution numbers (заморозить до test)

- primary score, meaningful ε, support floor, embargo, T_split, number of time-blocks,
  seed — фиксируются в этом файле по итогу train-калибровки, ДО открытия test.
  (Заполняется после train-only X-ray и до решающего прогона.)

## 12. Terminal statuses (без rescue)

1. **ADDITIONAL INFORMATION INTERNALLY VALIDATED**
2. **REPRESENTATION / CONTEXT REFINED**
3. **ADDITIONAL INFORMATION OF MEANINGFUL SIZE NOT ESTABLISHED AT ADEQUATE RESOLUTION**
4. **UNRESOLVED**

После открытия test запрещены смена outcome/split/population/нормировки/support/
recognition/порога/определения `C`. Отрицательная ветка не расширяется. Возврат к
архитектуре — только при противоречии, меняющем population, cursor, outcome или смысл
claim.

## 13. Повтор

numpy, pandas; `data/market/NQ/{open,high,low,close,close_ts_utc_ns}.npy`,
`work/081a/paths/films_NQ_discovery.parquet`, `base/084/race084.py`. Из `base/090/`:

    python -B pop.py     # prefix-honest population + CONT* + время q; печать counts
