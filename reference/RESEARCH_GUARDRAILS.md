# Research Guardrails — Recurring Failure Modes, Corrections, and Audit Lessons from G3

Это операционная память проекта. Не «архив ошибок» и не список стиля. Здесь собраны
повторяющиеся исследовательские ошибки, в которые агенты G3 систематически
скатываются, и конкретные коррекции, которыми мы из них выходили. Цель — чтобы
следующий холодный агент не повторял месяцы уже оплаченных ошибок.

Парный документ — [`SETUP_DISCOVERY_METHOD.md`](SETUP_DISCOVERY_METHOD.md) (как идти
вперёд). Читать оба перед новой candidate family. Worked case обоих —
[`CASE_086_091_FROM_XRAY_TO_ACTION.md`](CASE_086_091_FROM_XRAY_TO_ACTION.md).

Каждый guardrail передаёт **форму ошибки**, а не только финальное правило:
**Temptation** — что агент естественно хотел сделать; **Why it looked reasonable** —
почему это было соблазнительно; **What was actually wrong** — какое различие
терялось; **G3 case** — конкретный пример; **Correction** — как куратор/аудитор
изменил постановку; **Invariant** — правило, которое должно пережить кейс;
**Tripwire** — фраза/конструкция, после которой надо остановиться и проверить себя.

---

## A. Правильная арифметика на неправильном объекте
- **Temptation:** посчитать метрику и поверить ей, потому что код чист и тесты зелёные.
- **Why it looked reasonable:** арифметика действительно верна; unit-тесты проходят.
- **What was actually wrong:** population / scene / event / cursor не соответствуют
  смысловому вопросу — метрика отвечает на другой вопрос.
- **G3 case:** повторяется всей дугой; семантический сброс 2026-09-03 (per-RIZ Film-1
  против T0-пакета/фиксированного окна) — ровно об этом.
- **Correction:** строка «относительно объекта» введена в каждую карточку; курсор,
  observation minute и принадлежность сцене называются до счёта.
- **Invariant:** перед массовым счётом назвать физический объект, observation minute и
  что именно принадлежит сцене. Зелёные тесты не доказывают правильный объект.
- **Tripwire:** «unit tests pass, so the population is right».

## B. Future-conditioned population
- **Temptation:** включить сцену, потому что позже случился нужный terminal или
  известна длина её будущей жизни.
- **Why it looked reasonable:** «мы же изучаем именно такие случаи».
- **What was actually wrong:** право сцены существовать в parent population стало
  функцией будущего.
- **G3 case:** аудит Film-1 (080/088): членство префиксное; возврат к `b` — конец окна
  наблюдения, не критерий включения.
- **Correction:** membership определяется в T0/prefix; future может задавать длину
  наблюдаемого фильма, но не право сцены быть в популяции.
- **Invariant:** parent-population membership — prefix-honest.
- **Tripwire:** «keep only films that reached / lasted / eventually …».

## C. Future-dependent representative selection
- **Temptation:** дедуплицировать объект, выбрав «канонический» ряд по будущей длине.
- **Why it looked reasonable:** нужен один представитель на объект; таблица потом
  выглядит полностью prefix-like.
- **What was actually wrong:** выбор строки уже сделан будущим — leakage спрятан за
  чистым видом итоговой таблицы.
- **G3 case:** 089 выбирал физический акт через `life = end_pos - t0 …last()`; 090
  отменил это (терминал детерминирован тройкой, канонизация только по префиксу).
- **Correction:** канонизация только по prefix-идентичности.
- **Invariant:** любая дедупликация/canonicalization, решающая, какая запись
  представляет объект, сама prefix-honest для соответствующего estimand.
- **Tripwire:** `groupby(...).last()` после `sort_values('life'/'end'/future)`.

## D. Censoring silently becomes failure
- **Temptation:** последний наблюдаемый event → `CONT=0`.
- **Why it looked reasonable:** удобная бинарная метка.
- **What was actually wrong:** `not observed to continue` ≠ `observed to terminate`.
- **G3 case:** старое чтение `CONT` в 089; 090 ввёл `CONT*` с явной цензурой.
- **Correction:** `lost`/`archive`/`gap`/unresolved ordering → censored, не 0.
- **Invariant:** отсутствие наблюдения продолжения — не наблюдение терминала.
- **Tripwire:** «last event = 0 / failure».

## E. Resolved-only silently changes estimand
- **Temptation:** выкинуть censored и считать как обычно.
- **Why it looked reasonable:** «мы же считаем только чистые случаи».
- **What was actually wrong:** оценивается уже conditional-on-resolution population, не
  исходная.
- **G3 case:** 090/091 — resolved-only явно помечен как `P(·|resolved)`, не переносится
  на всю популяцию без обоснования цензуры или bounds.
- **Correction:** claim для всей популяции требует обработки цензуры или
  favorable/adverse bounds; иначе UNRESOLVED.
- **Invariant:** исключение censored меняет estimand — это отражается в claim, не
  только в preprocessing.
- **Tripwire:** «drop unknown, then report the probability».

## F. Leakage through test cleaning / purge
- **Temptation:** очистить test от «неудобных»/длинных сцен или задать его членство по
  будущей длине.
- **Why it looked reasonable:** «уберём выбросы, чтобы тест был чище».
- **What was actually wrong:** test-членство стало функцией будущего сцены.
- **G3 case:** 090 split — test-членство по времени `q`; purge/embargo применён к
  **train**, не к test.
- **Correction:** защищать test, убирая/эмбаргируя обучающий материал, а не очищая test
  через его будущее. Любая future-dependent фильтрация test — отдельная ограниченная
  population.
- **Invariant:** membership теста не зависит от его будущего.
- **Tripwire:** «test set should exclude long / unresolved scenes».

## G. Cursor drift
- **Temptation:** начать с `q`, затем кандидат требует ещё две свечи — и продолжать
  звать это тем же `q`.
- **Why it looked reasonable:** «это та же сцена».
- **What was actually wrong:** информация появилась после `q`; recognition сместился.
- **G3 case:** зафиксировано в 091 (q — закрытый front-event cursor; всё в C доступно
  ≤ q; иначе новый recognition — новый вопрос).
- **Correction:** сместился recognition → заново определить population, R0/context,
  remaining future, executable opportunity; движение между старым q и новым
  recognition не кредитуется.
- **Invariant:** информация после q ⇒ новый recognition cursor.
- **Tripwire:** «just two more bars to confirm, still the same signal».

## H. X-ray becomes live information
- **Temptation:** найти красивый будущий разделитель и описать его как раннее состояние.
- **Why it looked reasonable:** он же реально различает продолжения.
- **What was actually wrong:** пропущен rewind — нет prefix-факта до расхождения.
- **G3 case:** обязательный rewind в методе (B); 074 (первая Z-минута) — пример честной
  prefix-величины, найденной так.
- **Correction:** `future divergence → what existed before it → first honest closed
  minute`.
- **Invariant:** нет prefix-факта → различие остаётся описанием будущего.
- **Tripwire:** «scenes that later did X differ, so X-state exists early».

## I. Model residual becomes “hidden market state”
- **Temptation:** увидеть красивый остаток модели и назвать его пропущенной рыночной
  информацией.
- **Why it looked reasonable:** baseline systematically мимо в конкретной зоне.
- **What was actually wrong:** ошибка модели — только локатор, не свидетельство
  пропущенной информации.
- **G3 case:** 090 — coarse kNN дал resid −0,159 у границы; interaction/boundary logit
  и LOESS его сняли (resid ≈ 0).
- **Correction:** перед открытием нового `C` проверить adequacy разумных чтений уже
  объявленного representation.
- **Invariant:** model error is a locator, not evidence of omitted market information.
- **Tripwire:** «the residual shows a hidden state».

## J. Хорошая calibration становится “representation sufficient”
- **Temptation:** «модель откалибрована → представления достаточно».
- **Why it looked reasonable:** предсказания совпадают с наблюдаемыми частотами.
- **What was actually wrong:** идеальные 0,40 могут скрывать prefix-группы 0,20/0,60.
- **G3 case:** 090 — sufficiency-тест не проводился; calibration≠sufficiency назван явно.
- **Correction:** adequacy baseline и sufficiency representation — разные claims.
- **Invariant:** no miscalibration ≠ no additional information.
- **Tripwire:** «calibrated, therefore no more information».

## K. «Не добавляет predictive gain» → «поглощено»
- **Temptation:** написать «R0 absorbed side/ord» / «side/ord reducible to R0».
- **Why it looked reasonable:** добавление не улучшило out-of-fold score.
- **What was actually wrong:** empirical recoverability ≠ semantic information
  equivalence.
- **G3 case:** 090 — K (side, ord) Δlog-loss −0,0008.
- **Correction:** «did not provide detectable incremental predictive information under
  this analysis».
- **Invariant:** нет обнаружимой добавки ≠ семантическая сводимость.
- **Tripwire:** «absorbed by», «reducible to», «explained completely by».

## L. Timing / speed становится direction
- **Temptation:** «происходит быстрее/раньше → туда и торговать».
- **Why it looked reasonable:** сильный, переносимый эффект скорости.
- **What was actually wrong:** ускорение arrival/hazard/duration не даёт directional
  expectancy.
- **G3 case:** 083/085 — крупная переносимая структура СКОРОСТИ, directional перевес не
  установлен; σ → speed, не direction.
- **Correction:** отдельный directional/economic тест обязателен.
- **Invariant:** «faster» ≠ «trade that way».
- **Tripwire:** «resolves quicker, so the edge is …».

## M. Process transition становится setup
- **Temptation:** нашли воспроизводимый переход процесса — назвать сетапом.
- **Why it looked reasonable:** переход реален и предрегистрированно PASS.
- **What was actually wrong:** на honest recognition directional remaining path
  симметричен.
- **G3 case:** 087 WOECB′ — established transition, исполнимого directional edge нет.
- **Correction:** после recognition отдельно спросить: что ещё осталось ценой?
- **Invariant:** real mechanism / transition ≠ trade.
- **Tripwire:** «we found a real transition, so there's an edge».

## N. Absence of new state становится absence of setup
- **Temptation:** «сначала найти новую информацию, только потом спрашивать о деньгах».
- **Why it looked reasonable:** казалось методологически чистым.
- **What was actually wrong:** торговая возможность может целиком жить в известных
  координатах.
- **G3 case:** 086→091 — executable candidate без нового скрытого состояния.
- **Correction:** как только есть узнаваемый кандидат — спрашивать про remaining
  movement и деньги.
- **Invariant:** informational novelty и economic usefulness — разные оси.
- **Tripwire:** «no new state, so nothing to trade».
- **Tripwire (residual gate):** «if known state fully explains the outcome, stop the
  setup line». Before stopping, ask separately whether that already-known state
  organizes a **remaining executable payoff asymmetry** (favorable vs adverse path
  after honest recognition). No residual information is required for a setup —
  выявлено cold-agent transfer-check 2026-09-17: даже сильный агент склонен делать
  «residual сверх геометрии» обязательным пропуском к торговому вопросу. Не делать
  «no residual → stop»; сначала проверить экономику известной геометрии.

## O. Single opposite-outcome pair становится доказательством скрытого состояния
- **Temptation:** показать две сцены с одинаковым R0 и разными futures как omitted
  variable.
- **Why it looked reasonable:** «вот же, одинаковые входы — разный исход».
- **What was actually wrong:** одно и то же conditional distribution штатно даёт разные
  реализации.
- **G3 case:** 090 — единичные matched pairs явно НЕ использованы как доказательство;
  требовался ensemble.
- **Correction:** нужна воспроизводимая ensemble-level conditional structure.
- **Invariant:** один matched pair не доказывает omitted variable.
- **Tripwire:** «same state, different outcome proves a hidden variable».

## P. Feature leaderboard создаёт онтологию
- **Temptation:** посчитать сотни prefix-features, отсортировать по gain, назвать топ
  «новым состоянием рынка».
- **Why it looked reasonable:** топ-признак реально предсказывает.
- **What was actually wrong:** переход `best predictor → physical mechanism/state` без
  семантической реконструкции.
- **G3 case:** мандат 090 прямо запретил широкий feature matrix, ранжируемый по gain.
- **Correction:** feature engineering как инструмент разрешён; запрещён скачок к
  «механизму» без отдельной semantic reconstruction.
- **Invariant:** predictor ≠ mechanism.
- **Tripwire:** «the top feature is the market state / the mechanism is».

## Q. Threshold / horizon rescue
- **Temptation:** после отрицательного результата сменить horizon/threshold/filter/delay
  /subgroup.
- **Why it looked reasonable:** «просто уточняю ту же гипотезу».
- **What was actually wrong:** каждое изменение после outcome — новая discovery branch.
- **G3 case:** правило stopping/no-rescue в 086/091 freeze; отрицательные ветки 075/081
  не спасались.
- **Correction:** frozen negative branch нельзя спасать внутри того же claim; новый
  вопрос — новая ветка и провенанс.
- **Invariant:** post-outcome изменение = новая ветка, не уточнение.
- **Tripwire:** «let's try another threshold / horizon / filter».

## R. Arbitrary horizon заменяет natural market fork
- **Temptation:** задать исход как `+N bars` для удобства.
- **Why it looked reasonable:** просто в коде и универсально.
- **What was actually wrong:** горизонт может не соответствовать экономическому вопросу.
- **G3 case:** 086 — естественная вилка `b` против «за M» смыслово чище произвольного
  горизонта.
- **Correction:** fixed horizon не запрещён, но объясни, почему он отвечает
  экономическому вопросу, а не удобству кода.
- **Invariant:** горизонт обоснован экономикой, не кодом.
- **Tripwire:** «target = close in N bars» без обоснования.

## S. Row count становится evidence count
- **Temptation:** 100 минут/курсоров одной сцены как 100 наблюдений.
- **Why it looked reasonable:** больше строк — больше «данных».
- **What was actually wrong:** длинный фильм не даёт больший epistemic weight.
- **G3 case:** 089/090 — единица независимости = физический акт/сцена; block bootstrap
  по дням; 086 holdout — 87 фильмов из одного дня.
- **Correction:** хранить physical object/act, sequence, dependence, cluster/day.
- **Invariant:** row count ≠ number of independent market experiments.
- **Tripwire:** «N = number of rows, so we have plenty of power».

## T. Exact matching fails → «у рынка нет сопоставимых состояний»
- **Temptation:** из отсутствия точных совпадений вывести отсутствие сопоставимых сцен.
- **Why it looked reasonable:** «мы честно сравнили, совпадений нет».
- **What was actually wrong:** богатый список координат почти уникализирует каждую сцену.
- **G3 case:** 076 — ноль точных пар предрешён списком из 11 координат
  (ожидание 4,2·10⁻¹¹), точное сравнение существовало на подмножестве.
- **Correction:** проверять, не свойство ли это representation, а не рынка.
- **Invariant:** нет онтологического вывода из combinatorial sparsity.
- **Tripwire:** «no exact matches, so the market has no comparable states».

## U. Lack of support становится negative result
- **Temptation:** «эффекта нет», когда сравнимые сцены есть только в разных
  geometry/context.
- **Why it looked reasonable:** сравнение формально проведено.
- **What was actually wrong:** вопрос неидентифицируем на общей поддержке.
- **G3 case:** 078 — аудит общей поддержки как отдельный обязательный шаг.
- **Correction:** правильный исход — `insufficient/common support not established`.
- **Invariant:** нет общей поддержки → не «нет эффекта».
- **Tripwire:** «no effect» без карты общей поддержки.

## V. Statistical nonsignificance становится «no effect»
- **Temptation:** отсутствие значимости → закрыть гипотезу.
- **Why it looked reasonable:** CI включает ноль.
- **What was actually wrong:** CI может допускать economically meaningful effect.
- **G3 case:** 083/084/086 — «unresolved: точность/идентификация», а не «ноль».
- **Correction:** для meaningful negative uncertainty должна исключать заранее
  значимый ε.
- **Invariant:** absence of significance ≠ absence of effect.
- **Tripwire:** «CI crosses zero, therefore no effect».

## W. Monotonic estimator становится market law
- **Temptation:** описать гладкость/монотонность как свойство рынка.
- **Why it looked reasonable:** кривая гладкая и монотонная.
- **What was actually wrong:** это ограничение модели, пока не установлено отдельно.
- **G3 case:** 090 — монотонность baseline явно не навязывалась как закон.
- **Correction:** архитектурное предположение estimator не выдавать за discovered
  property.
- **Invariant:** estimator assumption ≠ market property.
- **Tripwire:** «monotone, so the market law is monotone».

## X. Selected-cell CI становится proof of edge
- **Temptation:** «дневной bootstrap внутри выбранной клетки положителен → edge».
- **Why it looked reasonable:** CI узкий и исключает ноль.
- **What was actually wrong:** он описывает dependence внутри клетки, не корректирует
  selection из search-map.
- **G3 case:** 086/091 — клетка `u+ k- ttc- r-` отобрана из карты; day-clustered CI
  этого не исправляет.
- **Correction:** формулировать `positive historical economics of the selected
  candidate`, не `selection-adjusted proof of edge`.
- **Invariant:** within-cell dependence CI ≠ selection correction.
- **Tripwire:** «bootstrap CI excludes zero, so the edge is proven».

## Y. Costs subtract twice
- **Temptation:** сравнить `net = gross - cost` снова с cost threshold.
- **Why it looked reasonable:** «надо же покрыть расходы».
- **What was actually wrong:** расходы вычтены дважды.
- **G3 case:** 091 correction — net_E сравнивается с нулём, не снова с 1,00 pt.
- **Correction:** после вычета cost порог — ноль.
- **Invariant:** subtract cost once; compare net to zero.
- **Tripwire:** «net is +8.97, still below the 1-point hurdle».

## Z. Weak holdout становится «nothing can be said»
- **Temptation:** объявить недомощный holdout полностью неинформативным.
- **Why it looked reasonable:** знак net не разрешён, CI огромный.
- **What was actually wrong:** другие величины могут дать жёсткое ограничение.
- **G3 case:** 091 — 84/290 target; даже все unknown→target = 59,0% < search 82,7%;
  цензура не объясняет всю просадку частоты.
- **Correction:** primary economic claim unresolved, но best-case frequency bound
  информативен.
- **Invariant:** `primary claim unresolved` ≠ `sample has no information`.
- **Tripwire:** «holdout is underpowered, so it says nothing».

## AA. Cross-instrument transport подменяет temporal persistence
- **Temptation:** проверить ES/YM и назвать это тестом устойчивости NQ.
- **Why it looked reasonable:** «перенос же».
- **What was actually wrong:** это разные вопросы.
- **G3 case:** 091 — ES/YM = instrument transport; новый NQ = temporal persistence.
- **Correction:** различать temporal transfer, instrument transport, historical
  replication, live forward evidence; один не спасает другой.
- **Invariant:** instrument transport ≠ temporal persistence.
- **Tripwire:** «NQ failed forward, let's confirm on ES».

## AB. Independent scenes магически становятся трейдером
- **Temptation:** положительный scene-level expectancy принять за торговую систему.
- **Why it looked reasonable:** сцены прибыльны в среднем.
- **What was actually wrong:** не определены concurrency/capital/overlap/re-entry/
  sizing/skipped.
- **G3 case:** 091-TEMPORAL — single-unit action policy введена явно.
- **Correction:** до forward validation — complete deterministic action policy.
- **Invariant:** scene expectancy ≠ trader.
- **Tripwire:** «average per-scene P&L is positive, so the system works».

## AC. Portfolio dependence игнорируется после overlap-policy
- **Temptation:** считать взятые сделки независимыми строками после single-unit.
- **Why it looked reasonable:** привычный per-trade bootstrap.
- **What was actually wrong:** решение пропустить сигнал зависит от длительности
  предыдущей позиции.
- **G3 case:** 091-TEMPORAL — inference на реализованном sequential дневном
  action-stream.
- **Correction:** bootstrap по дневному action-stream, не по взятым сделкам.
- **Invariant:** overlap-filtered decisions не независимы.
- **Tripwire:** «bootstrap the taken trades independently».

## AD. Unknown intrabar ordering разрешается в удобную сторону
- **Temptation:** придумать порядок high/low внутри минуты, если оба уровня в баре.
- **Why it looked reasonable:** «обычно сначала идёт …».
- **What was actually wrong:** OHLC не знает внутрибарный порядок.
- **G3 case:** 086/091 — same_bar = unknown; 091 — favorable/adverse bounds.
- **Correction:** unknown остаётся unknown либо получает predeclared bounds.
- **Invariant:** intrabar order не восстановим из минутного OHLC.
- **Tripwire:** «assume target hit first within the bar».

## AE. Structural price принимается за executable fill
- **Temptation:** считать ожидание от `close(q_event)` торговым результатом.
- **Why it looked reasonable:** структурное ожидание положительно.
- **What was actually wrong:** recognition завершается на этом закрытии; вход туда —
  hindsight.
- **G3 case:** 086→091 — entry = `open(q_event+1)`; give-up, missed, adverse gap, costs
  проверены; give-up ≈ 0.
- **Correction:** всегда проверять first honestly executable price и give-up.
- **Invariant:** structural reference price ≠ executable entry.
- **Tripwire:** «measured from close(signal), so that's the P&L».

## AF. Источник с тем же тикером считается эквивалентным
- **Temptation:** «`NQ 1m` по названию — та же лента».
- **Why it looked reasonable:** тикер и таймфрейм совпадают.
- **What was actually wrong:** сессии/roll/контракт/OHLC могут быть другими.
- **G3 case:** внешний GitHub `NQ_1m` — RTH/equity-сессия (405 мин/день против 1380),
  сдвиг цен ~281 пт, 0% совпадений OHLC на overlap → REJECT
  ([`base/091/source_gate.json`](../base/091/source_gate.json)).
- **Correction:** compatibility gate на известном overlap до открытия новой территории:
  timezone, sessions, rolls, OHLC, event reproduction.
- **Invariant:** external data должна воспроизвести существующий operator на overlap до
  открытия outcome-территории.
- **Tripwire:** «same ticker, so it's the same market».

## AG. Negative result переобобщается за пределы протестированного оператора
- **Temptation:** из узкого нуля вывести «history doesn't matter / market is random».
- **Why it looked reasonable:** результат отрицательный и аккуратный.
- **What was actually wrong:** отброшены representation, observation, outcome,
  estimator, resolution.
- **G3 case:** 076/082/083/088 — каждый ноль ограничен проверенной конструкцией, не
  всем RIZ/лентой.
- **Correction:** форма negative result — `not found X · under representation R · at
  observation P · for outcome Y · with estimator E · at resolution D`.
- **Invariant:** negative knowledge is bounded knowledge. Запрещены скачки «history
  does not matter / market is Markov / RIZ is useless / no mechanism / price is random».
- **Tripwire:** «therefore the market is random / RIZ is useless».

---

## Audit interventions that materially changed G3

Независимый концептуальный аудит — не церемониальная проверка после работы, а
механизм отлова категориальных ошибок, которые локально-связный агент систематически
не видит. Не создавать обязательный бюрократический аудит после каждого скрипта.
Независимый аудит особенно нужен перед: **canonization of population; freeze; opening
holdout; upgrading structural result to trade; upgrading historical candidate to
transferable claim.**

Случаи, где внешний взгляд поймал то, чего текущий агент не видел:

- **Film-1 population audit** — членство префиксное, не по достигнутому терминалу (B).
- **Future-conditioned canonicalization 089** — представитель по будущей `life`,
  отменён в 090 (C).
- **Censoring → zero** — последний event ≠ терминал; введён `CONT*` (D).
- **Resolved-only estimand** — исключение цензуры меняет claim (E).
- **Future-dependent test filtering** — purge к train, не к test (F).
- **q / recognition identity** — информация после q сдвигает recognition (G).
- **Semantic reducibility vs predictive gain** — «не добавил» ≠ «поглощён» (K).
- **Model miscalibration vs market residual** — остаток как локатор (I).
- **Calibration ≠ sufficiency** (J).
- **Informational novelty ≠ economic usefulness** — деньги не ждут новой информации (N).
- **Holdout best-case frequency bound** — недомощный holdout всё же информативен (Z).
- **Cost double counting** — сравнение net с нулём (Y).
- **Selected-cell CI vs data snooping** — historical economics, не proof (X).
- **Scene-level candidate → single-unit trader** — complete action policy (AB, AC).
- **Same-ticker external data** — compatibility gate до новой территории (AF).

---

## Agent tripwires

Если агент пишет одну из фраз ниже, он обязан остановиться и назвать: **population,
prefix, estimand, support, censoring, selection, resolution.** Это не запрет слов, а
audit trigger.

- «absorbed by» / «reducible to»
- «explained completely by»
- «therefore no additional information»
- «this proves» / «edge» (как установленный факт)
- «the mechanism is»
- «independent observations»
- «holdout says nothing»
- «we can rescue» / «let's try another threshold / horizon»
- «same state, different outcome proves a hidden variable»
- «test set should exclude long / unresolved scenes»
- «last observed event = failure»
- «same ticker, same market»
- «net is positive, but below the cost hurdle» (двойной вычет)

---

## Cold-agent failure-mode check

До нового массового research холодный агент должен ответить. Не может ответить —
массовый прогон преждевременен.

1. Может ли membership моей population зависеть от future?
2. Какая именно минута recognition?
3. Не использую ли я post-recognition movement как predictor?
4. Что означает censoring в моём outcome?
5. Если я удаляю unresolved, какой estimand я теперь оцениваю?
6. Мой residual — market residual или estimator residual?
7. Мой feature физически новый или re-encoding уже объявленного?
8. Моя finding про speed, direction или economics?
9. Есть ли движение после honest recognition?
10. Entry реально исполним по доступной цене?
11. Какая единица evidence (строка, курсор, акт, сцена, день)?
12. Как selection кандидата учтён в силе claim?
13. Что именно сможет опровергнуть эту frozen version?
14. Если результат отрицательный — что я обязан закрыть и что НЕ имею права обобщить?

---

## Самый важный принцип сохранения

Не переписывай историю G3 как будто исследование линейно и разумно пришло к 091. Оно
не было линейным. Сила методики возникла потому, что мы многократно неправильно
ставили вопрос, слишком сильно интерпретировали результат, путали observation с
outcome, описание с механизмом, механизм с trade, model failure с market residual,
пытались продолжить отрицательную ветку слишком далеко — и затем это исправляли. Не
очищай эту историю. Канонизируй исправления. Следующий агент должен наследовать не
только ответы, но и способность замечать, когда собственное рассуждение стало
правдоподобным, аккуратным — и всё равно неправильным.
