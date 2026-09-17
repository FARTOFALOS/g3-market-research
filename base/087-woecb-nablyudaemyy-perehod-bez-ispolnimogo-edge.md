# 087 — WOECB′: наблюдаемый переход процесса воспроизводим, исполнимого edge на recognition нет

**Статус: `ESTABLISHED — reproducible observable process transition; executable edge NOT established.`**
Не «WOECB′ не имеет предсказуемости» и не «edge в процессе нет» — это не доказано.

- **вывод:** держится.
  - **WOECB′ is established as a reproducible observable process transition on NQ development and
    preregistered NQ evaluation. Its child MB-resume structure also replicates. Under the frozen
    recognition rule (second MB-resume) and frozen symmetric future-path measurement, no residual
    directional asymmetry usable as an executable edge was established.**
  - **Переход.** От общего родителя «последнее outward-обновление = joint» ветвь WOECB′ и ветвь
    joint не реконвергируют за 1–2 события; дальнейшая грамматика держится ~10 баров. Внутри
    WOECB′ воспроизводится child-развилка: возобновление body-front (MB-resume).
  - **Нет edge.** От `close recognition` (2-й MB-resume) остаточный путь симметричен:
    per-film `path_out>path_in` 0,486 (dev) / 0,479 (eval), median(path_out−path_in) 0,00σ / −0,28σ.
    Направленной ещё не реализованной структуры на момент честного распознавания не найдено.
- **на чём стоит:** [`FREEZE_087`](087/FREEZE_087.md) — оператор, метрики, тесты 3→2→1, eval.
  Код рядом: `base/087/geom_reader.py` (геометрия + WOECB′ readings),
  `continuation_test.py` (ветви от joint-родителя), `continuation_grammar.py` (persistence),
  `woecb_child_measure.py` (MB-resume event/recog, τ), `woecb_residual.py` (симметрия пути).
- **чем подтверждено:** предрегистрированный NQ evaluation (2019–2026), тот же frozen-код, ноль
  подгонки, вход `films_NQ_evaluation`. Оба заранее объявленных утверждения PASS:
  - Process: MB-resume ever 0,659 (dev 0,661), confirmed-2nd 0,895 (0,873), persistence s10 TV
    0,247 (0,281), no-resume→b 0,97.
  - Symmetry: per-film path_out>path_in 0,479 (0,486), median разница −0,28σ (0,00; порог ±0,3σ).
  - Порядковый след усилился (out_first 0,71 против 0,64) — величины при этом симметричны.
- **относительно объекта:** живая наружная RIZ-сцена; наружные координаты, фронты `M=max W_out`,
  `MB=max B_out`. Единица detection — object-scene (свой terminal); единица evidence —
  физический акт (абсолютный бар + сторона), дедуп убирал 36–84 % копий на разных срезах.
- **территория:** NQ development 2006–2018 (discovery), NQ evaluation 2019–2026 (preregistered).
  Все часы, все ТФ. ES/YM не гонялись.
- **откуда вопрос:** бриф «Minimal Transition Structure of the RIZ Scene» (2026-09-16):
  становится ли одна общая RIZ-сцена наблюдаемо разными процессами и достаточно ли рано.

## Оператор (frozen)

- **WOECB′** (parent transition): последнее outward-обновление = **joint** (`M↑` и `MB↑` вместе) →
  текущее = **M-only** (`W_out>M_prev`, `B_out≤MB_prev`) с телом **внутрь** (`Co<Oo`).
  T0-независим, без порога величины. Три структурные грани, выведены чтением реальных фильмов;
  `Co<Oo` совпадает с close-back (внутри inward-body held 0,6 %).
- **child-событие MB-resume** = первый пост-WOECB′ бар с `B_out>MB_reco`. `EVENT`=1-й resume,
  **`RECOG`=2-й resume** (conservative recognition candidate, hold-подтверждение). Симметричное
  правило «нет resume → возврат» НЕ заморожено (отсутствие — не событие конкретной минуты).
- **Метрика остатка (frozen):** от `close recognition` симметрично `path_out`/`path_in` в σ,
  их порядок и время; движение, сформировавшее recognition, не кредитуется.

## NOT ESTABLISHED

- исполнимый entry/stop/exit;
- отсутствие предсказуемости вообще;
- отсутствие более раннего directional information (до WOECB′);
- универсальность за пределами NQ (ES/YM не проверялись);
- terminal probabilities при структурной цензуре (длинные extenders режутся сессионными
  разрывами: censored 186 баров / 5,7σ против reached-B 13 / 0,4σ; `P(B eventually)` не оценена).

## CLOSED / DO NOT REOPEN WITHOUT NEW INDEPENDENT REASON

- подбор другого `clo_pos`;
- изменение количества resume в определении recognition;
- перебор recognition delay;
- другие горизонты после того же recognition;
- спасение через `out_first`;
- повторный поиск directional edge внутри уже замороженной WOECB′-семьи только потому, что
  основной результат отрицательный.

## Статус объекта

**WOECB′ frozen as established knowledge. It may be used as context, parent state, landmark, or
comparison object in later research. It is not an active discovery target unless new independent
evidence creates a materially different question.** Если будущий более ранний механизм
естественно объяснит, почему WOECB′ возникает, вернуться к нему можно — но с новым вопросом,
а не чтобы заново добыть edge из той же конструкции.

## Guardrail

**A future research branch must have an independently stated market question. “Recovering edge
from WOECB′” is not sufficient justification for reopening discovery.**

## Повтор расчёта

numpy, pandas, numba; `data/market/NQ`, `work/081a/paths/films_NQ_{discovery,evaluation}.parquet`,
`base/084` (race084). Из `base/087/`, вход — территория аргументом:

    python -B woecb_child_measure.py discovery      # и evaluation
    python -B woecb_residual.py discovery           # и evaluation
    python -B continuation_grammar.py discovery      # и evaluation

## Открытый вопрос (новый самостоятельный branch, начинать заново)

**В общей живой RIZ-сцене ДО WOECB′ существуют ли наблюдаемые изменения процесса, после которых
ранее сходные истории начинают воспроизводимо развиваться различно, пока существенная часть
будущего пути ещё не реализована ценой?** WOECB′ используется только как временной landmark —
известная поздняя граница, к которой полезная directional structure уже не обнаруживается.
Порядок заново: реальные полные pre-WOECB′ фильмы → нейтральное описание развития →
повторяющееся различие → минимальный transition → prefix recognition → remaining path. Если
новый transition окажется не связан с WOECB′ — это допустимо; задача не объяснить старый
отрицательный результат, а узнать рынок.

## Связано

- продолжает [084](084-posle-close-break-asimmetriya-ne-ustanovlena.md) (язык событий, `close_break`)
  и [086](086-strukturnaya-vilka-close-break-kandidat-bez-perenosa.md) (вилка сцены);
- согласуется с [076](076-scena-ischerpyvaetsya-polozheniem-i-lokalnoy-sigmoy.md) (сцена
  исчерпывается положением и локальной σ), но получено другим оператором и популяцией — не повтор.
