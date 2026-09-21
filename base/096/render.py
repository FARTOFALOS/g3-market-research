#!/usr/bin/env python3
"""096 render — локальная страница разметки 60 сцен. Наружу ничего не уходит.

Строит один самодостаточный HTML из work/096/gold60.json. Никаких сетевых
запросов, никаких внешних ресурсов. Прогресс живёт в localStorage браузера,
результат выгружается файлом.

Показывается (FREEZE_096_INSTRUMENT §5): свечи T0..q_event, линия собственной
границы b, линия бегущего наружного экстремума M, номера баров 0..k, пометка
последнего бара. Всё зеркалено south->north, шкала Y локальная.

Не показывается: дата, год, инструмент, абсолютная цена, riz_id, ТФ, значение k
надписью, u/ttc/r, что-либо после q_event, исход.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'

CRITERIA = {
    'isolated_interruption':
        'До последнего бара локальные откаты внутрь ВОССТАНАВЛИВАЮТСЯ наружным '
        'процессом; ранее занятая наружная территория не показывает накопленного '
        'разрушения; последний бар — первый качественно новый отказ.',
    'progressive_deterioration':
        'До последнего бара уже видно не просто откат внутрь, а последовательное '
        'ухудшение способности наружного процесса восстанавливать локальные '
        'нарушения и удерживать/расширять ранее занятую территорию; последний бар '
        'завершает уже видимое ухудшение.',
    'mixed_or_unclear':
        'Видимого префикса не хватает, чтобы честно выбрать одно из двух.',
}

HTML = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>096 — разметка сцен</title>
<style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8b93a1;--up:#4bbf73;--dn:#e05c5c;--line:#2a2f3a;
      --acc:#6aa9ff;--warn:#e0a85c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;border-bottom:1px solid var(--line)}
header b{font-size:15px}
#bar{flex:1;height:6px;background:#1b1f27;border-radius:3px;overflow:hidden}
#bar i{display:block;height:100%;background:var(--acc);width:0}
main{max-width:1080px;margin:0 auto;padding:18px}
#pane{background:#12151b;border:1px solid var(--line);border-radius:8px;padding:8px;
  min-height:300px;display:flex;align-items:center;justify-content:center;overflow:auto}
#chart{display:block}
.btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:16px 0 10px}
button.lab{padding:14px 12px;border-radius:8px;border:1px solid var(--line);background:#181c24;
  color:var(--fg);cursor:pointer;text-align:left;font:inherit}
button.lab:hover{border-color:var(--acc)}
button.lab.on{border-color:var(--acc);background:#1d2735}
button.lab kbd{display:inline-block;background:#262c37;border-radius:4px;padding:0 6px;margin-right:8px;
  color:var(--mut);font:12px monospace}
button.lab small{display:block;color:var(--mut);font-size:12px;margin-top:6px;line-height:1.35}
textarea{width:100%;min-height:64px;background:#12151b;color:var(--fg);border:1px solid var(--line);
  border-radius:8px;padding:10px;font:inherit;resize:vertical}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
.row button{padding:8px 14px;border-radius:7px;border:1px solid var(--line);background:#181c24;
  color:var(--fg);cursor:pointer;font:inherit}
.row button:hover{border-color:var(--acc)}
.row button.pri{background:#1d2735;border-color:var(--acc)}
.mut{color:var(--mut)}
.done{color:var(--up)}
#note-hint{color:var(--mut);font-size:13px;margin:14px 0 6px}
#warn{display:none;background:#3a2020;color:#ffd9d9;border-bottom:1px solid #6b3030;
  padding:9px 18px;font-size:13px}
#warn.on{display:block}
</style>
<div id="warn">Браузер не сохраняет прогресс автоматически. Нажимай <b>сохранить черновик</b> время от времени, а вернувшись — <b>продолжить с файла</b>.</div>
<header>
  <b>096 · разметка сцен</b>
  <span id="pos" class="mut"></span>
  <div id="bar"><i></i></div>
  <span id="cnt" class="mut"></span>
</header>
<main>
  <div id="pane"><svg id="chart"></svg></div>
  <div class="btns" id="btns"></div>
  <div id="note-hint">Заметка — свободно. Самое полезное: «лента этого не покажет», «вижу вот что», «сомневаюсь потому что».</div>
  <textarea id="note" placeholder="…"></textarea>
  <div class="row">
    <button id="prev">← назад</button>
    <button id="next">вперёд →</button>
    <span class="mut">клавиши: 1 / 2 / 3 — метка, ← → — навигация</span>
    <span style="flex:1"></span>
    <button id="draft">сохранить черновик</button>
    <label class="row" style="margin:0"><button id="loadbtn">продолжить с файла</button>
      <input id="load" type="file" accept="application/json" hidden></label>
    <button id="export" class="pri">выгрузить результат</button>
  </div>
  <div class="row"><span id="status" class="mut"></span></div>
</main>
<script>
const DATA = __DATA__;
const CRIT = __CRIT__;
const KEYS = Object.keys(CRIT);
const SC = {}; DATA.scenes.forEach(s => SC[s.id] = s);
const ORDER = DATA.order;
const LS = 'g3-096-labels-v1';
let STORAGE_OK = true;
try { localStorage.setItem(LS + '-probe', '1'); localStorage.removeItem(LS + '-probe'); }
catch (e) { STORAGE_OK = false; }
let st = {};
if (STORAGE_OK) { try { st = JSON.parse(localStorage.getItem(LS) || '{}'); } catch (e) { st = {}; } }
let i = 0;
const firstGap = ORDER.findIndex(id => !(st[id] && st[id].label));
if (firstGap > 0) i = firstGap;

const $ = s => document.querySelector(s);
const svg = $('#chart'), NS = 'http://www.w3.org/2000/svg';
const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };

// Постоянный масштаб по обеим осям: шаг бара и высота сигмы одинаковы во всех
// сценах, поэтому геометрия свечей (перекрытия, тела, откаты) сравнима между
// сценой из 3 баров и сценой из 37. Кадр подгоняется под сцену, а не наоборот.
const PITCH = 26, SIGMA_PX = 50, L = 54, R = 24, T = 26, B = 36;

function draw(s) {
  svg.textContent = '';
  const n = s.n;
  let lo = Math.min(0, s.M, ...s.l), hi = Math.max(0, s.M, ...s.h);
  const pad = 0.35; lo -= pad; hi += pad;
  const pw = n * PITCH, ph = (hi - lo) * SIGMA_PX;
  const W = L + pw + R, H = T + ph + B;
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const X = k => L + PITCH * (k + 0.5);
  const Y = v => T + ph * (hi - v) / (hi - lo);
  const cw = PITCH * 0.62;

  svg.appendChild(el('rect', {x: L, y: T, width: pw, height: ph, fill: 'none', stroke: '#2a2f3a'}));
  // эталон масштаба: одна сигма сцены
  svg.appendChild(el('line', {x1: L + pw + 9, y1: T + 6, x2: L + pw + 9, y2: T + 6 + SIGMA_PX,
    stroke: '#8b93a1', 'stroke-width': 1.2}));
  const ts = el('text', {x: L + pw + 13, y: T + 6 + SIGMA_PX / 2 + 4, fill: '#8b93a1', 'font-size': 11});
  ts.textContent = 'σ'; svg.appendChild(ts);
  // собственная граница b
  svg.appendChild(el('line', {x1: L, y1: Y(0), x2: L + pw, y2: Y(0), stroke: '#e0a85c',
    'stroke-width': 1.6, 'stroke-dasharray': '7 5'}));
  const tb = el('text', {x: L - 8, y: Y(0) + 4, fill: '#e0a85c', 'text-anchor': 'end', 'font-size': 13});
  tb.textContent = 'b'; svg.appendChild(tb);
  // бегущий наружный экстремум M
  svg.appendChild(el('line', {x1: L, y1: Y(s.M), x2: L + pw, y2: Y(s.M), stroke: '#6aa9ff',
    'stroke-width': 1.4, 'stroke-dasharray': '3 4'}));
  const tm = el('text', {x: L - 8, y: Y(s.M) + 4, fill: '#6aa9ff', 'text-anchor': 'end', 'font-size': 13});
  tm.textContent = 'M'; svg.appendChild(tm);

  for (let k = 0; k < n; k++) {
    const up = s.c[k] >= s.o[k], col = up ? '#4bbf73' : '#e05c5c';
    svg.appendChild(el('line', {x1: X(k), y1: Y(s.h[k]), x2: X(k), y2: Y(s.l[k]), stroke: col, 'stroke-width': 1.4}));
    const y1 = Y(Math.max(s.o[k], s.c[k])), y2 = Y(Math.min(s.o[k], s.c[k]));
    svg.appendChild(el('rect', {x: X(k) - cw / 2, y: y1, width: cw, height: Math.max(1.2, y2 - y1),
      fill: up ? '#4bbf73' : '#e05c5c', stroke: col}));
    if (n <= 40 || k % 2 === 0 || k === n - 1) {
      const t = el('text', {x: X(k), y: T + ph + 18, fill: '#8b93a1', 'text-anchor': 'middle', 'font-size': 11});
      t.textContent = k; svg.appendChild(t);
    }
  }
  // пометка последнего бара
  const lx = X(n - 1), ly = Y(Math.max(s.h[n - 1], s.o[n - 1], s.c[n - 1])) - 10;
  svg.appendChild(el('path', {d: `M${lx - 5},${ly - 9} L${lx + 5},${ly - 9} L${lx},${ly} Z`, fill: '#e6e6e6'}));
  const q = el('text', {x: lx, y: ly - 14, fill: '#e6e6e6', 'text-anchor': 'middle', 'font-size': 12});
  q.textContent = 'q_event'; svg.appendChild(q);
}

function render() {
  const id = ORDER[i], s = SC[id], cur = st[id] || {};
  draw(s);
  $('#pos').textContent = `${i + 1} / ${ORDER.length}`;
  const done = ORDER.filter(x => st[x] && st[x].label).length;
  $('#bar i').style.width = (100 * done / ORDER.length) + '%';
  $('#cnt').textContent = `размечено ${done}`;
  $('#cnt').className = done === ORDER.length ? 'done' : 'mut';
  document.querySelectorAll('button.lab').forEach(b =>
    b.classList.toggle('on', b.dataset.k === cur.label));
  $('#note').value = cur.note || '';
  $('#status').textContent = '';
}

function setLabel(k) {
  const id = ORDER[i];
  st[id] = Object.assign({}, st[id], {label: k, note: $('#note').value, ts: Date.now()});
  save(); render();
  if (i < ORDER.length - 1) setTimeout(() => { i++; render(); }, 140);
}
function save() {
  if (!STORAGE_OK) return;
  try { localStorage.setItem(LS, JSON.stringify(st)); } catch (e) { STORAGE_OK = false; flagWarn(); }
}
function flagWarn() { $('#warn').classList.toggle('on', !STORAGE_OK); }
function payload() {
  const done = ORDER.filter(x => st[x] && st[x].label).length;
  return {tool: 'g3-096-label', version: 1, saved_utc: new Date().toISOString(),
    n_scenes: ORDER.length, n_labelled: done,
    labels: ORDER.map(id => ({id, label: (st[id] || {}).label || null, note: (st[id] || {}).note || ''}))};
}
function download(obj, name) {
  const blob = new Blob([JSON.stringify(obj, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
}
function saveNote() {
  const id = ORDER[i];
  st[id] = Object.assign({}, st[id], {note: $('#note').value});
  save();
}

const bw = $('#btns');
KEYS.forEach((k, n) => {
  const b = document.createElement('button');
  b.className = 'lab'; b.dataset.k = k;
  b.innerHTML = `<kbd>${n + 1}</kbd><b>${k}</b><small>${CRIT[k]}</small>`;
  b.onclick = () => setLabel(k);
  bw.appendChild(b);
});
$('#prev').onclick = () => { saveNote(); if (i > 0) { i--; render(); } };
$('#next').onclick = () => { saveNote(); if (i < ORDER.length - 1) { i++; render(); } };
$('#note').addEventListener('blur', saveNote);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  if (e.key >= '1' && e.key <= '3') setLabel(KEYS[+e.key - 1]);
  if (e.key === 'ArrowLeft') $('#prev').click();
  if (e.key === 'ArrowRight') $('#next').click();
});
$('#draft').onclick = () => {
  saveNote(); const p = payload();
  download(p, 'gold60_labels_draft.json');
  $('#status').textContent = `черновик сохранён: ${p.n_labelled} из ${ORDER.length}`;
};
$('#loadbtn').onclick = e => { e.preventDefault(); $('#load').click(); };
$('#load').onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    try {
      const j = JSON.parse(r.result);
      if (!j.labels) throw new Error('не тот файл');
      j.labels.forEach(x => { if (x.label || x.note) st[x.id] = {label: x.label || undefined, note: x.note || ''}; });
      save(); i = Math.max(0, ORDER.findIndex(id => !(st[id] && st[id].label)));
      render(); $('#status').textContent = `загружено ${j.n_labelled || 0} меток`;
    } catch (e) { $('#status').textContent = 'файл не прочитался: ' + e.message; }
  };
  r.readAsText(f);
};
$('#export').onclick = () => {
  saveNote(); const p = payload();
  download(p, 'gold60_labels.json');
  $('#status').textContent = p.n_labelled < ORDER.length
    ? `выгружено ${p.n_labelled} из ${ORDER.length} — размечено не всё`
    : `выгружено все ${ORDER.length}`;
};
flagWarn();
render();
</script>
"""


def main():
    data = json.loads((OUT / 'gold60.json').read_text(encoding='utf-8'))
    html = (HTML.replace('__DATA__', json.dumps(data, ensure_ascii=False))
                .replace('__CRIT__', json.dumps(CRITERIA, ensure_ascii=False)))
    p = OUT / 'label60.html'
    p.write_text(html, encoding='utf-8')
    print(f'{p}  ({p.stat().st_size / 1024:.0f} KB, {len(data["scenes"])} сцен)')
    print('открыть в браузере; наружу ничего не уходит, прогресс в localStorage')


if __name__ == '__main__':
    main()
