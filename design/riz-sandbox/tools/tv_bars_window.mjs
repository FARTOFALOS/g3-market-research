import { uiEvaluate } from 'file:///C:/Users/Admin/Claude/tradingview-mcp/src/core/ui.js';
const from = Number(process.argv[2]), to = Number(process.argv[3]);
const js = `
(function() {
  var bars = window.TradingViewApi._activeChartWidgetWV.value()._chartWidget.model().mainSeries().bars();
  if (!bars || !bars.lastIndex) return { error: 'no bars' };
  var fi = bars.firstIndex(), li = bars.lastIndex();
  var out = [], f0 = null, l0 = null;
  for (var i = fi; i <= li; i++) {
    var v = bars.valueAt(i);
    if (!v) continue;
    if (f0 === null) f0 = v[0];
    l0 = v[0];
    if (v[0] >= ${from} && v[0] <= ${to}) out.push({ t: v[0], o: v[1], h: v[2], l: v[3], c: v[4] });
  }
  return { total: bars.size(), firstTime: f0, lastTime: l0, matched: out.length, bars: out };
})()
`;
const r = (await uiEvaluate({ expression: js })).result;
console.log(JSON.stringify(r));
process.exit(0);
