// Extract the sandbox artboard's presets to JSON, and run the artboard's OWN
// machine over each one. Pair with verify_presets.py, which replays the same
// presets through the independent Python port: two implementations agreeing is
// the check that matters. One implementation checking itself proves nothing.
//
//   node tools/dump_presets.mjs > tools/presets.json
//
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '..', 'Main.dc.html'), 'utf8');

const m = html.match(/<script data-dc-script[^>]*>([\s\S]*?)<\/script>/);
if (!m) { console.error('no <script data-dc-script> block in Main.dc.html'); process.exit(1); }

// Minimal stand-in for the editor runtime's base class.
const shim = 'class DCLogic { constructor(p) { this.props = p || {}; } '
           + 'setState(o) { Object.assign(this.state, o); } }\n';

const Component = new Function(shim + m[1] + '\nreturn Component;')();
const c = new Component({ tick: 0.25, maxBars: 40 });
const presets = c.presets();

const out = {};
for (const key of Object.keys(presets)) {
  const p = presets[key];
  const inst = new Component({ tick: 0.25, maxBars: 40 });
  const bars = inst.toRows(p.bars).map((r) => ({
    o: Number(r.o), h: Number(r.h), l: Number(r.l), c: Number(r.c), t: r.t,
  }));
  const res = inst.machine(bars);
  out[key] = {
    label: p.label,
    bars: p.bars,
    zones: res.zones.map((z) => ({
      t: z.t, b: z.b, bull: z.bull, nn: z.nn, ss: z.ss, ti: z.ti,
      t3: z.t3, nx: z.nx, born: z.born, dead: z.dead,
      blue: inst.isBlue(z),
    })),
    events: res.log.map((e) => ({ i: e.i, kind: e.kind, text: e.text })),
  };
}
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
