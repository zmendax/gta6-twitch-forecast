// Diagnostics for the best kNN variant on the big launches: where does the loss come from?
// Run: node dev/diag.mjs
import { readFileSync } from 'node:fs';

const dir = process.argv[2] ?? '../gta6-twitch-ccu-forecast-01-kit/input';
const case_ = JSON.parse(readFileSync(`${dir}/p00.in`, 'utf8'));
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const GAUSS = [-1.6449, -0.6745, 0, 0.6745, 1.6449];
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
const ln = Math.log;
const hourOf = (iso, d) => Math.round((Date.parse(iso) - Date.parse(d)) / 3600e3);

const comps = case_.comparables
  .filter((c) => !/^Grand Theft Auto VI/.test(c.name))
  .map((c) => {
    const series = new Map();
    for (const p of c.hourly) series.set(hourOf(p.from, c.day_start), p.viewers);
    const at = (h) => (series.has(h) ? series.get(h) : null);
    const p1 = at(1);
    const pre = [];
    for (let h = -6; h <= 0; h++) if (at(h) != null) pre.push(at(h));
    pre.sort((a, b) => a - b);
    const preLevel = pre.length ? pre[Math.floor(pre.length / 2)] : null;
    let launch = null;
    let best = -Infinity;
    for (let h = -5; h <= 24; h++) {
      const a = at(h - 1);
      const b = at(h);
      if (a == null || b == null) continue;
      const j = ln(Math.max(1, b) / Math.max(1, a));
      if (j > best) {
        best = j;
        launch = h;
      }
    }
    return { name: c.name, at, p1, pre: preLevel, launch };
  })
  .filter((p) => p.p1 > 0 && p.pre > 0 && p.launch != null);

const feats = (p) => ({ scale: ln(p.p1), jump: ln(p.p1 / Math.max(1, p.pre)), launch: p.launch });

function predict(target, pool, o) {
  const a = feats(target);
  const out = new Map();
  for (const h of HOURS) {
    const rows = [];
    for (const c of pool) {
      const hh = h + (c.launch - a.launch);
      if (hh < 1 || hh > 27) continue;
      const v = c.at(hh);
      if (v == null || v <= 0) continue;
      const f = feats(c);
      const d = Math.hypot((f.scale - a.scale) / o.scaleTol, (f.jump - a.jump) / o.jumpTol, (f.launch - a.launch) / o.launchTol);
      rows.push({ r: ln(v) - ln(c.p1), w: Math.exp(-(d * d) / 2), name: c.name });
    }
    rows.sort((x, y) => x.r - y.r);
    const W = rows.reduce((s, x) => s + x.w, 0);
    const wq = (p) => {
      let acc = 0;
      for (const x of rows) {
        acc += x.w / W;
        if (acc >= p) return x.r;
      }
      return rows[rows.length - 1].r;
    };
    const med = wq(0.5);
    const sig = Math.max(0.08, (wq(0.75) - wq(0.25)) / 1.349) * o.inflate;
    out.set(h, { qs: GAUSS.map((z) => Math.exp(ln(target.p1) + med + z * sig)), med, sig });
  }
  return out;
}

const big = comps.filter((c) => c.p1 >= 100000);
const o = { scaleTol: 0.8, jumpTol: 2, launchTol: 4, inflate: 1.0 };
console.log('hour | n | bias(mean log err of median) | MAE median | sigma used | pinball | 90% cover');
let total = 0;
let count = 0;
for (const h of HOURS) {
  let bias = 0;
  let mae = 0;
  let sig = 0;
  let pb = 0;
  let cov = 0;
  let n = 0;
  for (const t of big) {
    const pred = predict(t, comps.filter((c) => c !== t), o).get(h);
    const y = t.at(h);
    if (y == null) continue;
    const e = ln(y) - (ln(t.p1) + pred.med);
    bias += e;
    mae += Math.abs(e);
    sig += pred.sig;
    let s = 0;
    for (let i = 0; i < 5; i++) {
      const d = ln(y) - ln(pred.qs[i]);
      s += 2 * Math.max(LEVELS[i] * d, (LEVELS[i] - 1) * d);
    }
    if (y >= pred.qs[0] && y <= pred.qs[4]) cov++;
    pb += s / 5;
    n++;
  }
  total += pb;
  count++;
  console.log(
    `${String(h).padStart(4)} | ${String(n).padStart(2)} | ${(bias / n).toFixed(3).padStart(6)} | ${(mae / n).toFixed(3).padStart(5)} | ${(sig / n).toFixed(2)} | ${(pb / n).toFixed(3)} | ${(100 * cov / n).toFixed(0)}%`
  );
}
console.log('mean per-hour pinball:', (total / count).toFixed(4));
