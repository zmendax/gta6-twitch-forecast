// Does anchoring the ratio on a blend of hour 1 and the pre-launch level beat anchoring on hour 1 alone?
// anchor = exp(theta * ln(p1) + (1 - theta) * ln(pre)), the same blend on both sides of the ratio.
// Run: node dev/cv4.mjs
import { readFileSync } from 'node:fs';

const dir = process.argv[2] ?? '../gta6-twitch-ccu-forecast-01-kit/input';
const c = JSON.parse(readFileSync(`${dir}/p00.in`, 'utf8'));
const L = [0.05, 0.25, 0.5, 0.75, 0.95];
const G = [-1.6449, -0.6745, 0, 0.6745, 1.6449];
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
const ln = Math.log;
const ho = (iso, d) => Math.round((Date.parse(iso) - Date.parse(d)) / 3600e3);

const comps = c.comparables
  .filter((x) => !/^Grand Theft Auto VI/.test(x.name))
  .map((x) => {
    const s = new Map();
    for (const p of x.hourly) s.set(ho(p.from, x.day_start), p.viewers);
    const at = (h) => (s.has(h) ? s.get(h) : null);
    const p1 = at(1);
    const pr = [];
    for (let h = -6; h <= 0; h++) if (at(h) != null) pr.push(at(h));
    pr.sort((a, b) => a - b);
    const pre = pr.length ? pr[Math.floor(pr.length / 2)] : null;
    return { name: x.name, at, p1, pre };
  })
  .filter((p) => p.p1 > 0 && p.pre > 0);

const pb = (tau, y, q) => {
  const d = ln(Math.max(1, y)) - ln(Math.max(1, q));
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

const wq = (rows, p) => {
  const rs = rows.slice().sort((a, b) => a.v - b.v);
  const W = rs.reduce((s, r) => s + r.w, 0);
  let acc = 0;
  for (const r of rs) {
    acc += r.w / W;
    if (acc >= p) return r.v;
  }
  return rs[rs.length - 1].v;
};

const anchor = (p, theta) => ln(Math.max(1, p.p1)) * theta + ln(Math.max(1, p.pre)) * (1 - theta);

function predict(target, pool, o) {
  const out = new Map();
  const aT = anchor(target, o.theta);
  const scaleT = ln(target.p1);
  const jumpT = ln(target.p1 / Math.max(1, target.pre));
  for (const h of HOURS) {
    const rows = [];
    for (const cc of pool) {
      const v = cc.at(h);
      if (v == null || v <= 0) continue;
      const ds = (ln(cc.p1) - scaleT) / o.sT;
      const dj = (ln(cc.p1 / Math.max(1, cc.pre)) - jumpT) / o.jT;
      rows.push({ v: ln(v) - anchor(cc, o.theta), w: Math.exp(-(ds * ds + dj * dj) / 2) });
    }
    if (rows.length < 4) return null;
    const centre = wq(rows, 0.5) + o.bias;
    const spread = (wq(rows, 0.75) - wq(rows, 0.25)) / 1.349;
    const sigma = Math.max(o.sigFloor, spread) * o.inflate;
    out.set(h, G.map((z) => Math.exp(aT + centre + z * sigma)));
  }
  return out;
}

function evaluate(o, subset) {
  let s = 0;
  let n = 0;
  for (const t of comps) {
    if (subset === 'big' && t.p1 < 100000) continue;
    if (subset === 'mid' && !(t.p1 >= 20000 && t.p1 < 100000)) continue;
    const p = predict(t, comps.filter((x) => x !== t), o);
    if (!p) return NaN;
    for (const [h, qs] of p) {
      const y = t.at(h);
      if (y == null) continue;
      for (let i = 0; i < 5; i++) {
        s += pb(L[i], y, qs[i]);
        n++;
      }
    }
  }
  return s / n;
}

const rows = [];
for (const theta of [0, 0.25, 0.5, 0.75, 1]) {
  for (const sT of [0.6, 0.9]) {
    for (const inflate of [1.0, 1.1, 1.2]) {
      for (const sigFloor of [0.12, 0.2]) {
        const o = { theta, sT, jT: 3, bias: -0.05, inflate, sigFloor };
        rows.push({ ...o, big: evaluate(o, 'big'), mid: evaluate(o, 'mid') });
      }
    }
  }
}
rows.sort((a, b) => a.big - b.big);
console.log('big     mid    theta sT   inflate floor');
for (const r of rows.slice(0, 16)) {
  console.log(`${r.big.toFixed(4)} ${r.mid.toFixed(4)}  ${String(r.theta).padEnd(5)} ${String(r.sT).padEnd(4)} ${String(r.inflate).padEnd(7)} ${r.sigFloor}`);
}
