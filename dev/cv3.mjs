// Compare a scale-weighted kNN against a per-hour weighted ridge on the comparable launches.
// Run: node dev/cv3.mjs
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
    return { name: x.name, at, p1, pre, j: ln(p1 / Math.max(1, pre)), s: ln(p1) };
  })
  .filter((p) => p.p1 > 0 && p.pre > 0);

const pb = (tau, y, q) => {
  const d = ln(Math.max(1, y)) - ln(Math.max(1, q));
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

const weightedQuantile = (rows, p) => {
  const rs = rows.slice().sort((a, b) => a.v - b.v);
  const W = rs.reduce((s, r) => s + r.w, 0);
  let acc = 0;
  for (const r of rs) {
    acc += r.w / W;
    if (acc >= p) return r.v;
  }
  return rs[rs.length - 1].v;
};

function knn(target, pool, o) {
  const out = new Map();
  for (const h of HOURS) {
    const rows = [];
    for (const cc of pool) {
      const v = cc.at(h);
      if (v == null || v <= 0) continue;
      const ds = (cc.s - target.s) / o.sT;
      const dj = (cc.j - target.j) / o.jT;
      rows.push({ v: ln(v) - ln(cc.p1), w: Math.exp(-(ds * ds + dj * dj) / 2) });
    }
    if (rows.length < 4) return null;
    const med = weightedQuantile(rows, 0.5) + o.bias;
    const sig = Math.max(o.sigFloor, (weightedQuantile(rows, 0.75) - weightedQuantile(rows, 0.25)) / 1.349) * o.inflate;
    out.set(h, G.map((z) => Math.exp(target.s + med + z * sig)));
  }
  return out;
}

// r_h = a + b * (ln p1/pre) fitted per hour, weighted by scale similarity, ridge on the slope.
function ridge(target, pool, o) {
  const out = new Map();
  for (const h of HOURS) {
    const rows = [];
    for (const cc of pool) {
      const v = cc.at(h);
      if (v == null || v <= 0) continue;
      const ds = (cc.s - target.s) / o.sT;
      rows.push({ x: cc.j - target.j, y: ln(v) - ln(cc.p1), w: Math.exp(-(ds * ds) / 2) });
    }
    if (rows.length < 5) return null;
    const W = rows.reduce((s, r) => s + r.w, 0);
    let sxx = 0;
    let sxy = 0;
    for (const r of rows) {
      sxx += r.w * r.x * r.x;
      sxy += r.w * r.x * r.y;
    }
    const b = sxy / (sxx + o.lam);
    const a = rows.reduce((s, r) => s + r.w * (r.y - b * r.x), 0) / W;
    const res = rows.map((r) => ({ v: r.y - (a + b * r.x), w: r.w }));
    const sig = Math.max(o.sigFloor, (weightedQuantile(res, 0.75) - weightedQuantile(res, 0.25)) / 1.349) * o.inflate;
    out.set(h, G.map((z) => Math.exp(target.s + a + z * sig)));
  }
  return out;
}

function evaluate(make, o, subset) {
  let s = 0;
  let n = 0;
  for (const t of comps) {
    if (subset === 'big' && t.p1 < 100000) continue;
    if (subset === 'mid' && !(t.p1 >= 20000 && t.p1 < 100000)) continue;
    const p = make(t, comps.filter((x) => x !== t), o);
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
for (const sT of [0.4, 0.6, 0.9]) {
  for (const jT of [2, 5]) {
    for (const bias of [-0.05, 0]) {
      for (const inflate of [1.0, 1.1]) {
        for (const sigFloor of [0.12, 0.2]) {
          const o = { sT, jT, bias, inflate, sigFloor };
          rows.push({ kind: 'knn', ...o, big: evaluate(knn, o, 'big'), mid: evaluate(knn, o, 'mid') });
        }
      }
    }
  }
  for (const lam of [0.3, 1, 4]) {
    for (const inflate of [1.0, 1.1]) {
      for (const sigFloor of [0.12, 0.2]) {
        const o = { sT, lam, inflate, sigFloor };
        rows.push({ kind: 'ridge', ...o, big: evaluate(ridge, o, 'big'), mid: evaluate(ridge, o, 'mid') });
      }
    }
  }
}
rows.sort((a, b) => a.big - b.big);
console.log('big     mid    kind  sT   jT/lam bias inflate floor');
for (const r of rows.slice(0, 20)) {
  console.log(
    `${r.big.toFixed(4)} ${r.mid.toFixed(4)}  ${r.kind.padEnd(5)} ${String(r.sT).padEnd(4)} ` +
      `${String(r.jT ?? r.lam).padEnd(6)} ${String(r.bias ?? '-').padEnd(4)} ${String(r.inflate).padEnd(7)} ${r.sigFloor}`
  );
}
