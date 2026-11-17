// v2 of the model search: pooled regression and weighted kNN, scored per subset (big launches matter most).
// Run: node dev/cv2.mjs
import { readFileSync } from 'node:fs';

const dir = process.argv[2] ?? '../gta6-twitch-ccu-forecast-01-kit/input';
const case_ = JSON.parse(readFileSync(`${dir}/p00.in`, 'utf8'));
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const GAUSS = [-1.6449, -0.6745, 0, 0.6745, 1.6449];
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
const ln = Math.log;

const hourOf = (iso, dayStart) => Math.round((Date.parse(iso) - Date.parse(dayStart)) / 3600e3);

function profile(comp) {
  const series = new Map();
  for (const p of comp.hourly) series.set(hourOf(p.from, comp.day_start), p.viewers);
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
    const jump = ln(Math.max(1, b) / Math.max(1, a));
    if (jump > best) {
      best = jump;
      launch = h;
    }
  }
  return { name: comp.name, at, p1, pre: preLevel, launch, jump: best };
}

const comps = case_.comparables
  .filter((c) => !/^Grand Theft Auto VI/.test(c.name))
  .map(profile)
  .filter((p) => p.p1 > 0 && p.pre > 0 && p.launch != null);

const feats = (p) => ({ scale: ln(p.p1), jump: ln(p.p1 / Math.max(1, p.pre)), launch: p.launch });

const pinball = (tau, y, q) => {
  const d = ln(Math.max(1, y)) - ln(Math.max(1, q));
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

function evalForecasts(forecasts, subsetNames) {
  const sel = forecasts.filter((f) => subsetNames.includes(f.name));
  let s = 0;
  let n = 0;
  for (const f of sel) {
    for (const [h, qs] of f.pred) {
      const y = f.truth.get(h);
      if (y == null) continue;
      for (let i = 0; i < 5; i++) {
        s += pinball(LEVELS[i], y, qs[i]);
        n++;
      }
    }
  }
  return n ? s / n : NaN;
}

// ---------------------------------------------------------------- weighted kNN with gaussian or empirical spread

function knn(target, pool, o) {
  const a = feats(target);
  const out = new Map();
  for (const h of HOURS) {
    const rows = [];
    for (const c of pool) {
      const hh = h + (c.launch - a.launch);
      if (hh < 1 || hh > 27) continue;
      const v = c.at(hh);
      if (v == null || v <= 0) continue;
      const d = Math.hypot(
        (feats(c).scale - a.scale) / o.scaleTol,
        (feats(c).jump - a.jump) / o.jumpTol,
        (feats(c).launch - a.launch) / o.launchTol
      );
      rows.push({ r: ln(v) - ln(c.p1), w: Math.exp(-(d * d) / 2) });
    }
    if (rows.length < 3) return null;
    rows.sort((x, y) => x.r - y.r);
    const W = rows.reduce((s, x) => s + x.w, 0);
    const cdf = rows.map((x) => {
      x.c = x.w / W;
      return x;
    });
    const wq = (p) => {
      let acc = 0;
      for (const x of cdf) {
        acc += x.c;
        if (acc >= p) return x.r;
      }
      return cdf[cdf.length - 1].r;
    };
    const med = wq(0.5);
    const qs = [];
    if (o.spread === 'gauss') {
      // weighted IQR -> sigma
      const sig = Math.max(0.08, (wq(0.75) - wq(0.25)) / 1.349) * o.inflate;
      for (const z of GAUSS) qs.push(Math.exp(ln(target.p1) + med + z * sig));
    } else {
      for (const p of LEVELS) qs.push(Math.exp(ln(target.p1) + med + (wq(p) - med) * o.inflate));
    }
    out.set(h, qs);
  }
  return out;
}

// ---------------------------------------------------------------- pooled ridge: ln V(h)/p1 = a_h + b1*scale + b2*jump + b3*(launch - launch_t)

function ridge(target, pool, o) {
  // Build one design row per comparable-hour, then solve normal equations with ridge.
  const a = feats(target);
  const rowsByHour = new Map();
  for (const h of HOURS) rowsByHour.set(h, []);
  for (const c of pool) {
    const f = feats(c);
    for (const h of HOURS) {
      const hh = h + (c.launch - a.launch);
      if (hh < 1 || hh > 27) continue;
      const v = c.at(hh);
      if (v == null || v <= 0) continue;
      rowsByHour.get(h).push({ y: ln(v) - ln(c.p1), x: [f.scale - a.scale, f.jump - a.jump, f.launch - a.launch] });
    }
  }
  const out = new Map();
  const sigmas = [];
  for (const h of HOURS) {
    const rows = rowsByHour.get(h);
    if (rows.length < 6) return null;
    // features: intercept, scale, jump, launch-offset
    const d = 4;
    const A = Array.from({ length: d }, () => new Array(d).fill(0));
    const b = new Array(d).fill(0);
    for (const r of rows) {
      const xv = [1, ...r.x];
      for (let i = 0; i < d; i++) {
        b[i] += xv[i] * r.y;
        for (let j = 0; j < d; j++) A[i][j] += xv[i] * xv[j];
      }
    }
    for (let i = 1; i < d; i++) A[i][i] += o.ridge;
    // gaussian elimination
    for (let i = 0; i < d; i++) {
      let p = i;
      for (let k = i + 1; k < d; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
      [A[i], A[p]] = [A[p], A[i]];
      [b[i], b[p]] = [b[p], b[i]];
      if (Math.abs(A[i][i]) < 1e-9) return null;
      for (let k = i + 1; k < d; k++) {
        const f = A[k][i] / A[i][i];
        for (let j = i; j < d; j++) A[k][j] -= f * A[i][j];
        b[k] -= f * b[i];
      }
    }
    const beta = new Array(d).fill(0);
    for (let i = d - 1; i >= 0; i--) {
      let s = b[i];
      for (let j = i + 1; j < d; j++) s -= A[i][j] * beta[j];
      beta[i] = s / A[i][i];
    }
    const xv = [1, 0, 0, 0]; // target's features are centred at zero
    const med = xv.reduce((s, x, i) => s + x * beta[i], 0);
    const resid = rows.map((r) => {
      const pred = [1, ...r.x].reduce((s, x, i) => s + x * beta[i], 0);
      return r.y - pred;
    });
    resid.sort((x, y) => x - y);
    const q = (p) => resid[Math.min(resid.length - 1, Math.max(0, Math.round((resid.length - 1) * p)))];
    const sig = Math.max(0.08, (q(0.75) - q(0.25)) / 1.349) * o.inflate;
    sigmas.push(sig);
    out.set(h, GAUSS.map((z) => Math.exp(ln(target.p1) + med + z * sig)));
  }
  return out;
}

// ---------------------------------------------------------------- run

const big = comps.filter((c) => c.p1 >= 100000).map((c) => c.name);
const mid = comps.filter((c) => c.p1 >= 20000).map((c) => c.name);

function run(name, fn, opts) {
  const fc = [];
  for (const t of comps) {
    const pool = comps.filter((c) => c !== t);
    const pred = fn(t, pool, opts);
    if (!pred || pred.size !== 23) continue;
    fc.push({ name: t.name, truth: new Map(HOURS.map((h) => [h, t.at(h)])), pred });
  }
  return {
    name,
    all: evalForecasts(fc, comps.map((c) => c.name)),
    big: evalForecasts(fc, big),
    mid: evalForecasts(fc, mid),
    fc,
  };
}

const results = [];
for (const spread of ['empirical', 'gauss']) {
  for (const inflate of [0.85, 1.0, 1.15, 1.3]) {
    for (const scaleTol of [0.8, 1.4, 3]) {
      for (const jumpTol of [0.5, 0.9, 2]) {
        results.push(
          run(`knn ${spread} inflate=${inflate} sT=${scaleTol} jT=${jumpTol}`, knn, {
            spread,
            inflate,
            scaleTol,
            jumpTol,
            launchTol: 4,
          })
        );
      }
    }
  }
}
for (const ridge of [0.5, 2, 8]) {
  for (const inflate of [0.85, 1.0, 1.15]) {
    results.push(run(`ridge ${ridge} inflate=${inflate}`, ridge_, { ridge, inflate }));
  }
}

function ridge_(t, p, o) {
  return ridge(t, p, o);
}

// baseline
const baseFc = comps.map((t) => {
  const pred = new Map();
  for (const h of HOURS) pred.set(h, GAUSS.map((z) => Math.exp(ln(t.p1) + z * 0.15 * Math.sqrt(h - 1))));
  return { name: t.name, truth: new Map(HOURS.map((h) => [h, t.at(h)])), pred };
});
const baseRow = {
  name: 'baseline persistence',
  all: evalForecasts(baseFc, comps.map((c) => c.name)),
  big: evalForecasts(baseFc, big),
  mid: evalForecasts(baseFc, mid),
};
results.push(baseRow);

results.sort((a, b) => a.big - b.big);
console.log('big-subset rank (p1 >= 100k, ' + big.length + ' launches)');
console.log('   big     mid     all   model');
for (const r of results.slice(0, 14)) {
  console.log(`${r.big.toFixed(4)}  ${r.mid.toFixed(4)}  ${r.all.toFixed(4)}  ${r.name}`);
}
console.log('\nbaseline:', baseRow.big.toFixed(4), baseRow.mid.toFixed(4), baseRow.all.toFixed(4));
