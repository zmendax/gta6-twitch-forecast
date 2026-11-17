// Cross-validate forecasting models on the kit's comparable launches.
// Each comparable is held out in turn and forecast from the other 37, scored with the contest's
// pinball loss on the log scale (RULES.txt, "Scoring").
// Run: node dev/cv.mjs [kit input dir]
import { readFileSync } from 'node:fs';

const dir = process.argv[2] ?? '../gta6-twitch-ccu-forecast-01-kit/input';
const case_ = JSON.parse(readFileSync(`${dir}/p00.in`, 'utf8'));
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const GAUSS = [-1.6449, -0.6745, 0, 0.6745, 1.6449];
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);

const hourOf = (iso, dayStart) => Math.round((Date.parse(iso) - Date.parse(dayStart)) / 3600e3);

function profile(comp) {
  const series = new Map();
  for (const p of comp.hourly) series.set(hourOf(p.from, comp.day_start), p.viewers);
  const at = (h) => (series.has(h) ? series.get(h) : null);
  const p1 = at(1);
  const prePts = [];
  for (let h = -6; h <= 0; h++) if (at(h) != null) prePts.push(at(h));
  prePts.sort((a, b) => a - b);
  const pre = prePts.length ? prePts[Math.floor(prePts.length / 2)] : null;
  let launch = null;
  let bestJump = -Infinity;
  for (let h = -5; h <= 24; h++) {
    const a = at(h - 1);
    const b = at(h);
    if (a == null || b == null) continue;
    const jump = Math.log(Math.max(1, b) / Math.max(1, a));
    if (jump > bestJump) {
      bestJump = jump;
      launch = h;
    }
  }
  return { name: comp.name, at, p1, pre, launch, jump: bestJump };
}

const comps = case_.comparables
  .filter((c) => !/^Grand Theft Auto VI/.test(c.name))
  .map(profile)
  .filter((p) => p.p1 > 0 && p.pre > 0);

// ---------------------------------------------------------------- helpers

const ln = Math.log;
const weightedQuantile = (items, p) => {
  // items: [{value, weight}]
  const xs = items.filter((i) => Number.isFinite(i.value) && i.weight > 0).sort((a, b) => a.value - b.value);
  if (!xs.length) return NaN;
  const total = xs.reduce((s, i) => s + i.weight, 0);
  let acc = 0;
  for (const i of xs) {
    acc += i.weight;
    if (acc >= p * total) return i.value;
  }
  return xs[xs.length - 1].value;
};

// Pinball loss of one item: 2 * max(tau * d, (tau - 1) * d), d = ln y - ln q
const pinball = (tau, y, q) => {
  const d = ln(Math.max(1, y)) - ln(Math.max(1, q));
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

function score(forecasts) {
  // forecasts: [{truth: Map(hour -> viewers), pred: Map(hour -> [5 quantiles])}]
  let sum = 0;
  let n = 0;
  const perCase = [];
  for (const { truth, pred } of forecasts) {
    let s = 0;
    let m = 0;
    for (const [h, qs] of pred) {
      const y = truth.get(h);
      if (y == null) continue;
      for (let i = 0; i < 5; i++) {
        s += pinball(LEVELS[i], y, qs[i]);
        m++;
      }
    }
    perCase.push({ loss: m ? s / m : NaN, n: m });
    sum += s;
    n += m;
  }
  return { loss: sum / n, perCase };
}

// ---------------------------------------------------------------- models

// features used for similarity
const features = (p) => ({
  scale: ln(Math.max(1, p.p1)),
  jump: ln(Math.max(1e-6, p.p1 / Math.max(1, p.pre))),
  launch: p.launch ?? 6,
});

function weightFor(target, other, opts) {
  const a = features(target);
  const b = features(other);
  const ds = (a.scale - b.scale) / opts.scaleTol;
  const dj = (a.jump - b.jump) / opts.jumpTol;
  const dl = (a.launch - b.launch) / opts.launchTol;
  const d2 = (ds * ds + dj * dj + dl * dl) / 3;
  return Math.exp(-d2);
}

// Model: ln V(h) = ln p1_target + weighted quantile over comparables of [ln V_k(h') - ln p1_k],
// with h' = h + (launch_k - launch_target). Options tune the similarity kernel, the spread inflation and
// whether the location is anchored on p1 or on pre.
function predict(target, pool, opts) {
  const L = target.launch ?? 6;
  const out = new Map();
  const baseA = ln(Math.max(1, target.p1));
  const baseB = ln(Math.max(1, target.pre));
  for (const h of HOURS) {
    const items = [];
    const raw = [];
    for (const o of pool) {
      const hh = h + ((o.launch ?? 6) - L);
      if (hh < 1 || hh > 27) continue;
      const v = o.at(hh);
      if (v == null || v <= 0) continue;
      const value = ln(v);
      const w = opts.uniform ? 1 : weightFor(target, o, opts);
      items.push({ value, weight: w });
      const r1 = value - ln(Math.max(1, o.p1));
      const r2 = value - ln(Math.max(1, o.pre));
      raw.push({ w, r1, r2 });
    }
    if (!items.length) continue;
    // location and quantiles on the anchored ratio
    const centre = weightedQuantile(
      raw.map((r) => ({ value: opts.anchor === 'pre' ? r.r2 : r.r1, weight: r.w })),
      0.5
    );
    const base = opts.anchor === 'pre' ? baseB : baseA;
    const qs = [];
    for (const tau of LEVELS) {
      const off = weightedQuantile(
        raw.map((r) => ({ value: (opts.anchor === 'pre' ? r.r2 : r.r1) - centre, weight: r.w })),
        tau
      );
      qs.push(Math.exp(base + centre + off * opts.spread));
    }
    out.set(h, qs);
  }
  return out;
}

// ---------------------------------------------------------------- run

const variants = [];
const baseOpts = { scaleTol: 1.2, jumpTol: 0.8, launchTol: 3, spread: 1, anchor: 'p1', uniform: false };
for (const uniform of [true, false]) {
  for (const anchor of ['p1', 'pre']) {
    for (const spread of [0.9, 1.0, 1.15, 1.3, 1.5]) {
      variants.push({ ...baseOpts, uniform, anchor, spread });
    }
  }
}

const results = [];
for (const opts of variants) {
  const forecasts = [];
  for (const target of comps) {
    const pool = comps.filter((c) => c !== target);
    const pred = predict(target, pool, opts);
    if (pred.size !== 23) continue;
    forecasts.push({ truth: new Map(HOURS.map((h) => [h, target.at(h)])), pred });
  }
  const { loss } = score(forecasts);
  results.push({ loss, ...opts });
}
results.sort((a, b) => a.loss - b.loss);
console.log('loss   anchor uniform spread');
for (const r of results.slice(0, 12)) {
  console.log(`${r.loss.toFixed(4)}  ${r.anchor.padEnd(6)} ${String(r.uniform).padEnd(7)} ${r.spread}`);
}

// Baseline: persistence at p1 with the house's random-walk spread
const baseForecasts = comps.map((t) => {
  const pred = new Map();
  for (const h of HOURS) {
    pred.set(h, GAUSS.map((z) => Math.exp(ln(Math.max(1, t.p1)) + z * 0.15 * Math.sqrt(h - 1))));
  }
  return { truth: new Map(HOURS.map((h) => [h, t.at(h)])), pred };
});
console.log(`baseline house persistence: ${score(baseForecasts).loss.toFixed(4)}`);
