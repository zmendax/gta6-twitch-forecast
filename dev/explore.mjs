// Explore the kit's practice cases: what the comparable launches look like on their release day.
// Run: node dev/explore.mjs <kit>/input/p00.in
import { readFileSync } from 'node:fs';

const path = process.argv[2];
const case_ = JSON.parse(readFileSync(path, 'utf8'));
const L = Math;

function hourOf(iso, dayStart) {
  return Math.round((Date.parse(iso) - Date.parse(dayStart)) / 3600e3);
}

// Release-day curve of one comparable: hour 1..24 -> viewers (null when unsampled)
function dayCurve(comp) {
  const out = new Array(25).fill(null);
  for (const p of comp.hourly) {
    const h = hourOf(p.from, comp.day_start);
    if (h >= 1 && h <= 24) out[h] = p.viewers;
  }
  return out;
}

// Pre-launch level: median of the 6 hourly points before the release day start
function preLevel(comp) {
  const pts = comp.hourly
    .map((p) => ({ h: hourOf(p.from, comp.day_start), v: p.viewers }))
    .filter((p) => p.h >= -6 && p.h <= 0)
    .map((p) => p.v)
    .sort((a, b) => a - b);
  if (!pts.length) return null;
  return pts[Math.floor(pts.length / 2)];
}

// The hour of the biggest log jump in the 24h window around the day: the launch, as the data shows it.
function inferredLaunch(comp) {
  const pts = comp.hourly
    .map((p) => ({ h: hourOf(p.from, comp.day_start), v: p.viewers }))
    .filter((p) => p.h >= -6 && p.h <= 24)
    .sort((a, b) => a.h - b.h);
  let best = null;
  for (let i = 1; i < pts.length; i++) {
    const prev = pts[i - 1];
    const cur = pts[i];
    if (cur.h - prev.h !== 1) continue;
    const jump = Math.log(Math.max(1, cur.v) / Math.max(1, prev.v));
    if (!best || jump > best.jump) best = { h: cur.h, jump };
  }
  return best;
}

const rows = [];
for (const c of case_.comparables) {
  const curve = dayCurve(c);
  const p1 = curve[1];
  const jump = inferredLaunch(c);
  rows.push({ name: c.name, p1, curve, pre: preLevel(c), jump });
}

console.log('comparable | p1 | preLevel | p1/pre | inferred launch hour | ratios h2..h24 (vs p1)');
for (const r of rows) {
  const ratios = [];
  for (let h = 2; h <= 24; h++) {
    ratios.push(r.curve[h] == null ? '  -  ' : (r.curve[h] / r.p1).toFixed(2));
  }
  console.log(
    `${r.name.slice(0, 26).padEnd(26)} p1=${String(r.p1).padStart(7)} pre=${String(r.pre).padStart(6)} ` +
      `x=${(r.p1 / Math.max(1, r.pre)).toFixed(1).padStart(6)} launch=${r.jump ? r.jump.h : '-'} | ${ratios.join(' ')}`
  );
}

// Cross-comparable median ratio curve, and the spread of log ratios per hour.
console.log('\nhour | n | median ratio | p05 p25 p50 p75 p95 of ratio');
for (let h = 2; h <= 24; h++) {
  const rs = rows.filter((r) => r.curve[h] != null && r.p1 > 0).map((r) => L.log(r.curve[h] / r.p1)).sort((a, b) => a - b);
  const q = (p) => {
    if (!rs.length) return NaN;
    const idx = (rs.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    return L.exp(rs[lo] + (rs[hi] - rs[lo]) * (idx - lo));
  };
  console.log(
    `${String(h).padStart(4)} | ${String(rs.length).padStart(2)} | ${q(0.5).toFixed(2).padStart(12)} | ` +
      [0.05, 0.25, 0.5, 0.75, 0.95].map((p) => q(p).toFixed(2)).join(' ')
  );
}

// The four practice cases: hour-1 print and history around it
console.log('\npractice cases:');
for (const id of ['p00', 'p01', 'p02', 'p03']) {
  try {
    const c = JSON.parse(readFileSync(new URL(`../../gta6-twitch-ccu-forecast-01-kit/input/${id}.in`, import.meta.url), 'utf8'));
    const p1 = c.known.find((k) => k.hour === 1).viewers;
    const hist = c.history_hourly.slice(-8).map((p) => `${p.from.slice(5, 13)}:${p.viewers}`);
    console.log(`${id} ${c.game.name} p1=${p1} (${c.known[0].channels} channels) last hours: ${hist.join(' ')}`);
  } catch (e) {
    console.log(id, 'unreadable');
  }
}
