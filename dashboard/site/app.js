"use strict";
// edge-lab dashboard. Reads the public data files the jobs write (no keys here) and replays NX3's real trees.
const RAW = "https://raw.githubusercontent.com/amit8f/edge-lab-public";
const Q = new URLSearchParams(location.search);
const CFG = {
  summary: Q.get("summary") || `${RAW}/state-dash/dash/summary.json`,
  live: d => `${RAW}/state-nx3/nx3/days/${d}/decisions.jsonl`,
  trees: "nx3_trees.json",
  krr: "nxk_krr.json",
  queue: [`${RAW}/claude/project-thread-s2jzh2/research/QUEUE.json`, `${RAW}/main/research/QUEUE.json`],
  runs: "https://api.github.com/repos/amit8f/edge-lab-public/actions/runs?per_page=30",
  logo: s => `https://financialmodelingprep.com/image-stock/${encodeURIComponent(s)}.png`,
};
const LABEL = {
  n_articles: "articles", lm_neg: "negative words", lm_pos: "positive words", lm_net: "word tone", fb_pos: "positive tone",
  fb_neg: "negative tone", fb_max_neg: "most negative", fb_max_pos: "most positive", fb_neu: "neutral tone",
  head_len: "headline length", f_earnings: "earnings", f_guidance: "guidance", f_analyst_up: "upgrade",
  f_analyst_down: "downgrade", f_offering: "share offering", f_merger: "merger", f_fda: "FDA", f_contract: "contract",
  f_legal: "legal", f_bankruptcy: "bankruptcy", f_dividend: "dividend", f_executive: "executive change",
  time_bucket: "time of day", novelty: "news in last days", r1: "1-day move", r5: "5-day move", r20: "20-day move",
  vol20: "volatility", log_dv: "dollar volume", log_px: "price level", spy_r1: "market 1-day", spy_r5: "market 5-day",
  mins_open: "minutes since open", r_open: "move since open", r_react: "move since news", rv30: "volume pace",
  spy_open: "market since open",
};
const FLAG_CHIPS = ["f_earnings", "f_guidance", "f_analyst_up", "f_analyst_down", "f_offering", "f_merger", "f_fda",
  "f_contract", "f_legal", "f_bankruptcy", "f_dividend", "f_executive"];
const SKIP_WHY = { "not fired": "news effect inside the cut", "side full": "already 3 picks on that side today",
  "not easy to borrow": "not easy to borrow for a short" };
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = id => document.getElementById(id);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (v != null) e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}
const pct = (x, d = 2) => x == null ? "—" : `${x >= 0 ? "+" : ""}${(100 * x).toFixed(d)}%`;
const num = (x, d = 2) => x == null ? "—" : Number(x).toFixed(d);
const money = x => x == null ? "—" : `$${Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
const sleep = ms => new Promise(r => setTimeout(r, reduced ? Math.min(ms, 30) : ms));
function etDate(d = new Date()) {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(d);
}
// times: the logs write New York wall-clock time; the page shows Israel time
function tzOffset(ms, tz) {
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-US", { timeZone: tz, hourCycle: "h23", year: "numeric", month: "2-digit",
    day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).formatToParts(new Date(ms)).map(x => [x.type, x.value]));
  return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second) - Math.floor(ms / 1000) * 1000;
}
function fromET(s) {
  if (!s) return null;
  const [d, t = "0:0:0"] = s.split("T"); const [Y, M, D] = d.split("-").map(Number); const [h, m, sec] = t.split(":").map(Number);
  const wall = Date.UTC(Y, M - 1, D, h, m, Math.floor(sec || 0));
  let u = wall; for (let k = 0; k < 2; k++) u = wall - tzOffset(u, "America/New_York");
  return new Date(u);
}
const IL = { timeZone: "Asia/Jerusalem", hour: "2-digit", minute: "2-digit", hourCycle: "h23" };
const ilTime = d => d ? new Intl.DateTimeFormat("en-GB", IL).format(d) : "";
const ilStamp = d => d ? new Intl.DateTimeFormat("en-GB", { ...IL, day: "2-digit", month: "short" }).format(d) : "";
function parseLines(txt) {
  return txt.split("\n").filter(Boolean).map(l => { try { return JSON.parse(l.replace(/\bNaN\b/g, "null")); } catch { return null; } })
    .filter(Boolean);
}

function logo(sym, cls = "logo") {
  const box = el("div", { class: cls });
  const img = new Image();
  img.alt = ""; img.loading = "lazy"; img.src = CFG.logo(sym);
  img.onerror = () => { box.textContent = sym.slice(0, 5); box.style.background = color(sym); box.style.color = "#fff"; };
  box.append(img);
  return box;
}
function color(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360; return `hsl(${h} 45% 40%)`; }

// ---------- trees ----------
let MODEL = null;
function rowOf(inputs) { return MODEL.inputs.map(k => (inputs && inputs[k] != null ? inputs[k] : null)); }
function ghostOf(x) { return MODEL.inputs.map((k, i) => (k in MODEL.no_news ? MODEL.no_news[k] : x[i])); }
function walk(t, x) {
  let n = 0; const path = [0];
  while (t.l[n] !== -1) {
    const v = x[t.f[n]];
    n = v == null || Number.isNaN(v) ? (t.d[n] ? t.l[n] : t.r[n]) : (Math.fround(v) < Math.fround(t.c[n]) ? t.l[n] : t.r[n]);  // 32-bit compare, as XGBoost does
    path.push(n);
  }
  return { leaf: t.c[n], path };
}

function layout(t) {
  const pos = {}; let leafIx = 0; let maxD = 0;
  (function rec(n, d) {
    maxD = Math.max(maxD, d);
    if (t.l[n] === -1) { pos[n] = { x: leafIx++, d }; return; }
    rec(t.l[n], d + 1); rec(t.r[n], d + 1);
    pos[n] = { x: (pos[t.l[n]].x + pos[t.r[n]].x) / 2, d };
  })(0, 0);
  const W = 640, H = TH - FS, nx = Math.max(leafIx, 1);
  for (const p of Object.values(pos)) { p.X = 20 + (p.x + 0.5) * (W - 40) / nx; p.Y = 24 + p.d * (H - 48) / Math.max(maxD, 1); }
  return pos;
}

const SVGNS = "http://www.w3.org/2000/svg";
const FS = innerWidth < 600 ? 19 : 11;   // tree label size in viewBox units: bigger on a phone, where the tree is scaled down
const TH = innerWidth < 600 ? 460 : 300; // taller tree on a phone so the bigger labels on adjacent levels don't touch
function s(tag, attrs) { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; }
function nodeLabel(t, n) {
  const k = MODEL.inputs[t.f[n]], v = t.c[n];
  const name = LABEL[k] || k;
  if (k.startsWith("f_") || ["n_articles", "lm_neg", "lm_pos", "novelty", "time_bucket"].includes(k)) return `${name} < ${num(v, 1)}?`;
  return `${name} < ${Math.abs(v) < 1 ? num(v, 3) : num(v, 1)}?`;
}
function drawTree(t, path, upto) {
  const svg = $("tree"); svg.replaceChildren(); svg.setAttribute("viewBox", `0 0 640 ${TH}`);
  const pos = layout(t); const on = new Set(path.slice(0, upto + 1));
  for (let n = 0; n < t.l.length; n++) {
    if (t.l[n] === -1 || !pos[n]) continue;
    for (const c of [t.l[n], t.r[n]]) {
      const lit = on.has(n) && on.has(c);
      svg.append(s("line", { x1: pos[n].X, y1: pos[n].Y, x2: pos[c].X, y2: pos[c].Y, class: lit ? "edge on" : "edge" }));
    }
  }
  for (let n = 0; n < t.l.length; n++) {
    const p = pos[n]; if (!p) continue;
    const lit = on.has(n);
    if (t.l[n] === -1) {
      const g = s("g", { class: lit ? "leaf on" : "leaf" });
      g.append(s("circle", { cx: p.X, cy: p.Y, r: lit ? 9 : 5 }));
      if (lit) {
        const tx = s("text", { x: Math.max(30, Math.min(610, p.X)), y: p.Y + 14 + FS, "text-anchor": "middle", "font-size": FS });
        tx.textContent = (t.c[n] >= 0 ? "+" : "") + t.c[n].toFixed(4);
        g.append(tx);
      }
      svg.append(g);
    } else if (lit) {
      // only the nodes on this headline's path get a question; the others stay small
      const g = s("g", { class: "node on" });
      const label = nodeLabel(t, n); const w = Math.min(300, 14 + label.length * FS * 0.56);
      const x = Math.max(4, Math.min(636 - w, p.X - w / 2));
      g.append(s("rect", { x, y: p.Y - FS, width: w, height: FS * 2, rx: 7 }));
      const tx = s("text", { x: x + w / 2, y: p.Y + FS * 0.36, "text-anchor": "middle", "font-size": FS }); tx.textContent = label;
      g.append(tx); svg.append(g);
    } else {
      const g = s("g", { class: "node dim" });
      g.append(s("rect", { x: p.X - 7, y: p.Y - 7, width: 14, height: 14, rx: 4 }));
      svg.append(g);
    }
  }
  return pos;
}
// the ball is the stock's logo (a colored circle with the ticker when no logo loads)
let ballN = 0;
function makeBall(sym, r) {
  const g = s("g", { class: "lball" }); const id = `bclip${++ballN}`;
  const cp = s("clipPath", { id }); cp.append(s("circle", { cx: 0, cy: 0, r }));
  const defs = s("defs", {}); defs.append(cp);
  const spin = s("g", {});
  const bg = s("circle", { cx: 0, cy: 0, r, fill: color(sym) });
  const tx = s("text", { x: 0, y: r * 0.28, "text-anchor": "middle", "font-size": r * 0.72, fill: "#fff", "font-weight": 700 });
  tx.textContent = sym.slice(0, 4);
  const im = s("image", { href: CFG.logo(sym), x: -r * 0.8, y: -r * 0.8, width: r * 1.6, height: r * 1.6,
    "clip-path": `url(#${id})`, preserveAspectRatio: "xMidYMid meet" });
  im.addEventListener("load", () => { bg.setAttribute("fill", "#fff"); tx.remove(); });
  im.addEventListener("error", () => im.remove());
  spin.append(bg, tx, im);
  g.append(defs, spin, s("circle", { cx: 0, cy: 0, r, class: "lball-ring" }));
  g.r = r; g.rot = 0;
  g.move = (x, y, rot = g.rot) => { g.x = x; g.y = y; g.rot = rot;
    g.setAttribute("transform", `translate(${x} ${y})`); };   // the logo stays upright (no spin)
  return g;
}
const BALL_R = innerWidth < 600 ? 27 : 16;
async function rollBall(pos, path, speed, sym, ball = makeBall(sym, BALL_R)) {
  const svg = $("tree"); ball.move(pos[0].X, pos[0].Y);
  svg.append(ball);
  for (let i = 1; i < path.length; i++) {
    const a = pos[path[i - 1]], b = pos[path[i]]; const T = reduced ? 1 : speed; const t0 = performance.now(); const r0 = ball.rot;
    const dir = b.X < a.X ? -1 : 1;
    await new Promise(res => {
      (function step(now) {
        const k = Math.min(1, (now - t0) / T), e = k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
        ball.move(a.X + (b.X - a.X) * e, a.Y + (b.Y - a.Y) * e, r0 + dir * 360 * e);
        k < 1 ? requestAnimationFrame(step) : res();
      })(t0);
    });
    drawTree(currentTree, path, i); svg.append(ball);
  }
  return ball;
}
let currentTree = null;

// ---------- meters ----------
const SCALE = { score: 0.4, eff: 0.6 };
function setMeters(real, ghost) {
  const w = v => Math.min(50, Math.abs(v) / SCALE.score * 50);
  for (const [id, v] of [["m-real", real], ["m-ghost", ghost]]) {
    const e = $(id); e.style.width = `${w(v)}%`; e.style.left = v >= 0 ? "50%" : `${50 - w(v)}%`;
  }
  const eff = real - ghost;
  $("v-real").textContent = num(real, 3); $("v-ghost").textContent = num(ghost, 3); $("v-eff").textContent = num(eff, 3);
  const x = 50 + Math.max(-50, Math.min(50, eff / SCALE.eff * 50));
  $("m-eff").style.left = `calc(${x}% - 6px)`;
}
function placeCuts(cut = MODEL.cut) {
  const c = cut / SCALE.eff * 50;
  document.querySelector(".cut.lo").style.left = `${50 - c}%`;
  document.querySelector(".cut.hi").style.left = `${50 + c}%`;
}

// ---------- KRR: score = b + sum over 300 reference events of weight x similarity ----------
let KRR = null, VIEW = "XGB";
const VIEW_NOTE = {
  XGB: "XGBoost: 400 small yes/no trees, each adds a little to the score. The live NX3 account trades on this one.",
  KRR: "KRR: a smooth curve fitted through 300 reference events from 2016-2025 (the ticks under the curve, darker = " +
    "more like this event). Shown against the news input that moves this event's score most. Ledger-only shadow book since 2026-11-09.",
  BOOM: "BOOM1 and BOOM-SHORT: two CatBoost models that run once each morning before 9:30 ET, so there is no tree to replay. " +
    "The day's 20 picks, how far each moved from the open, and the shorts.",
};
function krrScore(x) {
  const m = KRR, full = m.inputs.map((_, i) => (x[i] == null || Number.isNaN(x[i]) ? m.median[i] : x[i]));
  for (const i of m.missing_cols) full.push(x[i] == null || Number.isNaN(x[i]) ? 1 : 0);
  const z = full.map((v, i) => (v - m.mean[i]) / m.scale[i]);
  const sim = m.L.map(row => Math.exp(-m.gamma * row.reduce((a, v, i) => a + (v - z[i]) ** 2, 0)));
  const c = sim.map((k, j) => m.w[j] * k);
  return { score: m.b + c.reduce((a, v) => a + v, 0), sim, c };
}
// The view: KRR's score drawn as a smooth curve against the news input that moves it most for this event, all
// other inputs held at the event's values. The rug at the bottom is the 300 reference events (darker = more alike).
function krrAxis(x) {
  let best = null;
  for (let i = 0; i < KRR.inputs.length; i++) {
    const k = KRR.inputs[i];
    if (!(k in KRR.no_news)) continue;
    const col = KRR.L.map(row => row[i] * KRR.scale[i] + KRR.mean[i]);
    const lo = Math.min(...col), hi = Math.max(...col);
    if (!(hi > lo)) continue;
    const xs = Array.from({ length: 49 }, (_, n) => lo + (hi - lo) * n / 48);
    const ys = xs.map(v => { const y = x.slice(); y[i] = v; return krrScore(y).score; });
    const span = Math.max(...ys) - Math.min(...ys);
    if (!best || span > best.span) best = { i, k, lo, hi, xs, ys, span, col };
  }
  return best;
}
function drawCurve(ax, upto, at, ghost, sim) {
  const svg = $("tree"); svg.replaceChildren(); svg.setAttribute("viewBox", "0 0 640 300");
  const L = 56, R = 620, T = 18, B = 250;
  const yv = [...ax.ys, ghost, at ? at.y : ghost];
  const ylo = Math.min(...yv), yhi = Math.max(...yv), pad = (yhi - ylo) * 0.12 || 0.01;
  const X = v => L + (R - L) * (v - ax.lo) / (ax.hi - ax.lo), Y = v => B - (B - T) * (v - ylo + pad) / (yhi - ylo + 2 * pad);
  svg.append(s("line", { x1: L, y1: B, x2: R, y2: B, stroke: "var(--line)" }), s("line", { x1: L, y1: T, x2: L, y2: B, stroke: "var(--line)" }));
  const lab = (x, y, t, a = "middle") => { const e = s("text", { x, y, "text-anchor": a, "font-size": 12, fill: "var(--muted)" }); e.textContent = t; return e; };
  svg.append(lab((L + R) / 2, 292, LABEL[ax.k] || ax.k), lab(L - 6, T + 4, num(yhi + pad, 2), "end"), lab(L - 6, B, num(ylo - pad, 2), "end"),
    lab(L, 268, num(ax.lo, 2)), lab(R, 268, num(ax.hi, 2)));
  const mx = Math.max(1e-9, ...sim);
  ax.col.forEach((v, j) => svg.append(s("line", { x1: X(v), y1: B - 2, x2: X(v), y2: B - 12, stroke: "var(--accent)",
    "stroke-opacity": (0.08 + 0.9 * Math.sqrt(sim[j] / mx)).toFixed(3) })));
  svg.append(s("line", { x1: L, y1: Y(ghost), x2: R, y2: Y(ghost), stroke: "var(--ghost)", "stroke-dasharray": "5 4" }),
    lab(R, Y(ghost) - 5, "news blanked", "end"));
  const n = Math.max(2, Math.round(upto * ax.xs.length));
  svg.append(s("polyline", { points: ax.xs.slice(0, n).map((v, m) => `${X(v).toFixed(1)},${Y(ax.ys[m]).toFixed(1)}`).join(" "),
    fill: "none", stroke: "var(--text)", "stroke-width": 2.2, "stroke-linejoin": "round" }));
  if (at) svg.append(s("circle", { cx: X(at.x), cy: Y(at.y), r: 7, fill: at.y >= ghost ? "var(--buy)" : "var(--short)", stroke: "var(--text)", "stroke-width": 1.5 }));
}
async function playKRR(ev) {
  playing = true;
  $("verdict").hidden = true;
  showWho(ev);
  const inp = ev.inputs || {}, x = KRR.inputs.map(k => (inp[k] != null ? inp[k] : null));
  const g = KRR.inputs.map((k, i) => (k in KRR.no_news ? KRR.no_news[k] : x[i]));
  const r = krrScore(x), q = krrScore(g);
  placeCuts(KRR.cut); setMeters(q.score, q.score);
  const ax = krrAxis(x.map((v, i) => (v == null ? NaN : v)));
  const xv = x[ax.i] == null ? KRR.median[ax.i] : Math.min(ax.hi, Math.max(ax.lo, x[ax.i]));
  $("treeno").textContent = `KRR score vs ${LABEL[ax.k] || ax.k}`;
  $("phase").textContent = "drawing the curve, other inputs at this event's values";
  for (let f = 0; f <= 1.0001; f += 0.04) {
    if (VIEW !== "KRR") { placeCuts(MODEL.cut); playing = false; return; }
    drawCurve(ax, f, null, q.score, r.sim); await sleep(45);
  }
  $("phase").textContent = "where this event sits on it";
  const steps = 30, x0 = ax.lo;
  for (let n = 0; n <= steps; n++) {
    if (VIEW !== "KRR") { placeCuts(MODEL.cut); playing = false; return; }
    const v = x0 + (xv - x0) * n / steps, y = x.slice(); y[ax.i] = v;
    const sc = krrScore(y.map(t => (t == null ? NaN : t))).score;
    drawCurve(ax, 1, { x: v, y: sc }, q.score, r.sim); setMeters(sc, q.score); await sleep(40);
  }
  drawCurve(ax, 1, { x: xv, y: r.score }, q.score, r.sim); setMeters(r.score, q.score);
  $("phase").textContent = `the gap to the dashed line is the news effect (${num(r.score - q.score, 3)})`;
  await sleep(1600);
  const eff = r.score - q.score, side = Math.abs(eff) >= KRR.cut && Math.abs(eff) > 0 ? Math.sign(eff) : 0;
  await finale({ ...ev, side, skip: side ? null : "not fired", e: eff, result: side === ev.side ? ev.result : null }, eff, KRR.cut, "KRR");
  placeCuts(MODEL.cut);
  playing = false;
}
async function setView(v) {
  if (v === "KRR" && !KRR) {
    try { KRR = await getJSON(CFG.krr); } catch { $("viewnote").textContent = "KRR model file missing"; return; }
  }
  VIEW = v;
  try { localStorage.setItem("view", v); } catch { /* storage blocked: the choice lasts this visit */ }
  for (const b of document.querySelectorAll("#viewswitch button")) b.setAttribute("aria-pressed", String(b.dataset.view === v));
  $("viewnote").textContent = VIEW_NOTE[v];
  $("nx3view").hidden = v === "BOOM";
  $("boomview").hidden = v !== "BOOM";
  if (v === "BOOM") renderBoom();
}
for (const b of document.querySelectorAll("#viewswitch button")) b.addEventListener("click", () => setView(b.dataset.view));

// ---------- the stage ----------
let playing = false; const queue = []; const shown = new Set();
function decisionOf(ev) {
  if (ev.side === 1) return { cls: "buy", word: "BUY" };
  if (ev.side === -1) return { cls: "short", word: "SHORT" };
  return { cls: "skip", word: "SKIP" };
}
function showWho(ev) {
  $("logo").replaceWith(Object.assign(logo(ev.symbol), { id: "logo" }));
  $("sym").textContent = ev.symbol;
  $("when").textContent = `news ${ilTime(fromET(ev.t))} Israel · price at decision ${money(ev.entry_iex)}` +
    (ev.last_price ? ` · now ${money(ev.last_price)}` : "");
  const chips = $("chips"); chips.replaceChildren();
  const inp = ev.inputs || {};
  for (const k of FLAG_CHIPS) if (inp[k]) chips.append(el("span", { class: "chip", text: LABEL[k] }));
  if (inp.fb_neg != null && inp.fb_pos != null) {
    const neg = inp.fb_neg > inp.fb_pos;
    chips.append(el("span", { class: `chip ${neg ? "neg" : "pos"}`, text: `tone ${neg ? "negative" : "positive"} ${num(Math.max(inp.fb_neg, inp.fb_pos), 2)}` }));
  }
  if (!chips.children.length) chips.append(el("span", { class: "chip", text: "general news" }));
}
const SLOW = 3, LAST = 1;   // first trees step by step, the middle ones flash past, the last one step by step again
async function play(ev) {
  if (VIEW === "KRR" && KRR) return playKRR(ev);
  playing = true; let lastBall = null;
  $("verdict").hidden = true;
  showWho(ev);
  const x = rowOf(ev.inputs), g = ghostOf(x);
  let real = MODEL.base_score, ghost = MODEL.base_score; setMeters(real, ghost);
  const N = MODEL.trees.length;
  for (let i = 0; i < N; i++) {
    if (VIEW !== "XGB") { playing = false; return; }     // switched model: the next event plays in the new view
    const t = MODEL.trees[i]; currentTree = t;
    const w = walk(t, x), wg = walk(t, g);
    const slow = i < SLOW || i >= N - LAST;
    if (slow) {
      $("treeno").textContent = `tree ${i + 1} / ${N}`;
      $("phase").textContent = i < SLOW ? "step by step" : "the last tree";
      const pos = drawTree(t, w.path, 0);
      await sleep(i < SLOW ? 250 : 120);
      lastBall = await rollBall(pos, w.path, 420, ev.symbol, lastBall || undefined);
      real += w.leaf; ghost += wg.leaf; setMeters(real, ghost);
      await sleep(i < SLOW ? 500 : 250);
    } else {
      // every middle tree flashes past with the logo on the leaf it lands in
      real += w.leaf; ghost += wg.leaf;
      $("treeno").textContent = `tree ${i + 1} / ${N}`;
      $("phase").textContent = "the other trees, fast";
      const pos = drawTree(t, w.path, w.path.length), leaf = pos[w.path[w.path.length - 1]];
      if (!lastBall) lastBall = makeBall(ev.symbol, BALL_R);
      lastBall.move(leaf.X, leaf.Y, lastBall.rot + 24); $("tree").append(lastBall);
      setMeters(real, ghost); await sleep(16);
    }
  }
  setMeters(real, ghost);
  await finale(ev, real - ghost);
  playing = false;
}

// the finish: logo and number, three cards, a highlight that teases each one, then the pick and its result
function outcomeLine(ev, d) {
  if (ev.result && ev.result.net != null) return `result ${pct(ev.result.net)} after costs`;
  if (d.cls === "skip") return ev.next_move != null ? `not traded · next day ${pct(ev.next_move)}` : "not traded";
  const sgn = d.cls === "buy" ? 1 : -1;
  if (ev.last_price && ev.entry_iex) return `in at ${money(ev.entry_iex)} · now ${money(ev.last_price)} (${pct(sgn * (ev.last_price / ev.entry_iex - 1))} so far)`;
  return `in at ${money(ev.entry_iex)} · result after the exit`;
}
// the finish: the screen turns the decision's color with the logo, the score and the result
async function finale(ev, eff, cut = MODEL.cut, who = "") {
  const d = decisionOf(ev), e = ev.e != null ? ev.e : eff;
  await sleep(400);
  const v = $("verdict"); v.className = `verdict full ${d.cls}`;
  const why = d.cls === "skip" ? (SKIP_WHY[ev.skip] || ev.skip || "no trade") : `${who ? who + ": " : ""}news effect past the cut (${num(cut, 3)})`;
  v.replaceChildren(el("div", { class: "screen" },
    logo(ev.symbol, "logo lg"),
    el("div", { class: "ssym", text: ev.symbol }),
    el("div", { class: "big", text: d.word }),
    el("div", { class: "sscore", text: `score ${e >= 0 ? "+" : ""}${num(e, 3)}` }),
    el("div", { class: "why", text: why }),
    el("div", { class: "sres", text: outcomeLine(ev, d) })));
  v.hidden = false;
  await sleep(d.cls === "skip" ? 2600 : 4500);
}
async function loop() {
  for (;;) {
    if (VIEW === "BOOM") { await sleep(1000); continue; }     // the NX3 stage is hidden; live events wait in the queue
    let ev = queue.shift();
    if (!ev) {
      setMode(false);
      if (!replayList.length) { await sleep(2000); continue; }
      ev = replayList[replayIx++ % replayList.length];
    } else setMode(true);
    await play(ev);
  }
}
let replayList = [], replayIx = 0, replayDay = null;
function setMode(live) {
  const m = $("mode");
  m.className = `pill ${live ? "live" : "replay"}`;
  m.textContent = live ? "LIVE" : `REPLAY ${replayDay || ""}`;
}

// ---------- BOOM-SHORT on the Live tab ----------
let boomDay = null;
function renderBoom() {
  const days = (SUMMARY && SUMMARY.boomshort) || [];
  if (!days.length) {
    $("boomchips").replaceChildren();
    $("boomday").replaceChildren(el("div", { class: "muted small", text: "No mornings logged yet." }));
    return;
  }
  if (!days.some(d => d.day === boomDay)) boomDay = days[0].day;
  $("boomchips").replaceChildren(...days.map(d => {
    const b = el("button", { "aria-pressed": String(d.day === boomDay), text: d.day });
    b.addEventListener("click", () => { boomDay = d.day; renderBoom(); });
    return b;
  }));
  const d = days.find(x => x.day === boomDay);
  const pct = x => x == null ? "" : `${x >= 0 ? "+" : ""}${(100 * x).toFixed(2)}%`;
  const nShort = d.picks.filter(p => p.role === "short").length;
  const head = d.status !== "decided"
    ? el("div", { class: "card small warn", text: `${d.status}${d.reason ? `: ${d.reason}` : ""}` })
    : el("div", { class: "card small" },
        d.boom1_priced ? el("div", { class: "kv" }, el("b", { text: `BOOM1: ${d.boom1_hits} of ${d.boom1_priced} picks moved 8%+ from the open` }),
          el("span", { class: d.boom1_hits / d.boom1_priced >= 0.5 ? "up" : "muted", text: `${Math.round(100 * d.boom1_hits / d.boom1_priced)}%` })) : null,
        el("div", { class: "kv" }, el("span", { text: `${d.universe || "?"} stocks scored · ${nShort} shorted` }),
          el("span", { class: "muted", text: d.decided_at ? `decided ${d.decided_at.slice(11, 16)} ET` : "" })),
        d.settled ? el("div", { class: "kv" }, el("span", { text: `shorts ${pct(d.book_mean)} each · control ${pct(d.control_mean)}` }),
          el("span", { class: (d.pnl_usd || 0) >= 0 ? "up" : "down",
            text: d.pnl_usd == null ? "" : `${d.pnl_usd >= 0 ? "+" : "-"}$${Math.abs(d.pnl_usd).toFixed(0)}` })) : null,
        d.blocked && d.blocked.length ? el("div", { class: "muted", text: `not easy to borrow, skipped: ${d.blocked.join(", ")}` }) : null);
  const rows = d.picks.map(p => {
    const r = p.result;
    const mark = !r && p.role && p.last_price && p.prev_c ? -(p.last_price / p.prev_c - 1) : null;
    const res = r && r.status === "settled" ? pct(r.net) : mark != null ? `${pct(mark)} now` : "";
    const cls = r && r.net != null ? (r.net >= 0 ? "up" : "down") : mark != null ? (mark >= 0 ? "up" : "down") : "muted";
    const role = p.role === "short" ? "SHORTED" : p.role === "control" ? "control short" : "watched";
    return el("div", { class: "card small" },
      el("div", { class: "kv" }, el("span", {}, el("b", { text: p.symbol }), el("span", { class: "muted", text: ` · ${role}` })),
        el("span", { class: cls, text: res })),
      p.move ? el("div", { class: "kv" }, el("span", { class: p.move.hit ? "up" : "muted",
          text: p.move.hit ? "BOOM: moved 8%+" : "no 8% move yet" }),
        el("span", { class: "muted", text: `from open: high ${pct(p.move.up)} · low ${pct(p.move.down)}` })) : null,
      el("div", { class: "kv muted" },
        el("span", { text: `boom ${p.boom1 != null ? p.boom1.toFixed(2) : "?"} · direction ${p.d6 != null ? p.d6.toFixed(3) : "?"}` }),
        el("span", { text: p.pm_gap != null ? `pre-market ${pct(p.pm_gap)}` : "" })));
  });
  $("boomday").replaceChildren(head, ...rows, el("div", { class: "muted small",
    text: "\"now\" is the short's gain since yesterday's close at the latest price, before costs; the book settles from the first open." }));
}

// ---------- the Boom tab ----------
function renderBoomTab() {
  const B = SUMMARY && SUMMARY.boom;
  if (!B) { $("boom1sum").replaceChildren(el("div", { class: "muted small", text: "No BOOM data yet." })); return; }
  const pct = (x, d = 0) => x == null ? "–" : `${(100 * x).toFixed(d)}%`;
  const sgn = (x, d = 2) => x == null ? "–" : `${x >= 0 ? "+" : ""}${x.toFixed(d)}%`;
  const usd = x => `${x >= 0 ? "+" : "-"}$${Math.abs(x).toFixed(0)}`;
  const b1 = B.boom1, sh = B.short;
  const stat = (label, value, cls, sub) => el("div", { class: "card book" },
    el("div", { class: "muted small", text: label }), el("div", { class: `name ${cls || ""}`, text: value }),
    sub ? el("div", { class: "muted small", text: sub }) : null);
  $("boom1sum").replaceChildren(
    stat("Picks that moved 8%+", pct(b1.hit_rate), b1.hit_rate == null ? "muted" : b1.hit_rate >= 0.4 ? "up" : "down",
      `${b1.hits} of ${b1.picks} picks over ${b1.days.length} days`),
    stat("History (2020-2025)", pct(b1.history), "", "what BOOM1 scored before going live"),
    stat("Random stocks", pct(b1.random), "muted", "the same test on random picks"));
  $("boom1days").replaceChildren(...b1.days.slice().reverse().map(d => el("div", { class: "job" },
    el("span", { text: d.day }), el("span", { class: d.hits / d.picks >= 0.4 ? "up" : "muted",
      text: `${d.hits} of ${d.picks} moved 8%+ · ${pct(d.hits / d.picks)}` }))));
  $("boomssum").replaceChildren(
    stat("Overall P&L", sgn(sh.pnl_pct), sh.pnl_pct >= 0 ? "up" : "down", `${usd(sh.pnl_usd)} on $5,000`),
    stat("Trades", `${sh.trades} of ${sh.target}`, "", sh.trades ? `${sh.wins} winners (${pct(sh.wins / sh.trades)})` : "verdict at 100 trades"),
    stat("Average short", sh.mean == null ? "–" : sgn(100 * sh.mean), sh.mean == null ? "muted" : sh.mean >= 0 ? "up" : "down",
      `random control ${sh.control_mean == null ? "–" : sgn(100 * sh.control_mean)}`),
    stat("Days not traded", String(sh.missed), sh.missed ? "warn" : "muted", "crashed, late or too few stocks"));
  const why = w => w && w.length ? w.map(([k, v]) => `${k} ${v >= 0 ? "▲" : "▼"}`).join(", ") : "";
  const lean = (p, cut) => p.d6 == null ? "–" : p.d6 <= cut ? `down ${p.d6.toFixed(3)} (short)`
    : `${p.d6 < 0 ? "down" : "up"} ${p.d6.toFixed(3)}`;
  const moved = m => !m ? "–" : `${sgn(100 * m.up, 1)} / ${sgn(100 * m.down, 1)}${m.hit ? " ✓8%" : ""}`;
  const pickTable = d => el("details", {}, el("summary", { text: `All 20 picks, why and which way (${d.picks.length})` }),
    el("div", { class: "muted small", text: `Chance = BOOM1's chance of an 8% move. Lean = the direction model's score: below 0 leans down, `
      + `${d.cut == null ? "-0.119" : d.cut.toFixed(3)} or lower gets shorted. "Why" = the 3 inputs that pushed each score most `
      + `(▲ pushed it up, ▼ down; for the lean, ▲ means toward falling). High / low = how far it went from the open.` }),
    el("div", { class: "jobs" }, ...d.picks.map(p => el("div", { class: `job ${p.shorted ? "bad" : ""}` },
      el("span", { text: `${p.symbol} · chance ${pct(p.boom1)} · lean ${lean(p, d.cut == null ? -0.119 : d.cut)}` }),
      el("span", { class: "muted small", text: [`pre-market ${sgn(100 * (p.pm_gap || 0), 1)}`, p.pre_news ? "news" : "no news",
        p.move ? `high / low ${moved(p.move)}` : null,
        p.why_move ? `why picked: ${why(p.why_move)}` : null, p.why_down ? `why that lean: ${why(p.why_down)}` : null]
        .filter(Boolean).join(" · ") })))));
  $("boomsdays").replaceChildren(...sh.days.map(d => el("div", { class: `job ${d.status !== "decided" ? "bad" : ""}` },
    el("div", {}, el("span", { text: d.day }), " ",
    el("span", { class: d.status !== "decided" ? "down" : d.pnl_usd == null ? "muted" : d.pnl_usd >= 0 ? "up" : "down",
      text: d.pnl_usd != null && d.trades ? `${d.text} ${d.trades} closed · ${usd(d.pnl_usd)} · total ${sgn(d.cum_pct)}`
        : d.shorts && d.shorts.length ? `${d.text} Settles after the close.` : d.text })),
    d.picks && d.picks.length ? pickTable(d) : null)));
  const last = sh.days[0];
  $("boomtoday").replaceChildren(last ? el("div", { class: "card small" },
    el("div", { class: "kv" }, el("b", { text: `Latest morning (${last.day})` }),
      el("span", { class: sh.trades ? (sh.pnl_pct >= 0 ? "up" : "down") : "muted",
        text: sh.trades ? `book ${sgn(sh.pnl_pct)} overall` : "no trades yet, book at $5,000" })),
    el("div", { text: last.text })) : el("div", { class: "muted small", text: "No mornings logged yet." }));
}

// ---------- data ----------
let SUMMARY = null, liveRows = [];
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}
async function loadSummary() {
  try {
    SUMMARY = await getJSON(CFG.summary);
  } catch (e) {
    $("updated").textContent = "data not available yet"; return;
  }
  const age = Math.round((Date.now() - Date.parse(SUMMARY.generated_utc)) / 60000);
  $("updated").textContent = `data ${age < 1 ? "just now" : `${age} min ago`}`;
  // replays show only the trades taken (Buy or Short), from the last earlier day that had any
  const traded = d => d.events.filter(e => e.side);
  const withEvents = SUMMARY.nx3.find(d => traded(d).length && d.day !== etDate()) || SUMMARY.nx3.find(d => traded(d).length);
  if (withEvents && replayDay !== withEvents.day) {
    replayDay = withEvents.day;
    replayList = traded(withEvents).sort((a, b) => (a.t < b.t ? -1 : 1)); replayIx = 0;
  }
  renderDays(); renderBooks(); renderIdeas(); renderProgress(); renderBoom(); renderBoomTab();
}
async function pollLive() {
  const day = etDate();
  try {
    const r = await fetch(CFG.live(day), { cache: "no-store" });
    if (!r.ok) return;
    const rows = parseLines(await r.text()).filter(x => x.status === "decided");
    const first = !liveRows.length && !shown.size;
    liveRows = rows;
    const sorted = rows.sort((a, b) => (a.decided_at < b.decided_at ? -1 : 1));
    sorted.forEach((x, i) => {
      if (shown.has(x.key)) return;
      shown.add(x.key);
      // on first load only the 3 newest play as live; the rest are already in the feed
      if (!first || i >= sorted.length - 3) queue.push(x);
    });
    if (selDay === day) renderFeed(day);
  } catch { /* offline or no file yet today */ }
}

function dayEvents(day) {
  const d = SUMMARY && SUMMARY.nx3.find(x => x.day === day);
  if (day === etDate() && liveRows.length) {
    const known = new Map((d ? d.events : []).map(e => [e.symbol + e.t, e]));
    return liveRows.map(r => Object.assign({}, r, known.get(r.symbol + r.t) || {}));
  }
  return d ? d.events : [];
}
let selDay = null;        // the day shown under the tree; today unless another chip is tapped
function renderDays() {
  const days = SUMMARY.nx3.map(d => d.day);
  if (!days.includes(etDate())) days.unshift(etDate());
  days.sort().reverse();
  if (!days.includes(selDay)) selDay = days[0];
  const fmt = d => d === etDate() ? "Today" :
    new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" }).format(new Date(d + "T12:00Z"));
  $("daychips").replaceChildren(...days.map(d => {
    const b = el("button", { "aria-pressed": String(d === selDay), text: fmt(d) });
    b.addEventListener("click", () => { selDay = d; feedFilter = "all"; renderDays(); });
    return b;
  }));
  renderFeed(selDay);
}
let feedFilter = "all";   // all, buy, short or skip
let selModel = "NX3";     // whose decisions the day list shows: NX3 or one of the NXM / NXR books
const MODELS = [["NX3", "NX3"], ["LGBM", "NXM LightGBM"], ["LIN", "NXM linear"], ["MLP", "NXM neural net"],
  ["ENS", "NXM blend"], ["XGBR", "NXR XGBoost"], ["RFR", "NXR forest"], ["KRR", "NXK kernel model"], ["CAT", "NXK2 CatBoost"], ["ALL", "NX3-ALL (no cap)"],
  ["HI", "NX3-HI"], ["ALLHI", "NX3-ALL-HI (no cap)"], ["AGENT", "NX3-AGENT (Gemini)"]];
// an event as the chosen model saw it: NX3's own decision, or that book's pick (side and result) or a skip
function asModel(ev) {
  if (selModel === "NX3") return ev;
  const a = (ev.arms || {})[selModel];
  return Object.assign({}, ev, { side: a ? a.side : null, w: a ? a.w : null, skip: a ? null : "not picked", nx3_side: ev.side, orig: ev,
    result: a && a.net != null ? { net: a.net } : null });
}
function renderFeed(day) {
  const d = SUMMARY && SUMMARY.nx3.find(x => x.day === day);
  const evs = [...dayEvents(day)].sort((a, b) => (b.t > a.t ? 1 : -1)).map(asModel);
  // the NXM / NXR books pick after the close, so on today's list they only appear once they have
  const booksIn = day !== etDate() || evs.some(ev => ev.arms);
  if (!booksIn) selModel = "NX3";
  $("modelfilter").replaceChildren(...MODELS.filter(([k]) => booksIn || k === "NX3").map(([k, label]) => {
    const n = k === "NX3" ? evs.filter(ev => ev.side).length : evs.filter(ev => (ev.arms || {})[k]).length;
    const b = el("button", { "aria-pressed": String(selModel === k), text: `${label} ${n}`,
      title: k === "NX3" ? "NX3, the model the tree shows" : "this book's picks; the tree still shows NX3" });
    b.addEventListener("click", () => { selModel = k; feedFilter = "all"; renderFeed(day); });
    return b;
  }));
  const c = d ? d.counts : null;
  $("daysum").textContent = !d ? "No decisions logged yet for this day." :
    `${c.seen} headlines seen · ${c.decided} decided · ${c.fired} fired · ${c.picks} picks` +
    (d.excluded ? " · left out of the forward test (bug days)" : "") +
    (d.pick_minus_control != null ? ` · picks minus random control ${pct(d.pick_minus_control)}` : "") +
    (d.control && d.control.length ? ` · control: ${d.control.map(x => `${x.symbol} ${x.result ? pct(x.result.net) : "open"}`).join(", ")}` : "");
  // the selected book's P&L for the day as a share of its $5,000: each trade's size times its result (net once
  // settled, otherwise the move since the decision, before costs), summed. NX3 and the NXM / NXR arms put $500
  // (10%) on a pick; NX3-ALL sizes each pick itself.
  const open = evs.filter(ev => ev.side);
  if (open.length) {
    let tot = 0, cnt = 0;
    for (const ev of open) {
      const r = ev.result ? ev.result.net : (ev.last_price && ev.entry_iex ? ev.side * (ev.last_price / ev.entry_iex - 1) : null);
      if (r == null) continue;
      tot += ((selModel === "ALL" || selModel === "ALLHI" || selModel === "AGENT") && ev.w != null ? ev.w : 0.1) * r; cnt++;
    }
    if (cnt) $("daysum").textContent += ` · book P&L ${pct(tot)} of $5,000 so far (${cnt} of ${open.length} trades priced; open trades before costs)`;
  }
  if (day === etDate()) $("daycounts").textContent = c ? `today: ${c.decided} decided, ${c.picks} picks` : "";
  const shownEvs = feedFilter === "all" ? evs : evs.filter(ev => decisionOf(ev).cls === feedFilter);
  $("feedfilter").replaceChildren(...[["all", "All"], ["buy", "Buy"], ["short", "Short"], ["skip", "Skip"]].map(([k, label]) => {
    const n = k === "all" ? evs.length : evs.filter(ev => decisionOf(ev).cls === k).length;
    const b = el("button", { "aria-pressed": String(feedFilter === k), text: `${label} ${n}` });
    b.addEventListener("click", () => { feedFilter = k; renderFeed(day); });
    return b;
  }));
  $("feed").replaceChildren(...shownEvs.map(ev => evCard(ev, day === etDate())));
  if (!shownEvs.length) $("feed").append(el("p", { class: "muted small", text: "No headlines with this decision on this day." }));
}
function pickedBy(ev) {
  const names = Object.keys(ev.arms || {}).filter(k => k !== selModel);
  if (selModel !== "NX3" && ev.nx3_side) names.unshift("NX3");
  return names.length ? el("div", { class: "l2", text: `also picked by ${names.join(", ")}` }) : "";
}
function evCard(ev, today) {
  const dcs = decisionOf(ev);
  const res = el("div", { class: "res" });
  // every stock: its price now and how far the stock itself moved since the decision (grey: up or down
  // is good or bad depending on the side); an open trade's line is in its own direction (a short gains
  // when the stock falls), green when it is making money
  const mv = ev.last_price && ev.entry_iex ? ev.last_price / ev.entry_iex - 1 : null;
  if (ev.last_price) res.append(el("div", { class: "small" }, el("span", { text: `now ${money(ev.last_price)} ` }),
    el("span", { class: "muted", text: `stock ${pct(mv)}` })));
  const run = ev.side && mv != null ? ev.side * mv : null;
  if (ev.result) res.append(el("div", { class: `small ${ev.result.net >= 0 ? "up" : "down"}`, text: `trade ${pct(ev.result.net)} net` }));
  else if (run != null) res.append(el("div", { class: `small ${run >= 0 ? "up" : "down"}`,
    text: `trade ${pct(run)} so far${today ? "" : ", settling"}` }));
  else if (ev.side) res.append(el("div", { class: "muted small", text: today ? "trade open" : "settling" }));
  else if (ev.next_move != null) res.append(el("div", { class: "muted small", text: `next close ${pct(ev.next_move)}` }));
  const card = el("div", { class: `ev ${ev.side ? "" : "skipped"}`, role: "button", tabindex: "0" },
    logo(ev.symbol),
    el("div", {},
      el("div", { class: "l1" }, el("span", { text: ev.symbol }), el("span", { class: `dec ${dcs.cls}`, text: dcs.word }),
        el("span", { class: "muted small", text: ilTime(fromET(ev.t)) })),
      el("div", { class: "l2", text: `effect ${num(ev.e, 3)} · price at decision ${money(ev.entry_iex)}` +
        (ev.skip && ev.skip !== "not fired" && ev.skip !== "not picked" ? ` · ${SKIP_WHY[ev.skip] || ev.skip}` : "") }),
      pickedBy(ev)),
    res);
  const go = () => { queue.unshift(ev.orig || ev); showTab("live"); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); };
  card.addEventListener("click", go); card.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  return card;
}

function simPanel(sim) {
  const box = el("div", { class: "sim" }), cb = el("input", { type: "checkbox" });
  const body = el("div", { class: "simbody" });
  body.hidden = true;
  const xs = sim.curve.map(c => c[1]), lo = Math.min(...xs), hi = Math.max(...xs), n = xs.length;
  const pts = xs.map((v, i) => `${(300 * i / Math.max(1, n - 1)).toFixed(1)},${(70 - 66 * (v - lo) / Math.max(1e-9, hi - lo)).toFixed(1)}`).join(" ");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 300 74"); svg.setAttribute("class", "simchart");
  const pl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  pl.setAttribute("points", pts); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "currentColor"); pl.setAttribute("stroke-width", "1.5");
  svg.append(pl);
  const yrs = Object.entries(sim.per_year_pct).map(([y, v]) => `${y} ${v >= 0 ? "+" : ""}${v.toFixed(0)}%`).join(" · ");
  body.append(svg, el("div", { class: "kv small" }, el("span", { text: `${money(sim.start)} → ${money(sim.end)}` }),
    el("span", { text: `${sim.trades} trades, ${(100 * sim.mean_net_per_trade).toFixed(2)}% each` })),
    el("div", { class: "muted small", text: yrs }), el("div", { class: "muted small", text: sim.label + ". A picture of the past, not a forecast." }));
  cb.addEventListener("change", () => { body.hidden = !cb.checked; });
  box.append(el("label", { class: "small" }, cb, " show history simulation"), body);
  return box;
}

function renderBooks() {
  const S = SUMMARY;
  const twins = Object.fromEntries(S.books.filter(b => b.control).map(b => [b.name.replace(/-coin$/, ""), b]));
  $("books").replaceChildren(...S.books.filter(b => !b.control).map(b => {
    const tw = twins[b.name];
    const p = b.target ? Math.min(100, 100 * b.closed / b.target) : null;
    return el("div", { class: "card book" },
      el("div", { class: "top2" }, el("span", { class: "name", text: b.name }), el("span", { class: "muted small", text: b.idea })),
      p != null ? el("div", { class: "progress" }, Object.assign(el("div"), { style: `width:${p}%` })) : null,
      el("div", { class: "kv" }, el("span", { text: b.target ? `${b.closed} of ${b.target} ${b.unit}` : `${b.closed} ${b.unit}` }),
        el("span", { class: b.pnl_pct == null ? "muted" : b.pnl_pct >= 0 ? "up" : "down",
          text: b.pnl_pct == null ? "" : `${b.pnl_pct >= 0 ? "+" : ""}${b.pnl_pct.toFixed(2)}%` })),
      b.open != null ? el("div", { class: "kv muted small" }, el("span", { text: b.unrealized != null ? `${b.open} open, counted in the %` : `${b.open} open` }),
        el("span", { text: b.account || "" })) : null,
      b.edge_pct != null ? el("div", { class: "kv small" }, el("span", { class: "muted", text: "picks minus random so far" }),
        el("span", { class: b.edge_pct >= 0 ? "up" : "down", text: `${b.edge_pct >= 0 ? "+" : ""}${b.edge_pct.toFixed(2)}%` +
          (b.t_nw != null ? ` · t ${b.t_nw.toFixed(2)}` : "") })) : null,
      b.first_day && b.closed === 0 ? el("div", { class: "muted small", text: `counts from ${b.first_day}` }) : null,
      b.excluded_days && b.excluded_days.length ? el("div", { class: "muted small", text: `left out: ${b.excluded_days.join(", ")}` }) : null,
      tw ? el("div", { class: "kv muted small" }, el("span", { text: `coin-flip twin: ${tw.closed} trades` }),
        el("span", { text: tw.pnl_pct == null ? "" : `${tw.pnl_pct >= 0 ? "+" : ""}${tw.pnl_pct.toFixed(2)}%` })) : null,
      b.note ? el("div", { class: "muted small", text: b.note }) : null,
      b.sim ? simPanel(b.sim) : null);
  }));
  const ad = S.adaptive;
  $("accounts").replaceChildren(...S.accounts.map(a => el("div", { class: "card book" },
    el("div", { class: "top2" }, el("span", { class: "name", text: a.name }),
      el("span", { class: a.pnl_pct == null ? "muted" : a.pnl_pct >= 0 ? "up" : "down",
        text: a.pnl_pct == null ? (a.status || "") : `${a.pnl_pct >= 0 ? "+" : ""}${a.pnl_pct.toFixed(2)}%` })),
    a.equity != null ? el("div", { class: "kv small" }, el("span", { text: `equity ${money(a.equity)}` }),
      el("span", { class: "muted", text: `${(a.positions || []).length} positions` })) : null,
    ...(a.positions || []).slice(0, 8).map(p => el("div", { class: "kv small" }, el("span", { text: `${p.symbol} ${p.side}` }),
      el("span", { class: p.pl_pct >= 0 ? "up" : "down", text: `${p.pl_pct >= 0 ? "+" : ""}${p.pl_pct.toFixed(2)}%` }))))),
    ...(ad ? [el("div", { class: "card book" }, el("div", { class: "name", text: "M3 / M4 / M6 nightly" }),
      el("div", { class: "small", text: `last night logged ${ad.last_night}, orders ${ad.send ? "sent" : "dry run"}` }),
      ...Object.entries(ad.skipped).filter(([, n]) => n).map(([k, n]) =>
        el("div", { class: "small warn", text: `${k}: ${n} orders skipped (budget under one share)` })))] : []));
  $("jobs").replaceChildren(...S.jobs.map(j => {
    const bad = j.conclusion && !["success", "skipped", "cancelled"].includes(j.conclusion);
    return el("div", { class: `job ${bad ? "bad" : ""}` }, el("span", { text: j.name }),
      el("span", { class: bad ? "down" : "muted", text: `${j.status === "completed" ? j.conclusion : j.status} · ${ilStamp(new Date(j.at))} Israel` }));
  }));
}

let ideaFilter = "all";
function renderIdeas() {
  const all = SUMMARY.ideas;
  const n = f => all.filter(f).length;
  const F = [["all", "All", () => true], ["live", "Live books", i => i.status === "live"], ["open", "Open", i => i.status === "open"],
    ["stalled", "Stalled", i => i.stalled], ["passed", "Passed", i => i.status === "passed"], ["failed", "Failed", i => i.status === "failed"]];
  $("filters").replaceChildren(...F.map(([k, lab, f]) => {
    const b = el("button", { "aria-pressed": String(ideaFilter === k), text: `${lab} ${n(f)}` });
    b.onclick = () => { ideaFilter = k; renderIdeas(); };
    return b;
  }));
  const f = F.find(x => x[0] === ideaFilter)[2];
  const order = { stalled: 0, open: 1, live: 2, passed: 3, done: 4, failed: 5 };
  const list = all.filter(f).sort((a, b) => (order[a.stalled ? "stalled" : a.status] - order[b.stalled ? "stalled" : b.status]) || (a.days_since ?? 99) - (b.days_since ?? 99));
  $("ideas").replaceChildren(...list.map(i => el("div", { class: `idea ${i.stalled ? "stalled" : ""}` },
    el("div", { class: "h" }, el("span", { class: "id", text: i.id }), el("span", { class: `st ${i.status}`, text: i.status }),
      i.stalled ? el("span", { class: "st stall", text: `stalled ${i.days_since} days` }) : null,
      el("span", { class: "muted small", text: `${i.track} · registered ${i.registered}` +
        (i.last_change ? ` · last change ${i.last_change}` : "") })),
    el("p", { text: i.idea.length > 220 ? i.idea.slice(0, 220) + "…" : i.idea }),
    i.outcome ? el("details", {}, el("summary", { text: "outcome" }), el("div", { text: i.outcome })) : null)));
}

// ---------- progress ----------
// Tasks come from research/QUEUE.json (the working branch first, else main, whichever is newer). Running jobs come
// straight from GitHub's public API; their bar is time used against the job's time limit, not work done.
let QUEUE = null, RUNS = [];
async function loadProgress() {
  if (document.hidden) return;
  const qs = await Promise.all(CFG.queue.map(u => getJSON(u).catch(() => null)));
  QUEUE = qs.filter(Boolean).sort((a, b) => (a.updated < b.updated ? 1 : -1))[0] || QUEUE;
  try {
    const r = await getJSON(CFG.runs);
    RUNS = (r.workflow_runs || []).filter(x => x.status === "in_progress" || x.status === "queued");
  } catch { /* GitHub allows 60 calls an hour without a key; keep the last list */ }
  renderProgress();
}
function progressBar(frac, cls = "") {
  const w = Math.max(0, Math.min(1, frac));
  return el("div", { class: `pbar ${cls}`, role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
    "aria-valuenow": String(Math.round(100 * w)) }, el("div", { class: "pfill", style: `width:${(100 * w).toFixed(1)}%` }));
}
function runTime(run) {
  const limit = (SUMMARY && SUMMARY.timeouts && SUMMARY.timeouts[run.path]) || 360;
  const mins = run.status === "queued" ? 0 : Math.max(0, (Date.now() - Date.parse(run.run_started_at || run.created_at)) / 60000);
  return { mins, limit, frac: mins / limit };
}
function runRow(run) {
  const t = runTime(run);
  return el("div", { class: "run" },
    el("div", { class: "kv small" }, el("a", { href: run.html_url, target: "_blank", rel: "noopener", text: run.name }),
      el("span", { class: "muted", text: run.status === "queued" ? "waiting to start" : `${Math.round(t.mins)} of ${t.limit} min` })),
    progressBar(t.frac, "time"));
}
function renderProgress() {
  if (!$("running")) return;
  $("running").replaceChildren(...(RUNS.length ? RUNS.map(runRow) : [el("div", { class: "muted small", text: "nothing running right now" })]));
  if (!QUEUE) { $("tasks").replaceChildren(el("div", { class: "muted small", text: "task list not available yet" })); return; }
  $("qupdated").textContent = `updated ${ilStamp(new Date(QUEUE.updated))} Israel`;
  $("tasks").replaceChildren(...QUEUE.tasks.map(t => {
    const done = t.steps.filter(x => x.state === "done").length, run = t.steps.filter(x => x.state === "running").length;
    const frac = (done + 0.5 * run) / t.steps.length;
    const finished = done === t.steps.length;
    return el("div", { class: `card task ${finished ? "finished" : ""}` },
      el("div", { class: "top2" }, el("span", { class: "name", text: t.id }),
        el("span", { class: finished ? "up" : "muted", text: finished ? "done" : `${Math.round(100 * frac)}%` })),
      el("div", { class: "muted small", text: t.what }),
      progressBar(frac, finished ? "full" : ""),
      el("ol", { class: "steps" }, ...t.steps.map(x => {
        const live = x.state === "running" && x.workflow ? RUNS.find(r => r.path === x.workflow) : null;
        return el("li", { class: x.state },
          el("span", { text: `${x.state === "done" ? "✓" : x.state === "running" ? "●" : "○"} ${x.name}` +
            (x.passes ? ` (pass ${x.passes})` : "") }),
          live ? runRow(live) : null);
      })));
  }));
}

// ---------- tabs and start ----------
const TABS = ["live", "progress", "books", "boom", "ideas"];
function showTab(name) {
  for (const x of document.querySelectorAll(".tabs button")) x.setAttribute("aria-selected", String(x.dataset.tab === name));
  for (const t of TABS) $(`tab-${t}`).hidden = t !== name;
}
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));

(async function start() {
  try { MODEL = await getJSON(CFG.trees); } catch { $("updated").textContent = "tree file missing"; return; }
  placeCuts(); setMeters(MODEL.base_score, MODEL.base_score);
  let v0 = "XGB";
  try { v0 = localStorage.getItem("view") || "XGB"; } catch { /* no storage */ }
  await setView(["KRR", "BOOM"].includes(v0) ? v0 : "XGB");
  await loadSummary();
  await pollLive();
  setInterval(loadSummary, 5 * 60000);
  setInterval(pollLive, 60000);
  loadProgress();
  setInterval(loadProgress, 3 * 60000);
  document.addEventListener("visibilitychange", loadProgress);
  loop();
})();
