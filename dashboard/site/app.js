"use strict";
// edge-lab dashboard. Reads the public data files the jobs write (no keys here) and replays NX3's real trees.
const RAW = "https://raw.githubusercontent.com/amit8f/edge-lab-public";
const Q = new URLSearchParams(location.search);
const CFG = {
  summary: Q.get("summary") || `${RAW}/state-dash/dash/summary.json`,
  live: d => `${RAW}/state-nx3/nx3/days/${d}/decisions.jsonl`,
  trees: "nx3_trees.json",
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
async function rollBall(pos, path, speed = 420) {
  const svg = $("tree"); const ball = s("circle", { r: 7, class: "ball", cx: pos[0].X, cy: pos[0].Y });
  svg.append(ball);
  for (let i = 1; i < path.length; i++) {
    const a = pos[path[i - 1]], b = pos[path[i]]; const T = reduced ? 1 : speed; const t0 = performance.now();
    await new Promise(res => {
      (function step(now) {
        const k = Math.min(1, (now - t0) / T), e = k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
        ball.setAttribute("cx", a.X + (b.X - a.X) * e); ball.setAttribute("cy", a.Y + (b.Y - a.Y) * e);
        k < 1 ? requestAnimationFrame(step) : res();
      })(t0);
    });
    drawTree(currentTree, path, i); svg.append(ball);
  }
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
function placeCuts() {
  const c = MODEL.cut / SCALE.eff * 50;
  document.querySelector(".cut.lo").style.left = `${50 - c}%`;
  document.querySelector(".cut.hi").style.left = `${50 + c}%`;
}

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
const SLOW = 3, LAST = 5;   // first trees step by step, the middle ones flash past, the last few step by step again
async function play(ev) {
  playing = true;
  $("verdict").hidden = true;
  showWho(ev);
  const x = rowOf(ev.inputs), g = ghostOf(x);
  let real = MODEL.base_score, ghost = MODEL.base_score; setMeters(real, ghost);
  const N = MODEL.trees.length;
  for (let i = 0; i < N; i++) {
    const t = MODEL.trees[i]; currentTree = t;
    const w = walk(t, x), wg = walk(t, g);
    const slow = i < SLOW || i >= N - LAST;
    if (slow) {
      $("treeno").textContent = `tree ${i + 1} / ${N}`;
      $("phase").textContent = i < SLOW ? "step by step" : `the last ${LAST} trees`;
      const pos = drawTree(t, w.path, 0);
      await sleep(i < SLOW ? 250 : 120);
      await rollBall(pos, w.path, i < SLOW ? 420 : 260);
      real += w.leaf; ghost += wg.leaf; setMeters(real, ghost);
      await sleep(i < SLOW ? 500 : 300);
    } else {
      real += w.leaf; ghost += wg.leaf;
      if (i % 40 === 0 || i === N - LAST - 1) {    // a few frames only: the meters move, the counter jumps
        $("treeno").textContent = `tree ${i + 1} / ${N}`;
        $("phase").textContent = `trees ${SLOW + 1} to ${N - LAST}, adding up`;
        drawTree(t, w.path, w.path.length); setMeters(real, ghost); await sleep(90);
      }
    }
  }
  setMeters(real, ghost);
  await sleep(400);
  await finale(ev, real - ghost);
  addPlayed(ev);
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
async function finale(ev, eff) {
  const d = decisionOf(ev), cut = MODEL.cut;
  const e = ev.e != null ? ev.e : eff;
  const cards = [
    { cls: "short", word: "SHORT", rule: `effect below −${num(cut, 3)}` },
    { cls: "skip", word: "SKIP", rule: "inside the cut" },
    { cls: "buy", word: "BUY", rule: `effect above +${num(cut, 3)}` },
  ].map(c => Object.assign(c, { node: el("div", { class: `fcard ${c.cls}` }, el("div", { class: "fword", text: c.word }),
    el("div", { class: "frule", text: c.rule })) }));
  const numEl = el("div", { class: "fnum", text: "0.000" });
  const out = el("div", { class: "fout" });
  const v = $("verdict"); v.className = "verdict"; v.replaceChildren(el("div", { class: "finale" },
    el("div", { class: "fwho" }, logo(ev.symbol, "logo lg"), el("div", {}, el("div", { class: "sym", text: ev.symbol }),
      el("div", { class: "muted small", text: "news effect" }), numEl)),
    el("div", { class: "fcards" }, ...cards.map(c => c.node)), out));
  v.hidden = false;
  // the number counts up to the logged effect
  const t0 = performance.now(), T = reduced ? 1 : 900;
  await new Promise(res => (function step(now) {
    const k = Math.min(1, (now - t0) / T);
    numEl.textContent = `${e * k >= 0 ? "+" : ""}${num(e * k, 3)}`;
    k < 1 ? requestAnimationFrame(step) : res();
  })(t0));
  // the highlight hops between the cards, slowing down, and stops on the decision
  const target = cards.findIndex(c => c.cls === d.cls);
  const hops = reduced ? [target] : [0, 1, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2].concat([0, 1, 2].filter(i => i !== target), [target]);
  for (let h = 0; h < hops.length; h++) {
    cards.forEach((c, i) => c.node.classList.toggle("tease", i === hops[h]));
    await sleep(90 + h * h * 4);
  }
  cards.forEach((c, i) => { c.node.classList.remove("tease"); c.node.classList.add(i === target ? "chosen" : "out"); });
  await sleep(500);
  const why = d.cls === "skip" ? (SKIP_WHY[ev.skip] || ev.skip || "no trade") : `${num(e, 3)} is past the cut`;
  out.replaceChildren(el("div", { class: "why", text: why }), el("div", { class: `fres ${d.cls}`, text: outcomeLine(ev, d) }));
  out.classList.add("show");
  await sleep(d.cls === "skip" ? 2200 : 4200);
}
async function loop() {
  for (;;) {
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
  renderDays(); renderBooks(); renderIdeas();
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
    if ($("daysel").value === day) renderFeed(day);
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
function renderDays() {
  const sel = $("daysel"); const cur = sel.value;
  const days = SUMMARY.nx3.map(d => d.day);
  if (!days.includes(etDate())) days.unshift(etDate());
  sel.replaceChildren(...days.map(d => el("option", { value: d, text: d })));
  sel.value = days.includes(cur) ? cur : days[0];
  renderFeed(sel.value);
}
function renderFeed(day) {
  const d = SUMMARY && SUMMARY.nx3.find(x => x.day === day);
  const evs = [...dayEvents(day)].sort((a, b) => (b.t > a.t ? 1 : -1));
  const c = d ? d.counts : null;
  $("daysum").textContent = !d ? "No decisions logged yet for this day." :
    `${c.seen} headlines seen · ${c.decided} decided · ${c.fired} fired · ${c.picks} picks` +
    (d.excluded ? " · left out of the forward test (bug days)" : "") +
    (d.pick_minus_control != null ? ` · picks minus random control ${pct(d.pick_minus_control)}` : "") +
    (d.control && d.control.length ? ` · control: ${d.control.map(x => `${x.symbol} ${x.result ? pct(x.result.net) : "open"}`).join(", ")}` : "");
  if (day === etDate()) $("daycounts").textContent = c ? `today: ${c.decided} decided, ${c.picks} picks` : "";
  $("feed").replaceChildren(...evs.map(ev => evCard(ev, day === etDate())));
}
function evCard(ev, today) {
  const dcs = decisionOf(ev);
  const res = el("div", { class: "res" });
  if (ev.result) res.append(el("div", { class: ev.result.net >= 0 ? "up" : "down", text: `${pct(ev.result.net)} net` }),
    el("div", { class: "muted small", text: `in ${money(ev.result.entry)} out ${money(ev.result.exit)}` }));
  else if (ev.side) res.append(el("div", { class: "muted", text: today ? "open" : "settling" }),
    ev.last_price ? el("div", { class: "small", text: `now ${money(ev.last_price)} (${pct(ev.side * (ev.last_price / ev.entry_iex - 1))})` }) : null);
  else if (ev.next_move != null) res.append(el("div", { class: ev.next_move >= 0 ? "up" : "down", text: pct(ev.next_move) }),
    el("div", { class: "muted small", text: "if bought, to next close" }));
  else if (ev.last_price) res.append(el("div", { class: "small", text: `now ${money(ev.last_price)}` }),
    el("div", { class: "muted small", text: pct(ev.last_price / ev.entry_iex - 1) }));
  const card = el("div", { class: `ev ${ev.side ? "" : "skipped"}`, role: "button", tabindex: "0" },
    logo(ev.symbol),
    el("div", {},
      el("div", { class: "l1" }, el("span", { text: ev.symbol }), el("span", { class: `dec ${dcs.cls}`, text: dcs.word }),
        el("span", { class: "muted small", text: ilTime(fromET(ev.t)) })),
      el("div", { class: "l2", text: `effect ${num(ev.e, 3)} · price at decision ${money(ev.entry_iex)}` +
        (ev.skip && ev.skip !== "not fired" ? ` · ${SKIP_WHY[ev.skip] || ev.skip}` : "") })),
    res);
  const go = () => { queue.unshift(ev); showTab("live"); window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); };
  card.addEventListener("click", go); card.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
  return card;
}
// under the tree: every headline the stage has run, latest on top
function addPlayed(ev) {
  const box = $("played"); box.prepend(evCard(ev, etDate() === (ev.t || "").slice(0, 10)));
  while (box.children.length > 30) box.lastChild.remove();
  $("played-empty").hidden = true;
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
      b.open != null ? el("div", { class: "kv muted small" }, el("span", { text: `${b.open} open` }),
        el("span", { text: b.account || "" })) : null,
      b.first_day && b.closed === 0 ? el("div", { class: "muted small", text: `counts from ${b.first_day}` }) : null,
      b.excluded_days && b.excluded_days.length ? el("div", { class: "muted small", text: `left out: ${b.excluded_days.join(", ")}` }) : null,
      tw ? el("div", { class: "kv muted small" }, el("span", { text: `coin-flip twin: ${tw.closed} trades` }),
        el("span", { text: tw.pnl_pct == null ? "" : `${tw.pnl_pct >= 0 ? "+" : ""}${tw.pnl_pct.toFixed(2)}%` })) : null,
      b.note ? el("div", { class: "muted small", text: b.note }) : null);
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

// ---------- tabs and start ----------
const TABS = ["live", "headlines", "books", "ideas"];
function showTab(name) {
  for (const x of document.querySelectorAll(".tabs button")) x.setAttribute("aria-selected", String(x.dataset.tab === name));
  for (const t of TABS) $(`tab-${t}`).hidden = t !== name;
}
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));
$("daysel").addEventListener("change", e => renderFeed(e.target.value));

(async function start() {
  try { MODEL = await getJSON(CFG.trees); } catch { $("updated").textContent = "tree file missing"; return; }
  placeCuts(); setMeters(MODEL.base_score, MODEL.base_score);
  await loadSummary();
  await pollLive();
  setInterval(loadSummary, 5 * 60000);
  setInterval(pollLive, 60000);
  loop();
})();
