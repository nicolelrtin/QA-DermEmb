"""Interactive UMAP explorer: one self-contained HTML file, no server or internet needed.

Run after plot_umap.py (it reads the UMAP coordinates that script saves, so the static and
interactive plots show the same maps). Topics are looked up fresh from topic_map.csv, so
after editing the map you only need to re-run this script:
  python "Question Embeddings/plot_umap_interactive.py"

Writes outputs/umap_interactive.html. Open it in any browser (in VS Code: right-click the
file -> "Open with Live Preview", or download it). In the page you can
  - switch embedding model
  - color points by question set, or by unified topic (shape then shows the set)
  - toggle question sets and topics on/off by clicking their legend entries
  - search question text
  - hover a point for its question, topic, original section, set and ID (click to pin)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from topics import attach_topics, topic_order

HERE = Path(__file__).resolve().parent
MODEL_TITLES = {
    "qwen3": "Qwen3-Embedding (general SOTA)",
    "panderm": "DermLIP-PanDerm (text tower)",
    "skingpt": "SkinGPT-R1 (LLM hidden states)",
}


def build_payload(coords: pd.DataFrame) -> dict:
    coords = attach_topics(coords)
    models = [m for m in MODEL_TITLES if f"{m}_x" in coords.columns]
    if not models:
        raise SystemExit("umap_coords.csv has no <model>_x/_y columns; run plot_umap.py first.")
    sets = list(dict.fromkeys(coords["set"]))
    topics = topic_order(coords["topic"])
    if len(topics) > 16:
        print(f"warning: {len(topics)} topics but only 16 distinct topic colours (8 filled + 8 hollow); "
              "colours will repeat. Group them with group_topics.py / topic_map.csv.")
    return {
        "models": [{"key": m, "title": MODEL_TITLES[m]} for m in models],
        "sets": sets,
        "topics": topics,
        "points": [
            {
                "s": sets.index(r["set"]),
                "t": topics.index(r["topic"]),
                "id": str(r["qid"]),
                "sec": str(r["section"]),
                "q": str(r["question"]),
                "xy": [[round(float(r[f"{m}_x"]), 4), round(float(r[f"{m}_y"]), 4)] for m in models],
            }
            for _, r in coords.iterrows()
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "outputs")
    args = parser.parse_args()

    coords = pd.read_csv(args.out / "umap_coords.csv")
    payload = json.dumps(build_payload(coords), ensure_ascii=False, separators=(",", ":"))
    # Keep question text from ever closing the <script> block early.
    payload = payload.replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    path = args.out / "umap_interactive.html"
    path.write_text(html, encoding="utf-8")
    print(f"saved {path} ({len(coords)} questions)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question Coverage Explorer</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #dedcd5; --grid: #ebe9e3;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77756f;
  --accent: #2a78d6; --mark-muted: #c9c7bf;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100;
  --c5: #e87ba4; --c6: #008300; --c7: #6250d6; --c8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #383835; --grid: #262624;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --accent: #3987e5; --mark-muted: #4a4945;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
    --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #383835; --grid: #262624;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --accent: #3987e5; --mark-muted: #4a4945;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
  --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 650; }
.sub { color: var(--text-secondary); margin: 0 0 18px; max-width: 75ch; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: end; margin-bottom: 14px; }
.control { display: flex; flex-direction: column; gap: 4px; }
.control > span { font-size: 12px; color: var(--text-secondary); font-weight: 600; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface-1); }
.seg button {
  border: 0; background: transparent; color: var(--text-secondary); padding: 6px 12px;
  font: inherit; cursor: pointer; border-right: 1px solid var(--border);
}
.seg button:last-child { border-right: 0; }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
input[type="search"] {
  font: inherit; padding: 6px 10px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface-1); color: var(--text-primary); width: 240px; max-width: 100%;
}
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.chart-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 16px; align-items: baseline; }
.chart-head h2 { font-size: 15px; margin: 0; font-weight: 600; }
.count { color: var(--text-secondary); font-size: 13px; font-variant-numeric: tabular-nums; }
.legends { display: flex; flex-direction: column; gap: 8px; margin: 12px 0 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; }
.legend-title { font-size: 12px; color: var(--text-secondary); font-weight: 600; width: 92px; flex: none; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px 3px 6px; border-radius: 999px;
  border: 1px solid var(--border); background: transparent; color: var(--text-primary);
  font: inherit; font-size: 13px; cursor: pointer;
}
.chip[aria-pressed="false"] { color: var(--text-muted); text-decoration: line-through; opacity: .6; }
.chip svg { flex: none; }
.chip .n { color: var(--text-muted); font-variant-numeric: tabular-nums; font-size: 12px; }
.linkish { border: 0; background: none; color: var(--accent); font: inherit; font-size: 12px; cursor: pointer; padding: 2px 4px; }
.plot-wrap { position: relative; }
svg.plot { display: block; width: 100%; height: auto; touch-action: none; }
svg.plot:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 8px; }
.pt { stroke: var(--surface-1); stroke-width: 1; transition: transform 450ms cubic-bezier(.3,.7,.3,1), opacity 200ms; }
.pt.off { display: none; }
.pt.dim { opacity: .12; }
.hover-ring { fill: none; stroke: var(--text-primary); stroke-width: 2; pointer-events: none; }
@media (prefers-reduced-motion: reduce) { .pt { transition: none; } }
.tip {
  position: absolute; pointer-events: none; z-index: 2; max-width: 360px; min-width: 220px;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: 0 6px 24px rgba(0,0,0,.14); padding: 10px 12px; font-size: 13px; display: none;
}
.tip.pinned { pointer-events: auto; border-color: var(--text-secondary); }
.tip .q { font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }
.tip dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; }
.tip dt { color: var(--text-muted); font-size: 12px; }
.tip dd { margin: 0; color: var(--text-secondary); }
.tip .key { display: inline-block; width: 12px; height: 2px; border-radius: 1px; vertical-align: middle; margin-right: 6px; }
.tip .hint { color: var(--text-muted); font-size: 11px; margin-top: 8px; }
.note { color: var(--text-muted); font-size: 12px; margin: 8px 0 0; }
details { margin-top: 16px; }
summary { cursor: pointer; font-weight: 600; font-size: 14px; padding: 4px 0; }
.table-scroll { overflow-x: auto; margin-top: 8px; }
.list-scroll { max-height: 420px; overflow: auto; margin-top: 8px; border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.total td { font-weight: 600; border-top: 1px solid var(--border); }
.cell-key { display: inline-block; box-sizing: border-box; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
</style>
</head>
<body>
<main>
  <h1>Question coverage explorer</h1>
  <p class="sub">Each point is one generated question, placed by a 2D UMAP of its embedding. Nearby points are questions the model considers similar; the axes have no units, and distances can't be compared between models.</p>

  <div class="controls" role="group" aria-label="Chart controls">
    <div class="control"><span id="lbl-model">Embedding</span><div class="seg" id="model-seg" role="group" aria-labelledby="lbl-model"></div></div>
    <div class="control"><span id="lbl-color">Color by</span><div class="seg" id="color-seg" role="group" aria-labelledby="lbl-color"></div></div>
    <label class="control"><span>Search questions</span><input type="search" id="search" placeholder="e.g. vascular, polarized"></label>
  </div>

  <section class="card" aria-labelledby="chart-title">
    <div class="chart-head"><h2 id="chart-title"></h2><span class="count" id="count" aria-live="polite"></span></div>
    <div class="legends">
      <div class="legend" id="legend-sets"><span class="legend-title">Question sets</span></div>
      <div class="legend" id="legend-topics"><span class="legend-title">Topics</span></div>
    </div>
    <div class="plot-wrap" id="plot-wrap">
      <svg class="plot" id="plot" viewBox="0 0 1000 680" tabindex="0" role="img"
           aria-label="UMAP scatter of questions; use the question list below for the same data as text"></svg>
      <div class="tip" id="tip" role="status"></div>
    </div>
    <p class="note">Hover a point for details; click to pin it (Esc to unpin). Click legend entries to hide or show a set or topic.</p>
  </section>

  <details open>
    <summary>Coverage table: visible questions by topic and set</summary>
    <div class="table-scroll"><table id="coverage"></table></div>
  </details>
  <details>
    <summary>Question list (<span id="list-n"></span> visible)</summary>
    <div class="list-scroll"><table id="qlist"></table></div>
  </details>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("data").textContent);
  const W = 1000, H = 680, PAD = 24, R = 4.6, HIT_PX = 24;
  const SHAPES = ["circle", "square", "triangle", "diamond", "triangle-down", "plus"];
  const NS = "http://www.w3.org/2000/svg";
  const color = i => `var(--c${(i % 8) + 1})`;
  // Only 8 hues stay distinguishable, so categories 9-16 reuse hues 1-8 as hollow marks.
  const hollow = i => i % 16 >= 8;
  function paint(node, i) {
    node.style.fill = hollow(i) ? "var(--surface-1)" : color(i);
    node.style.stroke = hollow(i) ? color(i) : "";
    node.style.strokeWidth = hollow(i) ? "1.6" : "";
  }

  const state = {
    model: 0, colorBy: "set", query: "",
    setOn: D.sets.map(() => true), topicOn: D.topics.map(() => true),
    pinned: null,
  };

  function shapePath(kind, r) {
    switch (kind) {
      case "square":   { const s = r * 0.9; return `M${-s},${-s}H${s}V${s}H${-s}Z`; }
      case "triangle": { const s = r * 1.2; return `M0,${-s}L${s * 0.95},${s * 0.7}H${-s * 0.95}Z`; }
      case "diamond":  { const s = r * 1.2; return `M0,${-s}L${s},0L0,${s}L${-s},0Z`; }
      case "triangle-down": { const s = r * 1.2; return `M0,${s}L${s * 0.95},${-s * 0.7}H${-s * 0.95}Z`; }
      case "plus":     { const a = r * 1.1, b = r * 0.38;
                         return `M${-b},${-a}H${b}V${-b}H${a}V${b}H${b}V${a}H${-b}V${b}H${-a}V${-b}H${-b}Z`; }
      default:         return `M${r},0A${r},${r} 0 1 1 ${-r},0A${r},${r} 0 1 1 ${r},0Z`;
    }
  }
  function swatch(kind, fill) {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("width", 14); s.setAttribute("height", 14); s.setAttribute("viewBox", "-7 -7 14 14");
    s.setAttribute("aria-hidden", "true");
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", shapePath(kind, 5));
    typeof fill === "number" ? paint(p, fill) : (p.style.fill = fill);
    s.appendChild(p); return s;
  }
  function el(tag, attrs = {}, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  // Per-model scaling with equal aspect, so UMAP clusters are not stretched.
  const scaled = D.models.map((_, m) => {
    const xs = D.points.map(p => p.xy[m][0]), ys = D.points.map(p => p.xy[m][1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const k = Math.min((W - 2 * PAD) / (x1 - x0 || 1), (H - 2 * PAD) / (y1 - y0 || 1));
    const ox = (W - (x1 - x0) * k) / 2, oy = (H - (y1 - y0) * k) / 2;
    return D.points.map(p => [ox + (p.xy[m][0] - x0) * k, H - (oy + (p.xy[m][1] - y0) * k)]);
  });

  // Marks
  const svg = document.getElementById("plot");
  const gPts = document.createElementNS(NS, "g");
  const ring = document.createElementNS(NS, "circle");
  ring.setAttribute("class", "hover-ring"); ring.setAttribute("r", 9); ring.style.display = "none";
  svg.append(gPts, ring);
  const nodes = D.points.map(p => {
    const n = document.createElementNS(NS, "path");
    n.setAttribute("class", "pt");
    n.setAttribute("d", shapePath(SHAPES[p.s % SHAPES.length], R));
    gPts.appendChild(n); return n;
  });
  // Draw order shuffled once (fixed seed) so no set always sits on top of the others.
  let seed = 7; const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  nodes.map(n => [rand(), n]).sort((a, b) => a[0] - b[0]).forEach(([, n]) => gPts.appendChild(n));

  // Controls
  function segmented(container, options, get, set) {
    const host = document.getElementById(container);
    const btns = options.map(([value, label]) => {
      const b = el("button", { type: "button" }, label);
      b.addEventListener("click", () => { set(value); render(); });
      host.appendChild(b); return [value, b];
    });
    return () => btns.forEach(([v, b]) => b.setAttribute("aria-pressed", String(v === get())));
  }
  const syncModel = segmented("model-seg", D.models.map((m, i) => [i, m.title.split(" (")[0]]),
                              () => state.model, v => { state.model = v; });
  const syncColor = segmented("color-seg", [["set", "Question set"], ["topic", "Topic"]],
                              () => state.colorBy, v => { state.colorBy = v; });
  document.getElementById("search").addEventListener("input", e => { state.query = e.target.value.trim().toLowerCase(); render(); });

  function legend(hostId, labels, onArr, makeSwatch) {
    const host = document.getElementById(hostId);
    const chips = labels.map((label, i) => {
      const b = el("button", { type: "button", class: "chip" });
      const name = el("span", {}, label), n = el("span", { class: "n" });
      b.append(document.createElement("span"), name, n);
      b.addEventListener("click", () => { onArr[i] = !onArr[i]; render(); });
      host.appendChild(b); return { b, n };
    });
    const reset = el("button", { type: "button", class: "linkish" }, "show all");
    reset.addEventListener("click", () => { onArr.fill(true); render(); });
    host.appendChild(reset);
    return counts => chips.forEach(({ b, n }, i) => {
      b.replaceChild(makeSwatch(i), b.firstChild);
      b.setAttribute("aria-pressed", String(onArr[i]));
      n.textContent = counts[i];
    });
  }
  const syncSetLegend = legend("legend-sets", D.sets, state.setOn,
    i => swatch(SHAPES[i % SHAPES.length], state.colorBy === "set" ? color(i) : "var(--text-secondary)"));
  const syncTopicLegend = legend("legend-topics", D.topics, state.topicOn,
    i => swatch("circle", state.colorBy === "topic" ? i : "var(--mark-muted)"));

  // Tooltip
  const tip = document.getElementById("tip"), wrap = document.getElementById("plot-wrap");
  function showTip(i, pinned) {
    const p = D.points[i], [x, y] = scaled[state.model][i];
    tip.replaceChildren();
    tip.appendChild(el("div", { class: "q" }, p.q));
    const dl = el("dl");
    const row = (k, v, keyColor) => {
      const dd = el("dd");
      if (keyColor) { const key = el("span", { class: "key" }); key.style.background = keyColor; dd.appendChild(key); }
      dd.appendChild(document.createTextNode(v));
      dl.append(el("dt", {}, k), dd);
    };
    row("Topic", D.topics[p.t], state.colorBy === "topic" ? color(p.t) : null);
    row("Section", p.sec);
    row("Set", D.sets[p.s], state.colorBy === "set" ? color(p.s) : null);
    row("ID", p.id);
    tip.appendChild(dl);
    tip.appendChild(el("div", { class: "hint" }, pinned ? "Pinned. Click empty space or press Esc to unpin." : "Click to pin"));
    tip.classList.toggle("pinned", !!pinned);
    tip.style.display = "block";
    // Position in wrapper pixels, flipping to stay inside the card.
    const sx = wrap.clientWidth / W, px = x * sx, py = y * sx;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.max(0, px + 14 + tw > wrap.clientWidth ? px - 14 - tw : px + 14) + "px";
    tip.style.top = Math.max(0, Math.min(py - th / 2, wrap.clientHeight - th)) + "px";
    ring.setAttribute("cx", x); ring.setAttribute("cy", y); ring.style.display = "";
  }
  function hideTip() { tip.style.display = "none"; ring.style.display = "none"; }

  let visible = [];
  function nearest(evt) {
    const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const maxD = HIT_PX * (W / svg.getBoundingClientRect().width);
    let best = -1, bd = maxD * maxD;
    const xy = scaled[state.model];
    for (const i of visible) {
      const dx = xy[i][0] - loc.x, dy = xy[i][1] - loc.y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  svg.addEventListener("pointermove", e => {
    if (state.pinned != null) return;
    const i = nearest(e);
    i < 0 ? hideTip() : showTip(i, false);
  });
  svg.addEventListener("pointerleave", () => { if (state.pinned == null) hideTip(); });
  svg.addEventListener("click", e => {
    const i = nearest(e);
    state.pinned = i < 0 || i === state.pinned ? null : i;
    state.pinned == null ? (i < 0 ? hideTip() : showTip(i, false)) : showTip(i, true);
  });
  // Keyboard: arrow keys step through visible points (in list order), Esc unpins.
  let kbIdx = -1;
  svg.addEventListener("keydown", e => {
    if (e.key === "Escape") { state.pinned = null; hideTip(); return; }
    if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(e.key) || !visible.length) return;
    e.preventDefault();
    kbIdx = (kbIdx + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1) + visible.length) % visible.length;
    state.pinned = visible[kbIdx]; showTip(state.pinned, true);
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") { state.pinned = null; hideTip(); } });

  // Tables
  function renderCoverage(counts) {
    const t = document.getElementById("coverage");
    const sets = D.sets.map((s, i) => i).filter(i => state.setOn[i]);
    const thead = el("thead"), hr = el("tr");
    hr.appendChild(el("th", {}, "Topic"));
    sets.forEach(i => hr.appendChild(el("th", { class: "num" }, D.sets[i])));
    hr.appendChild(el("th", { class: "num" }, "Total"));
    thead.appendChild(hr);
    const tbody = el("tbody"), colTot = sets.map(() => 0);
    D.topics.forEach((name, ti) => {
      if (!state.topicOn[ti]) return;
      const tr = el("tr"), td = el("td"), key = el("span", { class: "cell-key" });
      if (hollow(ti)) key.style.border = `2px solid ${color(ti)}`; else key.style.background = color(ti);
      td.append(key, document.createTextNode(name)); tr.appendChild(td);
      let rowTot = 0;
      sets.forEach((si, c) => { const v = counts[si][ti]; rowTot += v; colTot[c] += v; tr.appendChild(el("td", { class: "num" }, v)); });
      tr.appendChild(el("td", { class: "num" }, rowTot)); tbody.appendChild(tr);
    });
    const tot = el("tr", { class: "total" });
    tot.appendChild(el("td", {}, "Total"));
    colTot.forEach(v => tot.appendChild(el("td", { class: "num" }, v)));
    tot.appendChild(el("td", { class: "num" }, colTot.reduce((a, b) => a + b, 0)));
    tbody.appendChild(tot);
    t.replaceChildren(thead, tbody);
  }
  function renderList() {
    const t = document.getElementById("qlist");
    const thead = el("thead"), hr = el("tr");
    ["Set", "ID", "Topic", "Section", "Question"].forEach(h => hr.appendChild(el("th", {}, h)));
    thead.appendChild(hr);
    const tbody = el("tbody");
    for (const i of visible) {
      const p = D.points[i], tr = el("tr");
      [D.sets[p.s], p.id, D.topics[p.t], p.sec, p.q].forEach(v => tr.appendChild(el("td", {}, v)));
      tbody.appendChild(tr);
    }
    t.replaceChildren(thead, tbody);
    document.getElementById("list-n").textContent = visible.length;
  }

  // Render
  function render() {
    syncModel(); syncColor();
    document.getElementById("chart-title").textContent = D.models[state.model].title;
    const xy = scaled[state.model], q = state.query;
    const setCounts = D.sets.map(() => 0), topicCounts = D.topics.map(() => 0);
    const counts = D.sets.map(() => D.topics.map(() => 0));
    visible = [];
    D.points.forEach((p, i) => {
      const n = nodes[i];
      const on = state.setOn[p.s] && state.topicOn[p.t];
      const match = !q || p.q.toLowerCase().includes(q) || p.sec.toLowerCase().includes(q);
      n.classList.toggle("off", !on);
      n.classList.toggle("dim", on && !match);
      n.style.transform = `translate(${xy[i][0]}px, ${xy[i][1]}px)`;
      paint(n, state.colorBy === "set" ? p.s : p.t);
      // Legend counts follow search and the *other* legend's toggles, so they read as
      // "how many questions would appear if this entry were on".
      if (match && state.topicOn[p.t]) setCounts[p.s]++;
      if (match && state.setOn[p.s]) topicCounts[p.t]++;
      if (on && match) { visible.push(i); counts[p.s][p.t]++; }
    });
    syncSetLegend(setCounts); syncTopicLegend(topicCounts);
    document.getElementById("count").textContent =
      `Showing ${visible.length} of ${D.points.length} questions` + (q ? ` matching "${state.query}"` : "");
    if (state.pinned != null && visible.includes(state.pinned)) showTip(state.pinned, true);
    else { state.pinned = null; hideTip(); }
    renderCoverage(counts); renderList();
  }
  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
