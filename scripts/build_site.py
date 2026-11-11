#!/usr/bin/env python3
"""Builds the static leaderboard site from results/**/*.json.

Reads the exact result JSON files LemonMatrix (the CLI/dashboard) writes --
this repo never computes or measures anything itself, it only renders what
each contributor's own Lemonade instance already reported. Filtering and
sorting happen client-side in the browser, so the output is a single
self-contained index.html with no server and no build framework: plain
GitHub Pages, "deploy from branch main / root", is enough to host it.

Usage: python3 scripts/build_site.py [--results-dir results] [--out index.html]
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_RUN_TYPES = {None, "model", "router"}
MODALITY_DIRS = {
    "classify": ("Classification", "classifications_per_sec", "class/sec"),
    "tts": ("Text-to-speech", "real_time_factor", "× real time"),
    "stt": ("Speech-to-text", "real_time_factor", "× real time"),
    "imagegen": ("Image generation", "images_per_sec", "img/sec"),
    "audiogen": ("Audio generation", "real_time_factor", "× real time"),
    "embeddings": ("Embeddings", "embeddings_per_sec", "emb/sec"),
    "rerank": ("Reranking", "documents_per_sec", "doc/sec"),
    "meshgen": ("3D generation", "meshes_per_sec", "mesh/sec"),
}
SKIP_DIRS = {"failures", "trials"}
# Results brought in with `lemonmatrix import` from other tools. They skipped
# LemonMatrix's validity checks, so they get their own unranked section.
IMPORTED_DIR = "imported"

LEADERBOARD_COLUMNS = [
    ("timestamp", "Date", "date"),
    ("_profile", "Profile", "text"),
    ("_device", "Device", "text"),
    ("config.compute_engine", "Engine", "chip"),
    ("config.backend", "Backend", "text"),
    ("config.power_state", "Power", "text"),
    ("metrics.decode.tokens_per_sec", "Decode tok/s ↑", "num1"),
    ("metrics.prefill.tokens_per_sec", "Prefill tok/s ↑", "num1"),
    ("metrics.ttft_ms", "TTFT ms ↓", "num0"),
    ("metrics.decode.joules_per_token", "J/tok (decode) ↓", "num3"),
    ("validity.valid", "Valid", "status"),
]


def get_path(result: dict, dotted: str) -> Any:
    value: Any = result
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


ENGINE_DEVICE_KEYS = {"cpu": "cpu", "dgpu": "dgpu", "igpu": "igpu", "npu": "npu", "router": "dgpu"}


def device_of(result: dict) -> str | None:
    """The device that ran the model; environment.device_model is the host CPU."""
    env = result.get("environment") or {}
    key = ENGINE_DEVICE_KEYS.get(get_path(result, "config.compute_engine"))
    name = key and env.get(key)
    # Some drivers report only a PCI-style number as the GPU name.
    if name and name.isdigit():
        name = f"{env.get(key + '_family') or 'unknown'} GPU"
    return name or env.get("device_model")


def collect(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Returns (model_and_router_rows, modality_rows)."""
    model_rows: list[dict] = []
    modality_rows: list[dict] = []

    if not results_dir.exists():
        return model_rows, modality_rows

    for profile_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        profile = profile_dir.name
        if profile == IMPORTED_DIR:
            continue

        # Top-level files directly under <profile>/ are model/router runs.
        for path in sorted(profile_dir.glob("*.json")):
            data = load_json(path)
            if data is None:
                continue
            data["_profile"] = profile
            data["_run_id"] = path.stem
            data["_device"] = device_of(data)
            model_rows.append(data)

        # Modality subdirectories.
        for sub in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
            if sub.name in SKIP_DIRS or sub.name not in MODALITY_DIRS:
                continue
            label, metric_key, metric_unit = MODALITY_DIRS[sub.name]
            for path in sorted(sub.glob("*.json")):
                data = load_json(path)
                if data is None:
                    continue
                modality_rows.append(
                    {
                        "profile": profile,
                        "modality": sub.name,
                        "modality_label": label,
                        "timestamp": data.get("timestamp"),
                        "model": get_path(data, "model.name"),
                        "device": device_of(data),
                        "engine": get_path(data, "config.compute_engine"),
                        "backend": get_path(data, "config.backend"),
                        "metric_value": get_path(data, f"metrics.{metric_key}"),
                        "metric_unit": metric_unit,
                        "valid": get_path(data, "validity.valid"),
                        "run_id": path.stem,
                    }
                )

    model_rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    modality_rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return model_rows, modality_rows


def collect_imported(results_dir: Path) -> list[dict]:
    docs = [d for d in (load_json(p) for p in sorted(results_dir.glob(f"{IMPORTED_DIR}/*/*.json"))) if d]
    return sorted(docs, key=lambda d: d.get("measured_at") or "", reverse=True)


DASH = '<span class="muted">--</span>'
VALID = '<span class="status-ok">valid</span>'
INVALID = '<span class="status-bad">invalid</span>'


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def _short_model(name: str) -> str:
    """openrouter.nvidia/nemotron-3.5-super-vl:free -> nemotron-3.5-super-vl:free, full name on hover."""
    short = name.rsplit("/", 1)[-1]
    if short == name:
        return _esc(name)
    return f'<span title="{_esc(name)}">{_esc(short)}</span>'


def _per_1k(cost: Any, requests: int) -> str:
    if cost is None or not requests:
        return DASH
    return f"${cost / requests * 1000:.2f}"


def _baseline(routing: dict) -> str:
    model = routing.get("baseline_model")
    if not model:
        return DASH
    return f"{_short_model(model)} <span class=\"muted\">({_esc(routing.get('baseline_method') or 'estimated')})</span>"


def render_router_costs(model_rows: list[dict]) -> str:
    rows = [r for r in model_rows if get_path(r, "metrics.routing")]
    if not rows:
        return ""
    body = []
    for r in rows:
        routing = r["metrics"]["routing"]
        split = " &middot; ".join(
            f"{_short_model(name)} {c['share'] * 100:.0f}%" for name, c in routing["candidates"].items()
        )
        saved = routing.get("savings_pct")
        requests = routing["requests"]
        timestamp = r.get("timestamp") or ""
        body.append(
            "<tr>"
            f'<td><span title="{_esc(timestamp)}">{_esc(timestamp[:10])}</span></td>'
            f"<td>{_esc(r.get('_profile'))}</td>"
            f"<td>{_esc(get_path(r, 'model.name'))}</td>"
            f'<td class="num"><strong>{_num(saved, 1)}{"%" if saved is not None else ""}</strong></td>'
            f'<td class="num">{_per_1k(routing["routed_cost_usd"], requests)}</td>'
            f'<td class="num">{_per_1k(routing.get("baseline_cost_usd"), requests)}</td>'
            f"<td>{_baseline(routing)}</td>"
            f"<td>{split}</td>"
            f'<td class="num">{routing["local_share"] * 100:.0f}%</td>'
            f'<td class="num">{requests}</td>'
            f"<td>{VALID if get_path(r, 'validity.valid') else INVALID}</td>"
            "</tr>"
        )
    return f"""
  <div class="card">
    <h2>Router cost runs</h2>
    <p class="hint">
      A <code>collection.router</code> run with a mixed prompt set: where the router sent each request, what that cost,
      and what sending every request to the baseline model would have cost. A <em>measured</em> baseline sends each prompt
      once to the baseline model and prices its own token counts; an <em>estimated</em> one reuses the routed model's counts,
      which understates a reasoning model's cost. Prices are USD per million tokens, from Lemonade's own cost metadata or
      the submitter's <code>--cost</code>; non-cloud models count as free. Costs are shown per 1,000 requests.
    </p>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th>Date</th><th>Profile</th><th>Router</th><th class="num">Saved ↑</th><th class="num">Cost / 1k req</th>
          <th class="num">Baseline / 1k req</th><th>Baseline</th><th>Split</th><th class="num">Local</th>
          <th class="num">Requests</th><th>Valid</th>
        </tr></thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>
  </div>
"""


def render_imported(docs: list[dict]) -> str:
    if not docs:
        return ""
    body, notes = [], []
    for d in docs:
        env, source = d.get("environment") or {}, d.get("source") or {}
        for row in d.get("rows") or []:
            body.append(
                "<tr>"
                f"<td>{_esc(source.get('tool'))}</td>"
                f"<td>{_esc(source.get('submitted_by')) or DASH}</td>"
                f"<td>{_esc(env.get('cpu'))}</td>"
                f"<td>{_esc(row.get('workload'))}</td>"
                f"<td>{_esc(row.get('model'))}</td>"
                f"<td>{_esc(row.get('scenario')) or DASH}</td>"
                f'<td><span class="chip">{_esc(row.get("compute_engine") or "?")}</span></td>'
                f"<td>{_esc(row.get('backend'))}</td>"
                f'<td class="num">{_num(row.get("value"), 2)} {_esc(row.get("unit"))}</td>'
                f'<td class="num">{_num(row.get("ttft_ms"), 0)}</td>'
                f'<td class="num">{_num(row.get("joules_per_token"), 3)}</td>'
                "</tr>"
            )
        for warning in (d.get("validity") or {}).get("warnings") or []:
            notes.append(f'<li class="status-bad">{_esc(warning)}</li>')
    warnings = f'<ul class="hint">{"".join(notes)}</ul>' if notes else ""
    return f"""
  <div class="card">
    <h2>Imported results (unvalidated)</h2>
    <p class="hint">
      Results from other Lemonade benchmark tools (lemonade_lab / amdaibenchmarks, LemonMetrics), brought in with
      <code>lemonmatrix import</code>. LemonMatrix's validity checks never ran on these numbers, so they are shown as
      reported and never ranked with the leaderboard above.
    </p>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th>Source</th><th>Submitted by</th><th>Machine</th><th>Workload</th><th>Model</th><th>Scenario</th>
          <th>Engine</th><th>Backend</th><th class="num">Value</th><th class="num">TTFT ms</th><th class="num">J/token</th>
        </tr></thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>
    {warnings}
  </div>
"""


def render(
    model_rows: list[dict], modality_rows: list[dict], generated_at: str, imported_docs: list[dict] | None = None
) -> str:
    router_costs_html = render_router_costs(model_rows)
    imported_html = render_imported(imported_docs or [])
    # A router's tok/s blends its candidates, so it is not ranked against single models.
    llm_rows = [r for r in model_rows if r.get("run_type") != "router"]
    router_rows = [r for r in model_rows if r.get("run_type") == "router"]
    columns = [c for c in LEADERBOARD_COLUMNS if any(get_path(r, c[0]) is not None for r in llm_rows)]
    columns_json = json.dumps(columns)
    model_rows_json = json.dumps(llm_rows)
    modality_rows_json = json.dumps(modality_rows)
    modality_meta_json = json.dumps(
        {key: {"label": label, "unit": unit} for key, (label, _metric, unit) in MODALITY_DIRS.items()}
    )

    total = len(llm_rows)
    valid = sum(1 for r in llm_rows if get_path(r, "validity.valid"))
    profiles = sorted({r.get("_profile") for r in model_rows if r.get("_profile")})

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LemonMatrix Leaderboard</title>
<meta name="description" content="Community-submitted benchmark results for AMD's Lemonade runtime, collected with LemonMatrix.">
<style>
:root {{
  --bg: #ffffff;
  --bg-alt: #f6f7f9;
  --fg: #1c1f26;
  --fg-muted: #5b6270;
  --border: #e2e5ea;
  --accent: #d97706;
  --accent-fg: #ffffff;
  --ok: #1a7f37;
  --bad: #cf222e;
  --chip-bg: #fff3e0;
  --chip-fg: #9a5b00;
  --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14161a;
    --bg-alt: #1b1e24;
    --fg: #e7e9ee;
    --fg-muted: #9aa1ad;
    --border: #2a2e37;
    --accent: #f0a94e;
    --accent-fg: #1c1300;
    --ok: #4ade80;
    --bad: #f87171;
    --chip-bg: #3a2a10;
    --chip-fg: #f0c078;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  line-height: 1.5;
}}
a {{ color: var(--accent); }}
header.hero {{
  padding: 2.5rem 1.5rem 1.5rem;
  max-width: 1100px;
  margin: 0 auto;
}}
header.hero h1 {{
  margin: 0 0 .35rem;
  font-size: 1.9rem;
  letter-spacing: -.02em;
}}
header.hero .tagline {{
  color: var(--fg-muted);
  max-width: 62ch;
  margin: 0 0 .75rem;
}}
header.hero .links {{
  font-size: .9rem;
  color: var(--fg-muted);
}}
header.hero .links a {{ margin-right: 1rem; }}
main {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
}}
.stat-row {{
  display: flex;
  gap: .75rem;
  flex-wrap: wrap;
  margin: 1rem 0 1.5rem;
}}
.stat-tile {{
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .85rem 1.1rem;
  min-width: 130px;
}}
.stat-tile .value {{ font-size: 1.5rem; font-weight: 600; }}
.stat-tile .label {{ font-size: .78rem; color: var(--fg-muted); margin-top: .15rem; }}
.card {{
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.35rem;
  margin-bottom: 1.75rem;
}}
.card h2 {{
  margin: 0 0 1rem;
  font-size: 1.05rem;
}}
.card p.hint {{
  color: var(--fg-muted);
  font-size: .85rem;
  margin: -.5rem 0 1rem;
}}
.filters {{
  display: flex;
  gap: .9rem;
  flex-wrap: wrap;
  align-items: end;
  margin-bottom: 1rem;
}}
.filters label {{
  display: block;
  font-size: .75rem;
  color: var(--fg-muted);
  margin-bottom: .3rem;
}}
.filters select, .filters input[type=text] {{
  background: var(--bg);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: .4rem .55rem;
  font-size: .85rem;
  font-family: var(--sans);
}}
.filters .checkbox {{
  display: flex;
  align-items: center;
  gap: .4rem;
  font-size: .85rem;
}}
.filters .checkbox label {{ margin: 0; color: var(--fg); }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: .84rem;
}}
.table-scroll {{ overflow-x: auto; }}
th, td {{
  padding: .5rem .6rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
th {{
  cursor: pointer;
  user-select: none;
  color: var(--fg-muted);
  font-weight: 600;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .03em;
}}
th:hover {{ color: var(--fg); }}
th .arrow {{ opacity: .5; font-size: .7em; margin-left: .2em; }}
th.rank, td.rank {{ width: 2.5rem; color: var(--fg-muted); text-align: right; cursor: default; }}
tr.group-row td {{
  background: var(--bg);
  font-weight: 600;
  padding-top: .9rem;
  border-bottom: 2px solid var(--border);
}}
tr.group-row .muted {{ font-weight: 400; margin-left: .5rem; }}
td.num {{ font-family: var(--mono); text-align: right; }}
.chip {{
  display: inline-block;
  background: var(--chip-bg);
  color: var(--chip-fg);
  border-radius: 999px;
  padding: .1rem .55rem;
  font-size: .75rem;
  font-weight: 600;
}}
.status-ok {{ color: var(--ok); font-weight: 600; }}
.status-bad {{ color: var(--bad); font-weight: 600; }}
.muted {{ color: var(--fg-muted); }}
.modality-tabs {{
  display: flex;
  gap: .4rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}}
.modality-tabs button {{
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--fg-muted);
  border-radius: 999px;
  padding: .35rem .85rem;
  font-size: .78rem;
  cursor: pointer;
}}
.modality-tabs button.active {{
  background: var(--accent);
  color: var(--accent-fg);
  border-color: var(--accent);
}}
footer {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem 1.5rem 3rem;
  color: var(--fg-muted);
  font-size: .82rem;
  border-top: 1px solid var(--border);
}}
footer a {{ color: var(--accent); }}
.empty-row td {{ color: var(--fg-muted); font-style: italic; }}
</style>
</head>
<body>

<header class="hero">
  <h1>🍋 LemonMatrix Leaderboard</h1>
  <p class="tagline">
    Community-submitted benchmark results for
    <a href="https://github.com/lemonade-sdk/lemonade" target="_blank" rel="noopener">AMD's Lemonade runtime</a>,
    collected with <a href="https://github.com/Bekhouche/LemonMatrix" target="_blank" rel="noopener">LemonMatrix</a> --
    every number here was measured by a contributor's own machine talking to its own local Lemonade instance;
    this site only renders what Lemonade itself reported, never anything read from the machine directly.
  </p>
  <p class="links">
    <a href="https://github.com/Bekhouche/LemonMatrix" target="_blank" rel="noopener">LemonMatrix (code)</a>
    <a href="https://github.com/LemonMatrix/lemonmatrix.github.io" target="_blank" rel="noopener">This repo (results + site)</a>
    <a href="https://github.com/LemonMatrix/lemonmatrix.github.io/tree/main/results" target="_blank" rel="noopener">Submit a result via PR</a>
  </p>
</header>

<main>

  <div class="stat-row">
    <div class="stat-tile"><div class="value">{total}</div><div class="label">LLM runs</div></div>
    <div class="stat-tile"><div class="value">{valid}</div><div class="label">Valid runs</div></div>
    <div class="stat-tile"><div class="value">{len(profiles)}</div><div class="label">Profiles (machines)</div></div>
    <div class="stat-tile"><div class="value">{len(modality_rows)}</div><div class="label">Other modality runs</div></div>
    <div class="stat-tile"><div class="value">{len(router_rows)}</div><div class="label">Router runs</div></div>
  </div>

  <div class="card">
    <h2>LLM leaderboard</h2>
    <p class="hint">
      Runs are grouped by model and quantization, and ranked by decode speed within each group. Click a column header to
      rank by another column. ↑ higher is better, ↓ lower is better. Filters are applied together (AND).
    </p>
    <div class="filters">
      <div>
        <label for="f-profile">Profile</label>
        <select id="f-profile"><option value="">All</option></select>
      </div>
      <div>
        <label for="f-engine">Engine</label>
        <select id="f-engine"><option value="">All</option></select>
      </div>
      <div>
        <label for="f-backend">Backend</label>
        <select id="f-backend"><option value="">All</option></select>
      </div>
      <div>
        <label for="f-model">Model contains</label>
        <input type="text" id="f-model" placeholder="e.g. Qwen3">
      </div>
      <div class="checkbox">
        <input type="checkbox" id="f-valid" checked>
        <label for="f-valid">Valid only</label>
      </div>
    </div>
    <div class="table-scroll">
      <table id="leaderboard">
        <thead><tr id="lb-head"></tr></thead>
        <tbody id="lb-body"></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Other modalities</h2>
    <p class="hint">
      Non-LLM Lemonade recipes -- text-to-speech, speech-to-text, image/audio/3D generation, embeddings, reranking,
      classification. Each has its own throughput metric, so these aren't ranked against the table above. Every metric
      here is higher-is-better; for audio, × real time is seconds of audio per second of processing.
    </p>
    <div class="modality-tabs" id="modality-tabs"></div>
    <div class="table-scroll">
      <table id="modality-table">
        <thead>
          <tr>
            <th>Date</th><th>Profile</th><th>Model</th><th>Backend</th><th>Engine</th>
            <th class="num">Metric ↑</th><th>Valid</th>
          </tr>
        </thead>
        <tbody id="modality-body"></tbody>
      </table>
    </div>
  </div>
{router_costs_html}{imported_html}
</main>

<footer>
  <p>
    Generated {generated_at}. Every result is validated against
    <a href="https://github.com/Bekhouche/LemonMatrix/tree/main/schema" target="_blank" rel="noopener">LemonMatrix's JSON schemas</a>
    by CI before merge (<code>lemonmatrix validate-submission</code>) -- built for AMD's Lemonade Developer Challenge.
  </p>
</footer>

<script id="lm-columns" type="application/json">{columns_json}</script>
<script id="lm-model-rows" type="application/json">{model_rows_json}</script>
<script id="lm-modality-rows" type="application/json">{modality_rows_json}</script>
<script id="lm-modality-meta" type="application/json">{modality_meta_json}</script>
<script>
(function () {{
  "use strict";

  function getPath(obj, dotted) {{
    var parts = dotted.split(".");
    var value = obj;
    for (var i = 0; i < parts.length; i++) {{
      if (value === null || typeof value !== "object") return null;
      value = value[parts[i]];
      if (value === undefined) return null;
    }}
    return value;
  }}

  function fmt(value, kind) {{
    if (value === null || value === undefined || value === "") return '<span class="muted">--</span>';
    if (kind === "num1") return Number(value).toFixed(1);
    if (kind === "num0") return Math.round(Number(value)).toString();
    if (kind === "num3") return Number(value).toFixed(3);
    if (kind === "date") return '<span title="' + escapeHtml(String(value)) + '">' + escapeHtml(String(value).slice(0, 10)) + '</span>';
    if (kind === "chip") return '<span class="chip">' + escapeHtml(String(value)) + '</span>';
    if (kind === "status") return value ? '<span class="status-ok">valid</span>' : '<span class="status-bad">invalid</span>';
    return escapeHtml(String(value));
  }}

  function escapeHtml(s) {{
    return s.replace(/[&<>"']/g, function (c) {{
      return {{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[c];
    }});
  }}

  var columns = JSON.parse(document.getElementById("lm-columns").textContent);
  var modelRows = JSON.parse(document.getElementById("lm-model-rows").textContent);
  var modalityRows = JSON.parse(document.getElementById("lm-modality-rows").textContent);
  var modalityMeta = JSON.parse(document.getElementById("lm-modality-meta").textContent);

  // ---- Model / router leaderboard ------------------------------------

  var sortKey = "metrics.decode.tokens_per_sec";
  var sortDir = "desc";

  function groupOf(r) {{
    var name = (getPath(r, "model.name") || "").replace(/^user\\./, "");
    var quant = getPath(r, "model.quantization");
    return quant && quant !== "none" ? name + " · " + quant : name;
  }}

  var headRow = document.getElementById("lb-head");
  var rankTh = document.createElement("th");
  rankTh.className = "rank";
  rankTh.textContent = "#";
  headRow.appendChild(rankTh);
  columns.forEach(function (col) {{
    var th = document.createElement("th");
    th.dataset.key = col[0];
    th.innerHTML = escapeHtml(col[1]) + '<span class="arrow"></span>';
    th.addEventListener("click", function () {{
      if (sortKey === col[0]) {{
        sortDir = sortDir === "asc" ? "desc" : "asc";
      }} else {{
        sortKey = col[0];
        sortDir = "desc";
      }}
      renderLeaderboard();
    }});
    headRow.appendChild(th);
  }});

  function uniqueSorted(rows, path) {{
    var seen = {{}};
    rows.forEach(function (r) {{
      var v = getPath(r, path);
      if (v) seen[v] = true;
    }});
    return Object.keys(seen).sort();
  }}

  function fillSelect(id, values) {{
    var el = document.getElementById(id);
    values.forEach(function (v) {{
      var opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      el.appendChild(opt);
    }});
  }}

  fillSelect("f-profile", uniqueSorted(modelRows, "_profile"));
  fillSelect("f-engine", uniqueSorted(modelRows, "config.compute_engine"));
  fillSelect("f-backend", uniqueSorted(modelRows, "config.backend"));

  function renderLeaderboard() {{
    var profile = document.getElementById("f-profile").value;
    var engine = document.getElementById("f-engine").value;
    var backend = document.getElementById("f-backend").value;
    var modelNeedle = document.getElementById("f-model").value.toLowerCase();
    var validOnly = document.getElementById("f-valid").checked;

    var rows = modelRows.filter(function (r) {{
      if (profile && r._profile !== profile) return false;
      if (engine && getPath(r, "config.compute_engine") !== engine) return false;
      if (backend && getPath(r, "config.backend") !== backend) return false;
      if (modelNeedle && (getPath(r, "model.name") || "").toLowerCase().indexOf(modelNeedle) === -1) return false;
      if (validOnly && !getPath(r, "validity.valid")) return false;
      return true;
    }});

    var groupSize = {{}};
    rows.forEach(function (r) {{ var g = groupOf(r); groupSize[g] = (groupSize[g] || 0) + 1; }});

    rows.sort(function (a, b) {{
      var ga = groupOf(a), gb = groupOf(b);
      if (ga !== gb) {{
        if (groupSize[ga] !== groupSize[gb]) return groupSize[gb] - groupSize[ga];
        return ga < gb ? -1 : 1;
      }}
      var av = getPath(a, sortKey);
      var bv = getPath(b, sortKey);
      var aNull = av === null || av === undefined;
      var bNull = bv === null || bv === undefined;
      if (aNull !== bNull) return aNull ? 1 : -1;
      if (aNull && bNull) return 0;
      var cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === "desc" ? -cmp : cmp;
    }});

    var arrows = headRow.querySelectorAll("th");
    arrows.forEach(function (th) {{
      var arrow = th.querySelector(".arrow");
      if (!arrow) return;
      if (th.dataset.key === sortKey) {{
        arrow.textContent = sortDir === "desc" ? "▼" : "▲";
      }} else {{
        arrow.textContent = "";
      }}
    }});

    var body = document.getElementById("lb-body");
    body.innerHTML = "";
    if (rows.length === 0) {{
      var tr = document.createElement("tr");
      tr.className = "empty-row";
      var td = document.createElement("td");
      td.colSpan = columns.length + 1;
      td.textContent = "No runs match the current filters.";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }}
    var currentGroup = null;
    var rank = 0;
    rows.forEach(function (r) {{
      var group = groupOf(r);
      if (group !== currentGroup) {{
        currentGroup = group;
        rank = 0;
        var gtr = document.createElement("tr");
        gtr.className = "group-row";
        var gtd = document.createElement("td");
        gtd.colSpan = columns.length + 1;
        var n = groupSize[group];
        gtd.innerHTML = escapeHtml(group) + '<span class="muted">' + n + (n === 1 ? " run" : " runs") + '</span>';
        gtr.appendChild(gtd);
        body.appendChild(gtr);
      }}
      rank += 1;
      var tr = document.createElement("tr");
      var rankTd = document.createElement("td");
      rankTd.className = "rank";
      rankTd.textContent = rank;
      tr.appendChild(rankTd);
      columns.forEach(function (col) {{
        var td = document.createElement("td");
        var kind = col[2];
        if (kind === "num1" || kind === "num0" || kind === "num3") td.className = "num";
        td.innerHTML = fmt(getPath(r, col[0]), kind);
        tr.appendChild(td);
      }});
      body.appendChild(tr);
    }});
  }}

  ["f-profile", "f-engine", "f-backend"].forEach(function (id) {{
    document.getElementById(id).addEventListener("change", renderLeaderboard);
  }});
  document.getElementById("f-model").addEventListener("input", renderLeaderboard);
  document.getElementById("f-valid").addEventListener("change", renderLeaderboard);

  renderLeaderboard();

  // ---- Other modalities ------------------------------------------------

  var activeModality = "all";
  var tabsEl = document.getElementById("modality-tabs");

  var counts = {{}};
  modalityRows.forEach(function (r) {{ counts[r.modality] = (counts[r.modality] || 0) + 1; }});

  function makeTab(key, label) {{
    var btn = document.createElement("button");
    btn.textContent = label;
    btn.className = key === activeModality ? "active" : "";
    btn.addEventListener("click", function () {{
      activeModality = key;
      Array.prototype.forEach.call(tabsEl.children, function (b) {{ b.classList.remove("active"); }});
      btn.classList.add("active");
      renderModalityTable();
    }});
    tabsEl.appendChild(btn);
  }}

  makeTab("all", "All (" + modalityRows.length + ")");
  Object.keys(modalityMeta).forEach(function (key) {{
    if (counts[key]) makeTab(key, modalityMeta[key].label + " (" + counts[key] + ")");
  }});

  function renderModalityTable() {{
    var body = document.getElementById("modality-body");
    body.innerHTML = "";
    var rows = modalityRows.filter(function (r) {{ return activeModality === "all" || r.modality === activeModality; }});
    if (rows.length === 0) {{
      var tr = document.createElement("tr");
      tr.className = "empty-row";
      var td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = "No runs in this category yet.";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }}
    rows.forEach(function (r) {{
      var tr = document.createElement("tr");
      var dateTd = document.createElement("td");
      dateTd.innerHTML = fmt(r.timestamp, "date");
      tr.appendChild(dateTd);
      var cells = [
        r.profile,
        r.model,
        r.backend,
        r.engine,
      ];
      cells.forEach(function (v) {{
        var td = document.createElement("td");
        td.innerHTML = fmt(v, "text");
        tr.appendChild(td);
      }});
      var metricTd = document.createElement("td");
      metricTd.className = "num";
      metricTd.textContent = (r.metric_value === null || r.metric_value === undefined)
        ? "--"
        : Number(r.metric_value).toFixed(2) + " " + r.metric_unit;
      tr.appendChild(metricTd);
      var validTd = document.createElement("td");
      validTd.innerHTML = fmt(r.valid, "status");
      tr.appendChild(validTd);
      body.appendChild(tr);
    }});
  }}

  renderModalityTable();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="index.html")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = (repo_root / args.results_dir).resolve()
    out_path = (repo_root / args.out).resolve()

    model_rows, modality_rows = collect(results_dir)
    imported_docs = collect_imported(results_dir)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = render(model_rows, modality_rows, generated_at, imported_docs)
    out_path.write_text(page)
    print(
        f"Wrote {out_path} ({len(model_rows)} model/router rows, {len(modality_rows)} modality rows, "
        f"{sum(len(d.get('rows') or []) for d in imported_docs)} imported rows)"
    )


if __name__ == "__main__":
    main()
