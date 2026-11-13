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


CSS = """/* Colors, type, spacing and radii follow Lemonade GUI3 (src/app/src/styles/tokens.css, src/app/DESIGN.md). */
:root {
  --surface-base: #0E0E0B;
  --surface-1: #1A1813;
  --surface-2: #1F1D17;
  --surface-3: #25221B;
  --surface-alpha-05: rgba(255, 255, 255, 0.05);
  --surface-alpha-02: rgba(255, 255, 255, 0.02);
  --text-primary: #F2EFE5;
  --text-secondary: #C7C2B5;
  --text-tertiary: #A8A39A;
  --accent: #FCD846;
  --accent-fg: #FCD846;
  --accent-on: #000000;
  --accent-soft: rgba(252, 216, 70, 0.14);
  --accent-border: rgba(252, 216, 70, 0.28);
  --success: #78A86B;
  --danger: #EF8D7F;
  --border: rgba(255, 255, 255, 0.10);
  --border-subtle: rgba(255, 255, 255, 0.07);
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", system-ui, "Inter",
               "Helvetica Neue", sans-serif;
  --font-mono: "SF Mono", "JetBrains Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --text-xs: 11px;
  --text-sm: 12.5px;
  --text-base: 14px;
  --text-2xl: 24px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --radius-pill: 9999px;
  --control-height: 36px;
}
@media (prefers-color-scheme: light) {
  :root {
    --surface-base: #fdfbf6;
    --surface-1: #fdfbf6;
    --surface-2: #f5f2ec;
    --surface-3: #fdfbf6;
    --surface-alpha-05: rgba(0, 0, 0, 0.05);
    --surface-alpha-02: rgba(0, 0, 0, 0.02);
    --text-primary: #000000;
    --text-secondary: #2D2F2F;
    --text-tertiary: #52525B;
    --accent: rgba(252, 216, 70, 0.58);
    --accent-fg: #5C4B00;
    --accent-soft: rgba(252, 216, 70, 0.18);
    --accent-border: rgba(92, 75, 0, 0.32);
    --success: #2C702F;
    --danger: #C62828;
    --border: rgba(0, 0, 0, 0.16);
    --border-subtle: rgba(92, 75, 0, 0.18);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--surface-base);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.5;
}
a { color: var(--accent-fg); text-underline-offset: 2px; }
a:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible, th:focus-visible {
  outline: 2px solid var(--accent-fg);
  outline-offset: 2px;
}
header.hero {
  padding: var(--space-6) var(--space-5) var(--space-4);
  max-width: 1100px;
  margin: 0 auto;
}
header.hero h1 {
  margin: 0 0 var(--space-2);
  font-size: var(--text-2xl);
  font-weight: 600;
  letter-spacing: -0.01em;
}
header.hero .tagline {
  color: var(--text-secondary);
  max-width: 68ch;
  margin: 0 0 var(--space-3);
}
header.hero .links { font-size: var(--text-sm); }
header.hero .links a { margin-right: var(--space-4); }
main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 var(--space-5) var(--space-6);
}
.stat-row {
  display: flex;
  gap: var(--space-3);
  flex-wrap: wrap;
  margin: var(--space-4) 0 var(--space-5);
}
.stat-tile {
  background: var(--surface-1);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  min-width: 130px;
}
.stat-tile .value { font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
.stat-tile .label { font-size: var(--text-xs); color: var(--text-tertiary); margin-top: 2px; }
.stat-tile.headline { border-color: var(--accent-border); background: var(--accent-soft); max-width: 340px; }
.stat-tile.headline .value { color: var(--accent-fg); }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  margin-bottom: var(--space-5);
}
.card h2 {
  margin: 0 0 var(--space-2);
  font-size: var(--text-base);
  font-weight: 600;
}
.card p.hint {
  color: var(--text-tertiary);
  font-size: var(--text-sm);
  line-height: 1.35;
  margin: 0 0 var(--space-4);
}
.filters {
  display: flex;
  gap: var(--space-3);
  flex-wrap: wrap;
  align-items: end;
  margin-bottom: var(--space-4);
}
.filters label {
  display: block;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: var(--space-1);
}
.filters select, .filters input[type=text] {
  height: var(--control-height);
  background: var(--surface-2);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0 var(--space-3);
  font-size: var(--text-sm);
  font-family: var(--font-sans);
}
.filters .checkbox {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  height: var(--control-height);
  font-size: var(--text-sm);
}
.filters .checkbox input { accent-color: #FCD846; }
.filters .checkbox label {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-weight: 400;
  letter-spacing: 0;
  text-transform: none;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}
.table-scroll { overflow-x: auto; }
th, td {
  padding: var(--space-2) var(--space-3);
  text-align: left;
  border-bottom: 1px solid var(--border-subtle);
  white-space: nowrap;
}
th {
  cursor: pointer;
  user-select: none;
  color: var(--text-tertiary);
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
th:hover { color: var(--text-primary); }
th .arrow { color: var(--accent-fg); font-size: .8em; margin-left: .25em; }
th.rank, td.rank { width: 2.5rem; color: var(--text-tertiary); text-align: right; cursor: default; }
tbody tr:not(.group-row):hover td { background: var(--surface-alpha-02); }
tr.group-row td {
  padding-top: var(--space-4);
  font-weight: 600;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border);
}
tr.group-row td::before {
  content: "";
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: var(--space-2);
  border-radius: 50%;
  background: #FCD846;
  vertical-align: 1px;
}
tr.group-row .muted { font-weight: 400; margin-left: var(--space-2); }
td.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; text-align: right; }
.chip {
  display: inline-block;
  background: var(--surface-alpha-05);
  color: var(--text-secondary);
  border-radius: var(--radius-sm);
  padding: 3px var(--space-2);
  font-size: var(--text-xs);
  font-weight: 500;
}
.status-ok { color: var(--success); font-weight: 500; }
.status-bad { color: var(--danger); font-weight: 500; }
.muted { color: var(--text-tertiary); }
.modality-tabs {
  display: inline-flex;
  gap: 2px;
  flex-wrap: wrap;
  padding: 3px;
  margin-bottom: var(--space-4);
  background: var(--surface-2);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-pill);
}
.modality-tabs button {
  background: transparent;
  border: 0;
  color: var(--text-secondary);
  border-radius: var(--radius-pill);
  padding: 6px var(--space-3);
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  cursor: pointer;
}
.modality-tabs button:hover { color: var(--text-primary); }
.modality-tabs button.active {
  background: #FCD846;
  color: var(--accent-on);
  font-weight: 600;
}
code { font-family: var(--font-mono); font-size: .92em; }
footer {
  max-width: 1100px;
  margin: 0 auto;
  padding: var(--space-5) var(--space-5) var(--space-6);
  color: var(--text-tertiary);
  font-size: var(--text-sm);
  border-top: 1px solid var(--border-subtle);
}
.empty-row td { color: var(--text-tertiary); font-style: italic; }
.more-runs { font-size: var(--text-xs); margin-left: var(--space-1); }
#leaderboard td.wrap { white-space: normal; min-width: 9rem; max-width: 15rem; }
tr.clickable { cursor: pointer; }
tr.clickable td a.run-link { color: inherit; text-decoration: none; }
tr.clickable:hover td a.run-link { color: var(--accent-fg); text-decoration: underline; }
"""

RUN_CSS = """
.run-page header.hero, .run-page main, .run-page footer { max-width: 1100px; }
.crumbs { font-size: var(--text-sm); color: var(--text-tertiary); margin-bottom: var(--space-2); }
.crumbs a { margin-right: var(--space-2); }
.run-meta { display: flex; flex-wrap: wrap; gap: var(--space-2); align-items: center; margin-top: var(--space-2); }
.run-meta .muted { font-size: var(--text-sm); }
.stat-tile .sub { font-size: var(--text-xs); color: var(--text-tertiary); font-family: var(--font-mono); margin-top: 2px; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: var(--space-4); }
.grid-2 .card { margin: 0; }
table.kv td:first-child { color: var(--text-tertiary); width: 40%; white-space: nowrap; }
table.kv td { white-space: normal; word-break: break-word; }
.checks { list-style: none; padding: 0; margin: 0 0 var(--space-3); display: flex; flex-wrap: wrap; gap: var(--space-2); }
.checks li {
  padding: 2px var(--space-3); border-radius: var(--radius-pill); border: 1px solid var(--border);
  font-size: var(--text-sm);
}
.checks li.ok { color: var(--success); }
.checks li.bad { color: var(--danger); border-color: var(--danger); }
ul.notes { margin: 0; padding-left: 1.25rem; color: var(--text-secondary); font-size: var(--text-sm); }
td.bar { width: 22%; min-width: 80px; }
.run-page header.hero .tagline { max-width: none; }
.bar-track { height: 8px; background: var(--surface-alpha-05); border-radius: var(--radius-pill); overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: var(--radius-pill); }
tr.current td { background: var(--accent-soft); }
tr.current td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.raw-links a { margin-right: var(--space-4); font-size: var(--text-sm); }
"""


HIDE_WHEN_CONSTANT = {"config.power_state"}
WRAP_COLUMNS = {"_profile", "_device"}


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


VENDOR_MARKERS = [
    ("AMD", ("AMD", "Radeon", "Ryzen", "EPYC", "Instinct", "gfx")),
    ("NVIDIA", ("NVIDIA", "GeForce", "RTX", "TITAN", "Tesla", "sm_")),
    ("Intel", ("Intel",)),
    ("Apple", ("Apple",)),
]


def vendor_of(device: str | None) -> str:
    for vendor, markers in VENDOR_MARKERS:
        if device and any(m in device for m in markers):
            return vendor
    return "Other"


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
            data["_vendor"] = vendor_of(data["_device"])
            data["_rel_path"] = path.relative_to(results_dir).as_posix()
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
                        "_rel_path": path.relative_to(results_dir).as_posix(),
                        "_data": data,
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


def group_of(result: dict) -> str:
    """Same grouping as the page script: model without the user. prefix, plus quantization."""
    name = (get_path(result, "model.name") or "").removeprefix("user.")
    quant = get_path(result, "model.quantization")
    return f"{name} · {quant}" if quant and quant != "none" else name


def _fastest_tile(title: str, group: str, run: dict) -> str:
    return (
        f'<div class="stat-tile headline"><div class="value">'
        f'{get_path(run, "metrics.decode.tokens_per_sec"):.0f} tok/s</div><div class="label">'
        f"{_esc(title)}, {_esc(group)}: {_esc(run.get('_device'))} ({_esc(run.get('_profile'))}), "
        f"{_esc(get_path(run, 'config.backend'))}</div></div>"
    )


def render_headlines(llm_rows: list[dict], router_rows: list[dict]) -> str:
    tiles = []
    valid = [r for r in llm_rows if get_path(r, "validity.valid") and get_path(r, "metrics.decode.tokens_per_sec")]
    if valid:
        groups: dict[str, list[dict]] = {}
        for r in valid:
            groups.setdefault(group_of(r), []).append(r)
        group, runs = max(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
        decode = lambda r: get_path(r, "metrics.decode.tokens_per_sec")  # noqa: E731
        best = max(runs, key=decode)
        amd = [r for r in runs if r.get("_vendor") == "AMD"]
        best_amd = max(amd, key=decode) if amd else None
        if best_amd is not None:
            tiles.append(_fastest_tile("Fastest on AMD", group, best_amd))
        if best is not best_amd:
            tiles.append(_fastest_tile("Fastest overall", group, best))
    priced = [
        r for r in router_rows
        if get_path(r, "validity.valid") and get_path(r, "metrics.routing.savings_pct") is not None
    ]
    if priced:
        latest = max(priced, key=lambda r: r.get("timestamp") or "")
        routing = latest["metrics"]["routing"]
        method = routing.get("baseline_method") or "estimated"
        tiles.append(
            f'<div class="stat-tile headline"><div class="value">{routing["savings_pct"]:.1f}% saved</div>'
            f'<div class="label">Latest router run ({_esc(get_path(latest, "model.name"))}), '
            f"{_esc(method)} baseline</div></div>"
        )
    return "".join(tiles)


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
            f'<tr class="clickable" data-href="{_esc(run_href(r.get("_run_id")))}">'
            f'<td><a class="run-link" href="{_esc(run_href(r.get("_run_id")))}" title="{_esc(timestamp)}">'
            f"{_esc(timestamp[:10])}</a></td>"
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
    # A setting every run shares (today: power state, always "plugged") only pushes the speed columns off screen.
    columns = [
        c for c in columns
        if c[0] not in HIDE_WHEN_CONSTANT or len({get_path(r, c[0]) for r in llm_rows}) > 1
    ]
    columns_json = json.dumps(columns)
    wrap_columns_json = json.dumps(sorted(WRAP_COLUMNS))
    model_rows_json = json.dumps(llm_rows)
    modality_rows_json = json.dumps([{k: v for k, v in r.items() if k != "_data"} for r in modality_rows])
    modality_meta_json = json.dumps(
        {key: {"label": label, "unit": unit} for key, (label, _metric, unit) in MODALITY_DIRS.items()}
    )

    headlines_html = render_headlines(llm_rows, router_rows)
    total = len(llm_rows)
    valid = sum(1 for r in llm_rows if get_path(r, "validity.valid"))
    profiles = sorted({r.get("_profile") for r in model_rows if r.get("_profile")})

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>LemonMatrix Leaderboard</title>
<meta name="description" content="Community-submitted benchmark results for AMD's Lemonade runtime, collected with LemonMatrix.">
<style>
{CSS}{GUIDE_CSS}</style>
</head>
<body>

<header class="hero">
  {nav("index.html")}
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
    <a href="guide.html">Submit a result: step by step</a>
  </p>
</header>

<main>

  <div class="stat-row">
    {headlines_html}
    <div class="stat-tile"><div class="value">{total}</div><div class="label">LLM runs</div></div>
    <div class="stat-tile"><div class="value">{valid}</div><div class="label">Valid runs</div></div>
    <div class="stat-tile"><div class="value">{len(profiles)}</div><div class="label">Profiles (machines)</div></div>
    <div class="stat-tile"><div class="value">{len(modality_rows)}</div><div class="label">Other modality runs</div></div>
    <div class="stat-tile"><div class="value">{len(router_rows)}</div><div class="label">Router runs</div></div>
  </div>

  <div class="card">
    <h2>LLM leaderboard</h2>
    <p class="hint">
      Runs are grouped by model and quantization, and ranked by decode speed within each group. Repeated runs of the same
      machine and backend are folded into their best one; untick the box to see them all. Click a column header to rank
      by another column. ↑ higher is better, ↓ lower is better. Filters are applied together (AND).
    </p>
    <div class="filters">
      <div>
        <label for="f-vendor">Vendor</label>
        <select id="f-vendor"><option value="">All</option></select>
      </div>
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
      <div class="checkbox">
        <input type="checkbox" id="f-best" checked>
        <label for="f-best">Best run per machine and backend</label>
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

  // Every run has its own page, runs/<run id>.html; the first cell is a real link, the rest of the row clicks through.
  function runHref(id) {{ return "runs/" + encodeURIComponent(id) + ".html"; }}
  function runLink(id, inner) {{
    return id ? '<a class="run-link" href="' + runHref(id) + '">' + inner + '</a>' : inner;
  }}
  function makeClickable(tr, id) {{
    if (!id) return;
    tr.className = "clickable";
    tr.dataset.href = runHref(id);
  }}
  document.addEventListener("click", function (e) {{
    if (e.target.closest("a, button, input, select")) return;
    var tr = e.target.closest("tr.clickable");
    if (!tr || !tr.dataset.href) return;
    if (e.ctrlKey || e.metaKey) window.open(tr.dataset.href, "_blank");
    else window.location.href = tr.dataset.href;
  }});

  var columns = JSON.parse(document.getElementById("lm-columns").textContent);
  var wrapColumns = {wrap_columns_json};
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

  fillSelect("f-vendor", uniqueSorted(modelRows, "_vendor"));
  fillSelect("f-profile", uniqueSorted(modelRows, "_profile"));
  fillSelect("f-engine", uniqueSorted(modelRows, "config.compute_engine"));
  fillSelect("f-backend", uniqueSorted(modelRows, "config.backend"));

  function renderLeaderboard() {{
    var vendor = document.getElementById("f-vendor").value;
    var profile = document.getElementById("f-profile").value;
    var engine = document.getElementById("f-engine").value;
    var backend = document.getElementById("f-backend").value;
    var modelNeedle = document.getElementById("f-model").value.toLowerCase();
    var validOnly = document.getElementById("f-valid").checked;
    var bestOnly = document.getElementById("f-best").checked;

    var rows = modelRows.filter(function (r) {{
      if (vendor && r._vendor !== vendor) return false;
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

    // Rows are already ranked, so the first run of each machine, engine and backend in a group is its best one.
    var extra = {{}};
    if (bestOnly) {{
      var firstOf = {{}};
      rows = rows.filter(function (r) {{
        var key = groupOf(r) + "|" + r._profile + "|" + getPath(r, "config.compute_engine") + "|" + getPath(r, "config.backend");
        if (firstOf[key]) {{ extra[firstOf[key]] = (extra[firstOf[key]] || 0) + 1; return false; }}
        firstOf[key] = r._run_id;
        return true;
      }});
    }}

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
      makeClickable(tr, r._run_id);
      var rankTd = document.createElement("td");
      rankTd.className = "rank";
      rankTd.textContent = rank;
      tr.appendChild(rankTd);
      columns.forEach(function (col, i) {{
        var td = document.createElement("td");
        var kind = col[2];
        if (kind === "num1" || kind === "num0" || kind === "num3") td.className = "num";
        else if (wrapColumns.indexOf(col[0]) !== -1) td.className = "wrap";
        td.innerHTML = i === 0 ? runLink(r._run_id, fmt(getPath(r, col[0]), kind)) : fmt(getPath(r, col[0]), kind);
        if (i === 0 && extra[r._run_id]) {{
          td.innerHTML += ' <span class="muted more-runs">+' + extra[r._run_id] + (extra[r._run_id] === 1 ? " run" : " runs") + '</span>';
        }}
        tr.appendChild(td);
      }});
      body.appendChild(tr);
    }});
  }}

  ["f-vendor", "f-profile", "f-engine", "f-backend"].forEach(function (id) {{
    document.getElementById(id).addEventListener("change", renderLeaderboard);
  }});
  document.getElementById("f-model").addEventListener("input", renderLeaderboard);
  document.getElementById("f-valid").addEventListener("change", renderLeaderboard);
  document.getElementById("f-best").addEventListener("change", renderLeaderboard);

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
      makeClickable(tr, r.run_id);
      var dateTd = document.createElement("td");
      dateTd.innerHTML = runLink(r.run_id, fmt(r.timestamp, "date"));
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


GUIDE_CSS = """
nav.site-nav { font-size: var(--text-sm); margin: 0 0 var(--space-3); }
nav.site-nav a { margin-right: var(--space-4); }
nav.site-nav a[aria-current] { color: var(--text-primary); text-decoration: none; font-weight: 600; }
.guide-page header.hero, .guide-page main, .guide-page footer { max-width: 860px; }
.guide h2 { font-size: 18px; margin: 0 0 var(--space-2); letter-spacing: -0.01em; }
.guide h2 .step {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; margin-right: var(--space-2);
  border-radius: var(--radius-pill); background: #FCD846; color: #000; font-size: var(--text-sm);
}
.guide h3 { font-size: var(--text-base); margin: var(--space-4) 0 var(--space-1); }
.guide p, .guide li { color: var(--text-secondary); }
.guide ul, .guide ol { padding-left: 1.25rem; }
.guide li { margin: var(--space-1) 0; }
.guide pre {
  background: var(--surface-2); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4); overflow-x: auto; font-size: var(--text-sm); line-height: 1.5;
}
.guide pre code { color: var(--text-primary); }
.guide .note {
  border-left: 3px solid #FCD846; background: var(--accent-soft);
  border-radius: var(--radius-sm); padding: var(--space-2) var(--space-3); margin: var(--space-3) 0;
}
.guide .note p { margin: 0; color: var(--text-primary); }
.guide table { margin: var(--space-2) 0 var(--space-3); }
.guide td, .guide th { white-space: normal; }
.guide th { cursor: default; }
.toc ol { columns: 2; column-gap: var(--space-6); margin: 0; }
"""

GUIDE_BODY = """
<div class="card toc">
  <h2>On this page</h2>
  <ol>
    <li><a href="#before">Before you start</a></li>
    <li><a href="#install">Install LemonMatrix</a></li>
    <li><a href="#profile">Create a profile</a></li>
    <li><a href="#run">Run an LLM benchmark</a></li>
    <li><a href="#valid">Make the run valid</a></li>
    <li><a href="#other">Other modalities</a></li>
    <li><a href="#router">Router cost runs</a></li>
    <li><a href="#submit">Check and submit</a></li>
    <li><a href="#published">What gets published</a></li>
    <li><a href="#trouble">Troubleshooting</a></li>
  </ol>
</div>

<section class="card" id="before">
  <h2><span class="step">1</span>Before you start</h2>
  <ul>
    <li><a href="https://lemonade-server.ai" target="_blank" rel="noopener">Lemonade</a> installed and running on the
      machine you want to measure. LemonMatrix only talks to Lemonade's HTTP API; it never reads the hardware itself.</li>
    <li>The model you want to benchmark, pulled in Lemonade. The most compared model on this board is
      <code>Qwen3-1.7B-GGUF</code>:
      <pre><code>lemonade pull Qwen3-1.7B-GGUF</code></pre></li>
    <li>Python 3.10 or newer, and <code>git</code>.</li>
    <li>To submit with one command: the <a href="https://cli.github.com" target="_blank" rel="noopener">GitHub CLI</a>,
      logged in with <code>gh auth login</code>.</li>
  </ul>
</section>

<section class="card" id="install">
  <h2><span class="step">2</span>Install LemonMatrix</h2>
  <p>Use a virtual environment so nothing touches your system Python.</p>
  <h3>Linux and macOS</h3>
  <pre><code>python3 -m venv lemonmatrix-env
source lemonmatrix-env/bin/activate
pip install lemonmatrix</code></pre>
  <h3>Windows (PowerShell)</h3>
  <pre><code>py -m venv lemonmatrix-env
lemonmatrix-env\\Scripts\\Activate.ps1
pip install lemonmatrix</code></pre>
  <p>If PowerShell refuses to run <code>Activate.ps1</code>, allow scripts for this window only with
    <code>Set-ExecutionPolicy -Scope Process Bypass</code> and run it again.</p>
  <p>Check it works with <code>lemonmatrix --version</code>. LemonMatrix is on
    <a href="https://pypi.org/project/lemonmatrix/" target="_blank" rel="noopener">PyPI</a>; to upgrade later, run
    <code>pip install -U lemonmatrix</code>.</p>
</section>

<section class="card" id="profile">
  <h2><span class="step">3</span>Create a profile</h2>
  <p>A profile is one Lemonade instance. LemonMatrix asks Lemonade for the machine's CPU, GPU, NPU, OS and Lemonade
    version once and attaches that to every result from this profile.</p>
  <pre><code>lemonmatrix profile detect --save Radeon-RX-7900-XTX</code></pre>
  <p><code>detect</code> scans localhost on Lemonade's usual ports (13305 first, then the older 8000). If Lemonade runs
    on another machine, or needs an API key, add it directly:</p>
  <pre><code>lemonmatrix profile add Radeon-RX-7900-XTX --host 192.168.1.42 --port 13305 --token $LEMONADE_API_KEY</code></pre>
  <p>Check what was discovered:</p>
  <pre><code>lemonmatrix profile show Radeon-RX-7900-XTX</code></pre>
  <div class="note"><p>The profile name becomes the folder under <code>results/</code> and the Profile column on the
    leaderboard. Name it after the hardware, not after yourself or your computer's hostname.</p></div>
</section>

<section class="card" id="run">
  <h2><span class="step">4</span>Run an LLM benchmark</h2>
  <p>This measures prefill speed, decode speed and time to first token over 5 trials, after 2 warm-up requests:</p>
  <pre><code>lemonmatrix run --profile Radeon-RX-7900-XTX \\
  --model Qwen3-1.7B-GGUF --model-class dense --quant Q4_0 \\
  --context-length 4096 --engine dgpu --backend llamacpp-vulkan \\
  --power-state plugged</code></pre>
  <table>
    <thead><tr><th>Flag</th><th>What to put</th></tr></thead>
    <tbody>
      <tr><td><code>--model</code></td><td>The model name exactly as Lemonade lists it (<code>lemonade list</code>).</td></tr>
      <tr><td><code>--model-class</code></td><td><code>dense</code> or <code>moe</code>.</td></tr>
      <tr><td><code>--quant</code></td><td>The quantization in the model's file, e.g. <code>Q4_0</code> or <code>Q4_K_M</code>.</td></tr>
      <tr><td><code>--engine</code></td><td>Where it runs: <code>cpu</code>, <code>igpu</code>, <code>dgpu</code>, <code>npu</code> or <code>hybrid</code>.</td></tr>
      <tr><td><code>--backend</code></td><td>The backend Lemonade uses, e.g. <code>llamacpp-vulkan</code>, <code>llamacpp-rocm</code>, <code>llamacpp-cpu</code>.</td></tr>
      <tr><td><code>--power-state</code></td><td><code>plugged</code>, <code>battery</code> or <code>power_capped</code> (with <code>--power-cap-w</code>).</td></tr>
      <tr><td><code>--trials</code>, <code>--warmup</code></td><td>Optional. Defaults are 5 and 2.</td></tr>
    </tbody>
  </table>
  <p>The result is written to <code>results/&lt;profile&gt;/&lt;run_id&gt;.json</code>, and the command prints the speeds
    and whether the run is valid. Run the same model on another backend or engine to get a comparison row.</p>
</section>

<section class="card" id="valid">
  <h2><span class="step">5</span>Make the run valid</h2>
  <p>Only valid runs are ranked. LemonMatrix marks a run invalid, and says why in <code>validity.notes</code>, when:</p>
  <ul>
    <li>another model or app was using the GPU or NPU during the measured trials;</li>
    <li>the device Lemonade reports doesn't match <code>--engine</code>, or the backend doesn't fit the engine;</li>
    <li>Lemonade couldn't say which backend served the model;</li>
    <li>the model was reloaded in the middle of the run;</li>
    <li>the quantization doesn't match <code>--quant</code>;</li>
    <li>the hardware was thermally throttled.</li>
  </ul>
  <div class="note"><p>Close other GPU-heavy apps, keep a laptop plugged in, and let Lemonade load only the model being
    measured. <code>--force</code> runs past a mismatch, but the result is then always marked invalid.</p></div>
</section>

<section class="card" id="other">
  <h2><span class="step">6</span>Other modalities</h2>
  <p>Speech, image, embedding and other models have their own commands, because their speed is not measured in
    tokens. Each writes to its own folder and shows up under <em>Other modalities</em>:</p>
  <table>
    <thead><tr><th>Command</th><th>Measures</th></tr></thead>
    <tbody>
      <tr><td><code>lemonmatrix tts</code></td><td>Text to speech, seconds of audio per second</td></tr>
      <tr><td><code>lemonmatrix stt</code></td><td>Speech to text, seconds of audio per second</td></tr>
      <tr><td><code>lemonmatrix audiogen</code></td><td>Music and audio generation, seconds of audio per second</td></tr>
      <tr><td><code>lemonmatrix imagegen</code></td><td>Images per second</td></tr>
      <tr><td><code>lemonmatrix embeddings</code></td><td>Embeddings per second</td></tr>
      <tr><td><code>lemonmatrix rerank</code></td><td>Documents per second</td></tr>
      <tr><td><code>lemonmatrix classify</code></td><td>Classifications per second</td></tr>
      <tr><td><code>lemonmatrix meshgen</code></td><td>3D meshes per second</td></tr>
    </tbody>
  </table>
  <p>Each command's <code>--help</code> shows an example, for instance:</p>
  <pre><code>lemonmatrix tts --profile Radeon-RX-7900-XTX --model kokoro-v1 \\
  --engine cpu --backend kokoro-cpu --power-state plugged \\
  --input-text "Lemonade can speak"</code></pre>
</section>

<section class="card" id="router">
  <h2><span class="step">7</span>Router cost runs</h2>
  <p>A Lemonade <code>collection.router</code> can send easy prompts to a local model and hard ones to a cloud model.
    A router run records where each request went and what that cost, compared with sending everything to the most
    expensive candidate:</p>
  <pre><code>lemonmatrix run --profile Radeon-RX-7900-XTX --run-type router \\
  --model user.My-Router --context-length 4096 --power-state plugged \\
  --prompt-set router-mix --trials 20</code></pre>
  <ul>
    <li>Cloud prices come from Lemonade's cost metadata. <code>--cost MODEL=IN,OUT</code> sets a price in USD per million
      input and output tokens; local models count as free.</li>
    <li>After the trials, each prompt the router served locally is sent once to the baseline model, so the
      comparison uses the tokens that model really writes. <code>--estimate-baseline</code> skips these extra requests.</li>
    <li>Cloud API keys stay in Lemonade's environment (<code>LEMONADE_&lt;PROVIDER&gt;_API_KEY</code>); they are never
      written into the result.</li>
  </ul>
</section>

<section class="card" id="submit">
  <h2><span class="step">8</span>Check and submit</h2>
  <p>Check the file against the same schema the leaderboard's CI uses:</p>
  <pre><code>lemonmatrix validate-submission results/Radeon-RX-7900-XTX/&lt;run_id&gt;.json</code></pre>
  <p>Preview the pull request, then open it:</p>
  <pre><code>lemonmatrix submit results/Radeon-RX-7900-XTX/&lt;run_id&gt;.json --dry-run
lemonmatrix submit results/Radeon-RX-7900-XTX/&lt;run_id&gt;.json</code></pre>
  <p><code>submit</code> forks
    <a href="https://github.com/LemonMatrix/lemonmatrix.github.io" target="_blank" rel="noopener">LemonMatrix/lemonmatrix.github.io</a>
    (or reuses your fork), adds the file under the same <code>results/&lt;profile&gt;/</code> path, and opens a PR. CI
    validates it again; once it is merged, this site rebuilds and your run appears.</p>
  <h3>Without the GitHub CLI</h3>
  <ol>
    <li>Fork the repository on GitHub.</li>
    <li>Add your file at <code>results/&lt;profile&gt;/&lt;run_id&gt;.json</code> (modality results go in
      <code>results/&lt;profile&gt;/&lt;modality&gt;/</code>, e.g. <code>tts/</code>).</li>
    <li>Open a pull request to <code>main</code>.</li>
  </ol>
</section>

<section class="card" id="published">
  <h2><span class="step">9</span>What gets published</h2>
  <p>The result file is public once merged. It holds:</p>
  <ul>
    <li>the hardware names Lemonade reports (CPU, GPU, NPU), memory size, OS version, driver, Lemonade and backend versions;</li>
    <li>the model, quantization, engine, backend and run settings;</li>
    <li>the measured numbers and the validity checks.</li>
  </ul>
  <p>It does not hold your hostname, IP address, user name, the model's answers or API keys. Open the JSON and read it
    before you submit; it is plain text.</p>
</section>

<section class="card" id="trouble">
  <h2><span class="step">10</span>Troubleshooting</h2>
  <table>
    <thead><tr><th>Problem</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td><code>profile detect</code> finds nothing</td><td>Check Lemonade is running and note its port, then use
        <code>profile add --port</code>.</td></tr>
      <tr><td>401 or 403 from Lemonade</td><td>The instance has <code>LEMONADE_API_KEY</code> set; pass it with
        <code>--token</code>.</td></tr>
      <tr><td>The run is marked invalid</td><td>Read <code>validity.notes</code> in the JSON, fix the cause (see step 5) and
        run again.</td></tr>
      <tr><td><code>submit</code> fails with "Permission denied (publickey)"</td><td>The GitHub CLI clones over SSH without
        a key it can use. Run <code>gh config set git_protocol https</code> and try again.</td></tr>
    </tbody>
  </table>
  <p>Still stuck? Open an issue on <a href="https://github.com/Bekhouche/LemonMatrix/issues" target="_blank"
    rel="noopener">Bekhouche/LemonMatrix</a>.</p>
</section>
"""


def nav(current: str, prefix: str = "") -> str:
    links = [("index.html", "Leaderboard"), ("guide.html", "How to submit a result")]
    items = []
    for href, label in links:
        attr = ' aria-current="page"' if href == current else ""
        items.append(f'<a href="{prefix}{href}"{attr}>{label}</a>')
    return '<nav class="site-nav">' + "".join(items) + "</nav>"


# ---- Per-run detail pages ------------------------------------------------

RUNS_DIR = "runs"
REPO_BLOB = "https://github.com/LemonMatrix/lemonmatrix.github.io/blob/main/results/"
VALIDITY_LABELS = {
    "warmup_discarded": "Warm-up discarded",
    "thermal_ok": "Thermal OK",
    "exclusive_run": "Exclusive run",
    "model_reload_free": "No model reload",
}


def run_href(run_id: Any, prefix: str = "") -> str:
    return f"{prefix}{RUNS_DIR}/{run_id}.html"


def _fmt_value(value: Any) -> str:
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1000 else f"{value:,.1f}"
    if isinstance(value, (dict, list)):
        return f"<code>{_esc(json.dumps(value))}</code>"
    return _esc(value)


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(obj, dict):
        return [(prefix, obj)]
    items: list[tuple[str, Any]] = []
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        items.extend(_flatten(value, name) if isinstance(value, dict) else [(name, value)])
    return items


def _kv_table(pairs: list[tuple[str, Any]]) -> str:
    rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_fmt_value(v)}</td></tr>" for k, v in pairs if v is not None)
    return f'<table class="kv"><tbody>{rows or "<tr class=empty-row><td colspan=2>Not reported.</td></tr>"}</tbody></table>'


def _bar(value: Any, top: float) -> str:
    if not isinstance(value, (int, float)) or not top:
        return ""
    pct = max(2.0, min(100.0, value / top * 100))
    return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'


def _short_num(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _tile(value: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="stat-tile"><div class="value">{value}</div><div class="label">{_esc(label)}</div>{sub_html}</div>'


def _spread(block: dict) -> str:
    half = block.get("ci95_half_width")
    sd = block.get("stddev")
    parts = []
    if half is not None:
        parts.append(f"± {half:.3g} (95% CI)")
    if sd is not None:
        parts.append(f"sd {sd:.3g}")
    return " · ".join(parts)


def run_kind(data: dict, modality: str | None) -> str:
    if modality:
        return modality
    return "router" if data.get("run_type") == "router" else "llm"


def main_metric(data: dict, kind: str) -> tuple[Any, str]:
    if kind == "llm":
        return get_path(data, "metrics.decode.tokens_per_sec"), "decode tok/s"
    if kind == "router":
        return get_path(data, "metrics.routing.savings_pct"), "% saved"
    _label, key, unit = MODALITY_DIRS[kind]
    return get_path(data, f"metrics.{key}"), unit


def _metric_tiles(data: dict, kind: str) -> str:
    metrics = data.get("metrics") or {}
    tiles = []
    if kind == "llm":
        decode, prefill = metrics.get("decode") or {}, metrics.get("prefill") or {}
        if decode.get("tokens_per_sec") is not None:
            tiles.append(_tile(f"{decode['tokens_per_sec']:.1f}", "Decode tok/s ↑", _spread(decode)))
        if prefill.get("tokens_per_sec") is not None:
            tiles.append(_tile(f"{prefill['tokens_per_sec']:.0f}", "Prefill tok/s ↑", _spread(prefill)))
        if metrics.get("ttft_ms") is not None:
            half = metrics.get("ttft_ms_ci95_half_width")
            tiles.append(_tile(f"{metrics['ttft_ms']:.0f}", "TTFT ms ↓", f"± {half:.3g} (95% CI)" if half else ""))
        if decode.get("inter_token_latency_ms") is not None:
            tiles.append(_tile(f"{decode['inter_token_latency_ms']:.1f}", "Inter-token ms ↓"))
        if decode.get("joules_per_token") is not None:
            tiles.append(_tile(f"{decode['joules_per_token']:.3f}", "J/token (decode) ↓"))
    else:
        value, unit = main_metric(data, kind)
        if value is not None:
            tiles.append(_tile(_short_num(value), f"{unit} ↑"))
        for key, val in metrics.items():
            if (
                isinstance(val, (int, float)) and not isinstance(val, bool)
                and not key.endswith(("stddev", "p95", "half_width", "sha256"))
                and key not in ("trial_count", "peak_memory_gb") and len(tiles) < 5
                and f"{val:.3g}" != (f"{value:.3g}" if value is not None else None)
            ):
                tiles.append(_tile(_short_num(val), key.replace("_", " ")))
    if metrics.get("peak_memory_gb") is not None:
        tiles.append(_tile(f"{metrics['peak_memory_gb']:.1f}", "Peak memory GB"))
    if metrics.get("trial_count") is not None:
        tiles.append(_tile(str(metrics["trial_count"]), "Trials"))
    return "".join(tiles)


def _validity_card(data: dict) -> str:
    validity = data.get("validity") or {}
    checks = []
    for key, label in VALIDITY_LABELS.items():
        if key in validity:
            ok = bool(validity[key])
            checks.append(f'<li class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"} {_esc(label)}</li>')
    notes = [n.strip() for n in (validity.get("notes") or "").split(";") if n.strip()]
    notes_html = f'<ul class="notes">{"".join(f"<li>{_esc(n)}</li>" for n in notes)}</ul>' if notes else ""
    return f"""
  <div class="card">
    <h2>Validity: {VALID if validity.get("valid") else INVALID}</h2>
    <p class="hint">The checks LemonMatrix ran while measuring; an invalid run is shown but never ranked.</p>
    <ul class="checks">{"".join(checks)}</ul>
    {notes_html}
  </div>"""


def _trials_card(trials: list[dict] | None) -> str:
    if not trials:
        return ""
    keys: list[str] = []
    for t in trials:
        keys.extend(k for k in t if k not in keys)
    keys = [k for k in keys if any(t.get(k) is not None for t in trials)]
    bar_key = next((k for k in ("decode_tokens_per_sec", "tokens_per_sec") if k in keys), None)
    top = max((t.get(bar_key) or 0 for t in trials), default=0) if bar_key else 0
    head = "".join(f'<th class="num">{_esc(k.replace("_", " "))}</th>' for k in keys)
    rows = []
    for i, t in enumerate(trials, 1):
        cells = "".join(f'<td class="num">{_fmt_value(t.get(k))}</td>' for k in keys)
        bar = f'<td class="bar">{_bar(t.get(bar_key), top)}</td>' if bar_key else ""
        rows.append(f'<tr><td class="rank">{i}</td>{cells}{bar}</tr>')
    bar_head = f"<th>{_esc(bar_key.replace('_', ' '))}</th>" if bar_key else ""
    return f"""
  <div class="card">
    <h2>Trials</h2>
    <p class="hint">Each measured trial after warm-up, from the run's trial sidecar. The summary above is computed from these.</p>
    <div class="table-scroll"><table>
      <thead><tr><th class="rank">#</th>{head}{bar_head}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
  </div>"""


def _siblings_card(entry: dict, siblings: list[dict]) -> str:
    if len(siblings) < 2:
        return ""
    unit = main_metric(entry["data"], entry["kind"])[1]
    ordered = sorted(siblings, key=lambda s: (s["metric"] is None, -(s["metric"] or 0)))
    top = max((s["metric"] or 0 for s in ordered), default=0)
    rows = []
    for s in ordered:
        d = s["data"]
        cls = "current" if s["run_id"] == entry["run_id"] else "clickable"
        href = run_href(s["run_id"], "../")
        rows.append(
            f'<tr class="{cls}" data-href="{_esc(href)}">'
            f'<td><a class="run-link" href="{_esc(href)}">{_esc(s["profile"])}</a></td>'
            f"<td>{_esc(device_of(d))}</td>"
            f'<td><span class="chip">{_esc(get_path(d, "config.compute_engine"))}</span></td>'
            f"<td>{_esc(get_path(d, 'config.backend'))}</td>"
            f'<td class="num">{_num(s["metric"], 2)}</td>'
            f'<td class="bar">{_bar(s["metric"], top)}</td>'
            f"<td>{VALID if get_path(d, 'validity.valid') else INVALID}</td>"
            "</tr>"
        )
    return f"""
  <div class="card">
    <h2>Same model, other runs</h2>
    <p class="hint">Every run of {_esc(entry["group"])} on the board, ranked by {_esc(unit)}. This run is highlighted.</p>
    <div class="table-scroll"><table>
      <thead><tr><th>Profile</th><th>Device</th><th>Engine</th><th>Backend</th>
        <th class="num">{_esc(unit)} ↑</th><th></th><th>Valid</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
  </div>"""


def render_run_page(entry: dict, siblings: list[dict], trials: list[dict] | None, generated_at: str) -> str:
    d, kind = entry["data"], entry["kind"]
    model = get_path(d, "model.name") or "unknown model"
    kind_label = {"llm": "LLM", "router": "Router"}.get(kind) or MODALITY_DIRS[kind][0]
    chips = [kind_label, get_path(d, "model.quantization"), get_path(d, "model.class"),
             get_path(d, "config.compute_engine"), get_path(d, "config.backend")]
    chips_html = "".join(f'<span class="chip">{_esc(c)}</span>' for c in chips if c and c != "none")
    timestamp = d.get("timestamp") or ""
    trials_link = (
        f'<a href="{REPO_BLOB}{_esc(entry["profile"])}/trials/{_esc(entry["run_id"])}.json" target="_blank" '
        'rel="noopener">Trial sidecar</a>' if trials else ""
    )
    metrics_pairs = [(k, v) for k, v in _flatten(d.get("metrics") or {}) if k != "routing.candidates"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{_esc(model)} on {_esc(entry["profile"])} · LemonMatrix</title>
<meta name="description" content="LemonMatrix run {_esc(entry["run_id"])}: {_esc(model)} on {_esc(device_of(d))}.">
<style>
{CSS}{GUIDE_CSS}{RUN_CSS}</style>
</head>
<body class="run-page">

<header class="hero">
  {nav("", "../")}
  <div class="crumbs"><a href="../index.html">← Leaderboard</a> {_esc(kind_label)} run · <code>{_esc(entry["run_id"])}</code></div>
  <h1>{_esc(model)}</h1>
  <p class="tagline">{_esc(device_of(d))} · profile <strong>{_esc(entry["profile"])}</strong></p>
  <div class="run-meta">{chips_html}{VALID if get_path(d, "validity.valid") else INVALID}
    <span class="muted" title="{_esc(timestamp)}">{_esc(timestamp[:16].replace("T", " "))} UTC</span></div>
</header>

<main>
  <div class="stat-row">{_metric_tiles(d, kind)}</div>
{_validity_card(d)}
{_trials_card(trials)}
{_siblings_card(entry, siblings)}
  <div class="grid-2">
    <div class="card"><h2>Model and configuration</h2>{_kv_table(_flatten(d.get("model") or {}, "model") + _flatten(d.get("config") or {}, "config"))}</div>
    <div class="card"><h2>Environment</h2>{_kv_table(list((d.get("environment") or {}).items()))}</div>
  </div>
  <div class="card"><h2>All metrics</h2>{_kv_table(metrics_pairs)}</div>
  <div class="card raw-links">
    <h2>Raw data</h2>
    <a href="{REPO_BLOB}{_esc(entry["rel_path"])}" target="_blank" rel="noopener">Result JSON on GitHub</a>{trials_link}
    <span class="muted">schema {_esc(d.get("schema_version"))}</span>
  </div>
</main>

<footer><p>Generated {generated_at}. Numbers are as reported by the contributor's own Lemonade instance.</p></footer>
<script>
document.addEventListener("click", function (e) {{
  if (e.target.closest("a")) return;
  var tr = e.target.closest("tr.clickable");
  if (tr && tr.dataset.href) window.location.href = tr.dataset.href;
}});
</script>
</body>
</html>
"""


def write_run_pages(
    results_dir: Path, out_dir: Path, model_rows: list[dict], modality_rows: list[dict], generated_at: str
) -> int:
    entries = [
        {"data": r, "kind": run_kind(r, None), "profile": r["_profile"], "run_id": r["_run_id"], "rel_path": r["_rel_path"]}
        for r in model_rows
    ] + [
        {"data": r["_data"], "kind": r["modality"], "profile": r["profile"], "run_id": r["run_id"], "rel_path": r["_rel_path"]}
        for r in modality_rows
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        e["group"] = group_of(e["data"])
        e["metric"] = main_metric(e["data"], e["kind"])[0]
        groups.setdefault((e["kind"], e["group"]), []).append(e)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.html"):
        old.unlink()
    for e in entries:
        sidecar = load_json(results_dir / e["profile"] / "trials" / f"{e['run_id']}.json") or {}
        page = render_run_page(e, groups[(e["kind"], e["group"])], sidecar.get("trials"), generated_at)
        (out_dir / f"{e['run_id']}.html").write_text(page, encoding="utf-8")
    return len(entries)


def render_guide(generated_at: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Submit a result · LemonMatrix</title>
<meta name="description" content="Step-by-step guide to benchmarking a Lemonade instance with LemonMatrix and submitting the result.">
<style>
{CSS}{GUIDE_CSS}</style>
</head>
<body class="guide-page">

<header class="hero">
  {nav("guide.html")}
  <h1>🍋 How to submit a result</h1>
  <p class="tagline">
    Benchmark your own Lemonade instance with LemonMatrix and add the result to the public leaderboard. Each step
    below is a command you run on the machine you want to measure.
  </p>
</header>

<main class="guide">
{GUIDE_BODY}
</main>

<footer>
  <p>Generated {generated_at}. Built for AMD's Lemonade Developer Challenge.</p>
</footer>
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
    out_path.with_name("guide.html").write_text(render_guide(generated_at))
    pages = write_run_pages(results_dir, out_path.with_name(RUNS_DIR), model_rows, modality_rows, generated_at)
    print(
        f"Wrote {out_path} ({len(model_rows)} model/router rows, {len(modality_rows)} modality rows, "
        f"{sum(len(d.get('rows') or []) for d in imported_docs)} imported rows) and {pages} run pages"
    )


if __name__ == "__main__":
    main()
