#!/usr/bin/env python3
"""
benchmark.py -- comprehensive benchmarking driver for the DISSCO render pipeline.

Runs a cmod render across a matrix of configurations (thread counts x
repetitions), captures both wall-clock and the in-process per-stage timings
emitted by StageProfiler.h (DISSCO_PROFILE=1), aggregates them, and writes:

  <out>/results.csv    one row per (threads, rep)
  <out>/results.json   full structured results + aggregates
  <out>/report.html    self-contained visual report (open or publish as artifact)
  ... and prints a console summary that ranks stages to spotlight bottlenecks.

The driver is deterministic-friendly: it pins a fixed <Seed> and, per run,
rewrites <NumberOfThreads> in a scratch copy of the .dissco so the original
project file is never touched. It records each output's md5 so you can see
where multi-threaded FP-composite nondeterminism kicks in.

Usage:
  benchmark.py PROJECT.dissco [--threads "1 2 4 8 20"] [--reps 2]
               [--seed 42] [--out DIR] [--cmod PATH] [--label NAME]

Requires only Python 3 + the cmod binary. No third-party packages.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # restructure/profiling -> repo root

# Stage keys as emitted by StageProfiler.h, in report order.
WALL_STAGES = ["parse + config", "event-tree build", "render + join", "write AIFF"]
AGG_STAGES = ["loudness", "partial synth", "sound reverb", "spatialize"]
OTHER_STAGES = ["composite drain", "final reverb", "clip management"]


def rewrite_dissco(src: Path, dst: Path, threads: int, seed: int):
    """Copy src->dst, forcing <NumberOfThreads> and <Seed>."""
    text = src.read_text()
    text = re.sub(r"<NumberOfThreads>\s*\d*\s*</NumberOfThreads>",
                  f"<NumberOfThreads>{threads}</NumberOfThreads>", text)
    text = re.sub(r"<Seed>\s*[^<]*</Seed>",
                  f"<Seed>{seed}</Seed>", text)
    dst.write_text(text)


def md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(cmod: Path, proj: Path, workdir: Path, threads: int, seed: int):
    """Render once; return (wall_s, stages_dict, out_md5, ok)."""
    workdir.mkdir(parents=True, exist_ok=True)
    bench = workdir / proj.name
    rewrite_dissco(proj, bench, threads, seed)
    stages_path = workdir / "stages.json"
    env = dict(os.environ)
    env["DISSCO_PROFILE"] = "1"
    env["DISSCO_PROFILE_OUT"] = str(stages_path)
    log = open(workdir / "render.log", "w")
    t0 = time.perf_counter()
    proc = subprocess.run([str(cmod), str(bench)], input=b"1\n",
                          stdout=log, stderr=subprocess.STDOUT, env=env)
    wall = time.perf_counter() - t0
    log.close()
    ok = proc.returncode == 0
    stages = {}
    if stages_path.exists():
        stages = json.loads(stages_path.read_text()).get("stages", {})
    # locate the rendered aiff
    aiffs = list((workdir / "SoundFiles").glob("*.aiff")) if ok else []
    out_md5 = md5(aiffs[0]) if aiffs else ""
    return wall, stages, out_md5, ok


def ms(stages, key):
    return float(stages.get(key, {}).get("ms", 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="path to a .dissco project")
    ap.add_argument("--threads", default="1 2 4 8 20",
                    help='space-separated thread counts (default "1 2 4 8 20")')
    ap.add_argument("--reps", type=int, default=2, help="repetitions per config")
    ap.add_argument("--seed", type=int, default=42, help="fixed random seed")
    ap.add_argument("--out", type=Path, default=HERE / "bench_runs" / "sweep",
                    help="output directory")
    ap.add_argument("--cmod", type=Path, default=REPO / "cmod",
                    help="cmod binary (default: repo-root ./cmod)")
    ap.add_argument("--label", default=None, help="label for the report")
    args = ap.parse_args()

    proj = args.project.resolve()
    cmod = args.cmod.resolve()
    if not proj.exists():
        sys.exit(f"no such project: {proj}")
    if not cmod.exists():
        sys.exit(f"no such cmod binary: {cmod}  (build with: make cmod config=release)")
    thread_list = [int(t) for t in args.threads.split()]
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    label = args.label or proj.stem

    print(f"# DISSCO benchmark: {label}")
    print(f"  project : {proj}")
    print(f"  cmod    : {cmod}")
    print(f"  threads : {thread_list}   reps: {args.reps}   seed: {args.seed}")
    print(f"  out     : {out}\n")

    rows = []  # per (threads, rep)
    for t in thread_list:
        for r in range(args.reps):
            wd = out / f"t{t}_r{r}"
            print(f"  running threads={t:>3} rep={r} ...", end="", flush=True)
            wall, stages, out_md5, ok = run_one(cmod, proj, wd, t, args.seed)
            print(f" {wall:7.2f}s  md5={out_md5[:8]}  {'ok' if ok else 'FAIL'}")
            row = {"threads": t, "rep": r, "wall_s": round(wall, 4),
                   "md5": out_md5, "ok": ok,
                   "total_ms": ms(stages, "TOTAL (piece)")}
            for k in WALL_STAGES + AGG_STAGES + OTHER_STAGES:
                row[k] = round(ms(stages, k), 3)
            rows.append(row)

    # ---- aggregate per thread count (min wall = best-case; median for report) ----
    aggregates = []
    for t in thread_list:
        grp = [x for x in rows if x["threads"] == t and x["ok"]]
        if not grp:
            continue
        walls = [x["wall_s"] for x in grp]
        # representative stage breakdown: the rep whose wall is the median
        rep_row = sorted(grp, key=lambda x: x["wall_s"])[len(grp) // 2]
        md5s = sorted({x["md5"] for x in grp})
        aggregates.append({
            "threads": t,
            "wall_min": round(min(walls), 3),
            "wall_median": round(statistics.median(walls), 3),
            "wall_max": round(max(walls), 3),
            "deterministic": len(md5s) == 1,
            "stage_ms": {k: rep_row[k] for k in
                         WALL_STAGES + AGG_STAGES + OTHER_STAGES},
            "total_ms": rep_row["total_ms"],
            "render_join_ms": rep_row["render + join"],
        })

    result = {"label": label, "project": str(proj), "seed": args.seed,
              "reps": args.reps, "threads": thread_list,
              "rows": rows, "aggregates": aggregates}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    write_csv(out / "results.csv", rows)
    write_report(out / "report.html", result)
    print_console_summary(result)
    print(f"\n  wrote: {out}/results.csv")
    print(f"         {out}/results.json")
    print(f"         {out}/report.html")


def write_csv(path, rows):
    import csv
    cols = ["threads", "rep", "wall_s", "md5", "ok", "total_ms"] + \
        WALL_STAGES + AGG_STAGES + OTHER_STAGES
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_console_summary(result):
    aggs = result["aggregates"]
    if not aggs:
        print("no successful runs"); return
    base = aggs[0]["wall_min"]
    print("\n================= scaling (wall clock) =================")
    print(f"{'threads':>7} {'wall_min':>9} {'median':>8} {'speedup':>8} "
          f"{'det?':>5}")
    for a in aggs:
        sp = base / a["wall_min"] if a["wall_min"] else 0
        print(f"{a['threads']:>7} {a['wall_min']:>9.2f} {a['wall_median']:>8.2f} "
              f"{sp:>7.2f}x {'yes' if a['deterministic'] else 'no':>5}")

    # stage breakdown from the single-thread run (aggregate == wall there)
    a = aggs[0]
    sm = a["stage_ms"]
    agg_total = sum(sm[k] for k in AGG_STAGES) or 1.0
    print("\n=========== render sub-stage cost (1 thread) ===========")
    print(f"{'stage':>16} {'ms':>10} {'share':>8}")
    ranked = sorted(AGG_STAGES, key=lambda k: -sm[k])
    for k in ranked:
        print(f"{k:>16} {sm[k]:>10.1f} {100*sm[k]/agg_total:>7.1f}%")
    print(f"  -> bottleneck: {ranked[0]} ({100*sm[ranked[0]]/agg_total:.0f}% of synth)")
    print("========================================================")


# ---------------------------------------------------------------------------
# self-contained HTML report (no external assets; safe to publish as artifact)
# ---------------------------------------------------------------------------
def write_report(path, result):
    import html
    aggs = result["aggregates"]
    label = html.escape(result["label"])
    base = aggs[0]["wall_min"] if aggs else 1.0

    # scaling rows
    scale_rows = ""
    for a in aggs:
        sp = base / a["wall_min"] if a["wall_min"] else 0
        det = "yes" if a["deterministic"] else "no"
        scale_rows += (f"<tr><td>{a['threads']}</td><td>{a['wall_min']:.2f}</td>"
                       f"<td>{a['wall_median']:.2f}</td><td>{sp:.2f}×</td>"
                       f"<td>{det}</td></tr>")

    # stage breakdown (1-thread run)
    palette = {"loudness": "#4e79a7", "partial synth": "#59a14f",
               "sound reverb": "#e15759", "spatialize": "#f28e2b",
               "composite drain": "#b07aa1", "final reverb": "#76b7b2",
               "clip management": "#9c755f", "parse + config": "#bab0ac",
               "event-tree build": "#ff9da7", "write AIFF": "#8cd17d"}
    a0 = aggs[0] if aggs else {"stage_ms": {}}
    sm = a0["stage_ms"]
    agg_total = sum(sm.get(k, 0) for k in AGG_STAGES) or 1.0
    bar = ""
    for k in sorted(AGG_STAGES, key=lambda k: -sm.get(k, 0)):
        pct = 100 * sm.get(k, 0) / agg_total
        bar += (f'<div class="seg" style="width:{pct:.2f}%;background:{palette[k]}" '
                f'title="{html.escape(k)}: {sm.get(k,0):.0f}ms ({pct:.1f}%)"></div>')
    legend = ""
    for k in sorted(AGG_STAGES, key=lambda k: -sm.get(k, 0)):
        pct = 100 * sm.get(k, 0) / agg_total
        legend += (f'<li><span class="sw" style="background:{palette[k]}"></span>'
                   f'{html.escape(k)} — {sm.get(k,0):.0f} ms ({pct:.1f}%)</li>')

    # scaling bar chart (wall vs threads), normalized to base
    maxw = max((a["wall_min"] for a in aggs), default=1.0)
    sc_bars = ""
    for a in aggs:
        h = 100 * a["wall_min"] / maxw
        sp = base / a["wall_min"] if a["wall_min"] else 0
        sc_bars += (f'<div class="col"><div class="colbar" style="height:{h:.1f}%" '
                    f'title="{a["wall_min"]:.2f}s ({sp:.2f}×)"></div>'
                    f'<div class="collab">{a["threads"]}t<br>{a["wall_min"]:.1f}s</div></div>')

    # per-thread stacked stage bars (aggregate render sub-stages)
    stack = ""
    for a in aggs:
        segs = ""
        tot = sum(a["stage_ms"].get(k, 0) for k in AGG_STAGES) or 1.0
        for k in AGG_STAGES:
            pct = 100 * a["stage_ms"].get(k, 0) / tot
            segs += (f'<div class="seg" style="width:{pct:.2f}%;background:{palette[k]}" '
                     f'title="{html.escape(k)}: {pct:.1f}%"></div>')
        stack += (f'<div class="stackrow"><div class="stacklab">{a["threads"]}t</div>'
                  f'<div class="bar">{segs}</div></div>')

    doc = f"""<title>DISSCO benchmark — {label}</title>
<style>
  :root {{ --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fff; --bg:#fafafa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8e8; --muted:#999; --line:#333; --card:#1c1c1c; --bg:#141414; }}
  }}
  :root[data-theme="dark"] {{ --fg:#e8e8e8; --muted:#999; --line:#333; --card:#1c1c1c; --bg:#141414; }}
  :root[data-theme="light"] {{ --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fff; --bg:#fafafa; }}
  body {{ font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--fg); background:var(--bg); margin:0; padding:2rem; }}
  .wrap {{ max-width:860px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .sub {{ color:var(--muted); margin:0 0 1.5rem; font-size:.9rem; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:1.25rem 1.5rem; margin:1rem 0; }}
  h2 {{ font-size:1.05rem; margin:0 0 1rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
  th,td {{ text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--line); }}
  th:first-child,td:first-child {{ text-align:left; }}
  .bar {{ display:flex; height:26px; border-radius:6px; overflow:hidden; width:100%;
         background:var(--line); }}
  .seg {{ height:100%; }}
  ul.legend {{ list-style:none; padding:0; margin:1rem 0 0; font-size:.85rem;
              display:grid; grid-template-columns:1fr 1fr; gap:.3rem 1rem; }}
  ul.legend li {{ display:flex; align-items:center; }}
  .sw {{ width:12px; height:12px; border-radius:3px; margin-right:.5rem; display:inline-block; }}
  .chart {{ display:flex; align-items:flex-end; gap:1rem; height:180px; padding-top:1rem; }}
  .col {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
         flex:1; height:100%; }}
  .colbar {{ width:60%; background:#4e79a7; border-radius:4px 4px 0 0; min-height:2px; }}
  .collab {{ font-size:.75rem; color:var(--muted); margin-top:.4rem; text-align:center; }}
  .stackrow {{ display:flex; align-items:center; gap:.6rem; margin:.35rem 0; }}
  .stacklab {{ width:34px; font-size:.8rem; color:var(--muted); }}
  code {{ background:var(--line); padding:.1rem .3rem; border-radius:4px; font-size:.85em; }}
</style>
<div class="wrap">
  <h1>DISSCO render benchmark — {label}</h1>
  <p class="sub">seed {result['seed']} · {result['reps']} reps/config · per-stage timing via
     <code>StageProfiler</code>. Wall phases are non-overlapping; render sub-stages are
     aggregate CPU across worker threads.</p>

  <div class="card">
    <h2>Where the compute goes (1 thread)</h2>
    <div class="bar">{bar}</div>
    <ul class="legend">{legend}</ul>
  </div>

  <div class="card">
    <h2>Thread scaling (wall clock, best of {result['reps']})</h2>
    <div class="chart">{sc_bars}</div>
  </div>

  <div class="card">
    <h2>Scaling table</h2>
    <table>
      <tr><th>threads</th><th>wall min (s)</th><th>median (s)</th><th>speedup</th><th>deterministic</th></tr>
      {scale_rows}
    </table>
  </div>

  <div class="card">
    <h2>Render sub-stage mix by thread count</h2>
    {stack}
    <ul class="legend">{legend}</ul>
  </div>
</div>
"""
    path.write_text(doc)


if __name__ == "__main__":
    main()
