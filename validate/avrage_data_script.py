"""some what promt engineerd script to avrage by n"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# run folder name: seed_n  (e.g., 111_5)
_RUN_RE = re.compile(r"^(?P<seed>\d+)_(?P<n>\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate tracking metrics JSONs across runs grouped by n objects."
    )
    p.add_argument(
        "data_root",
        type=str,
        help="Upper folder containing run folders like 111_5, 579_5, ...",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder for summaries. Default: <data_root>/summaries",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="metrics_*.json",
        help="Glob for metrics json inside each run folder (default: metrics_*.json).",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _mean_std(vals: List[float]) -> Tuple[float | None, float | None]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _model_name_from_results_file(results_file: str) -> str:
    """
    Your scorer writes results_file like:
      - resultsp3.csv -> model "p3"
      - resultsv6.csv -> model "v6"
      - results.csv   -> model "results"
    """
    stem = Path(results_file).stem.lower()  # drops .csv
    if stem.startswith("results") and len(stem) > len("results"):
        suffix = stem[len("results") :].lstrip("_-")
        return suffix or "results"
    return stem


def _model_name_from_metrics_filename(metrics_json: Path) -> str:
    """
    If metrics file is like:
      metrics_resultsp3.json -> model "p3"
      metrics_resultsv6.json -> model "v6"
      metrics_results.json   -> model "results"
    """
    name = metrics_json.stem.lower()  # metrics_resultsp3
    if name.startswith("metrics_"):
        name = name[len("metrics_") :]
    if name.startswith("results") and len(name) > len("results"):
        suffix = name[len("results") :].lstrip("_-")
        return suffix or "results"
    return name


def _flatten_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten the parts of metrics you care about into numeric scalar values.
    (Edits here are the easiest way to control what's averaged.)
    """
    out: Dict[str, float] = {}

    # groups in your scorer output
    counts = m.get("counts", {})
    coverage = m.get("coverage", {})
    identity = m.get("identity", {})
    composite = m.get("composite", {})
    track_mgmt = m.get("track_management", {})
    performance = m.get("performance", {})  # where avg_fps should live

    def add(prefix: str, d: Dict[str, Any], keys: List[str]) -> None:
        for k in keys:
            v = _safe_float(d.get(k))
            if v is not None:
                out[f"{prefix}.{k}"] = v

    add("counts", counts, ["total_gt", "total_tr", "matches", "FN", "FP"])
    add("coverage", coverage, ["recall", "precision"])
    add(
        "identity",
        identity,
        ["total_id_switches", "avg_swaps_per_gt", "total_frags", "avg_frags_per_gt", "IDF1"],
    )
    add("composite", composite, ["MOTA", "MOTP"])
    add("track_management", track_mgmt, ["mean_abs_count_error"])
    add("performance", performance, ["avg_fps", "duration_s", "frames_processed"])

    return out


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data_root: {data_root}")

    out_dir = Path(args.out) if args.out else (data_root / "summaries")
    out_dir.mkdir(parents=True, exist_ok=True)

    # n -> model -> list of flattened metric dicts
    grouped: Dict[int, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    provenance: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # scan run folders
    run_folders = [p for p in sorted(data_root.iterdir()) if p.is_dir() and _RUN_RE.match(p.name)]
    if not run_folders:
        raise FileNotFoundError(f"No run folders like <seed>_<n> found in {data_root}")

    for run in run_folders:
        m = _RUN_RE.match(run.name)
        if not m:
            continue
        n = int(m.group("n"))

        metrics_files = sorted(run.glob(args.glob))
        if not metrics_files:
            continue

        for mf in metrics_files:
            try:
                payload = json.loads(mf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skip] {mf}: can't read json ({e})")
                continue

            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                print(f"[skip] {mf}: missing 'metrics' dict")
                continue

            # model name: prefer results_file inside JSON, else infer from mf name
            results_file = payload.get("results_file")
            if isinstance(results_file, str) and results_file.strip():
                model = _model_name_from_results_file(results_file)
            else:
                model = _model_name_from_metrics_filename(mf)

            flat = _flatten_metrics(metrics)
            if not flat:
                print(f"[skip] {mf}: no numeric metrics found")
                continue

            grouped[n][model].append(flat)
            provenance[n][model].append(run.name)

    if not grouped:
        raise RuntimeError(
            f"No metrics jsons found matching '{args.glob}'. "
            "Make sure your scoring script created metrics_*.json in each run folder."
        )

    # aggregate and write summaries per n
    for n in sorted(grouped.keys()):
        summary: Dict[str, Any] = {"n_objects": n, "models": {}}

        print("=" * 72)
        print(f"n = {n}")

        for model in sorted(grouped[n].keys()):
            rows = grouped[n][model]
            all_keys = sorted(set().union(*[r.keys() for r in rows]))

            means: Dict[str, float | None] = {}
            stds: Dict[str, float | None] = {}

            for k in all_keys:
                vals = [r[k] for r in rows if k in r]
                mu, sd = _mean_std(vals)
                means[k] = mu
                stds[k] = sd

            model_block = {
                "num_runs": len(rows),
                "run_folders": sorted(set(provenance[n][model])),
                "mean": means,
                "std": stds,
            }
            summary["models"][model] = model_block

            # console quick view
            def g(key: str) -> float | None:
                v = means.get(key)
                return None if v is None else float(v)

            print(f"\n  model: {model} | runs: {len(rows)}")
            print(f"    mean MOTA:      {g('composite.MOTA')}")
            print(f"    mean MOTP:      {g('composite.MOTP')}")
            print(f"    mean IDF1:      {g('identity.IDF1')}")
            print(f"    mean swaps/gt:  {g('identity.avg_swaps_per_gt')}")
            print(f"    mean frags/gt:  {g('identity.avg_frags_per_gt')}")
            print(f"    mean recall:    {g('coverage.recall')}")
            print(f"    mean precision: {g('coverage.precision')}")
            print(f"    mean fps:       {g('performance.avg_fps')}")

        out_path = out_dir / f"summary_n{n}.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[saved] {out_path}")

    print("=" * 72)
    print(f"Done. Summaries in: {out_dir}")


if __name__ == "__main__":
    main()