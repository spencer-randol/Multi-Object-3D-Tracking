"""some what promt engineerd score script to eval runs"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tracking evaluation (MOTA/MOTP/IDF1 + swaps/frags).")
    p.add_argument(
        "folder",
        type=str,
        help="Folder containing ground_truth.csv and results*.csv (e.g., resultsp3.csv, resultsv6.csv, etc.)",
    )
    p.add_argument(
        "--gate",
        type=float,
        default=1.0,
        help="Euclidean gating threshold. Use a very large number to effectively disable.",
    )
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Output folder for JSON metric files. Default: same as input folder.",
    )
    return p.parse_args()


def count_id_switches_ignore_gaps(track_series: np.ndarray) -> int:
    idx = np.flatnonzero(np.isfinite(track_series))
    if idx.size <= 1:
        return 0
    vals = track_series[idx].astype(int)
    return int(np.sum(vals[1:] != vals[:-1]))


def count_frags(track_series: np.ndarray) -> int:
    finite = np.isfinite(track_series)
    if finite.sum() == 0:
        return 0
    starts = np.sum((~finite[:-1]) & (finite[1:])) + (1 if finite[0] else 0)
    return int(max(0, starts - 1))


def _require_columns(df: pd.DataFrame, name: str) -> None:
    for col in ["frame", "id", "x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"{name} missing required column '{col}'")


def _find_time_column(res: pd.DataFrame) -> str | None:
    """Return the time column name if present, else None."""
    for c in ["time(s)", "time", "t", "elapsed_s"]:
        if c in res.columns:
            return c
    return None


def _compute_avg_fps(res: pd.DataFrame) -> dict:
    """
    Compute average FPS using results' elapsed-time column.
    time(s) is assumed to be elapsed seconds since start of first processed frame.
    """
    time_col = _find_time_column(res)
    if time_col is None:
        return {"avg_fps": None, "duration_s": None, "frames_processed": int(res["frame"].nunique())}

    # Coerce to numeric, drop NaNs
    t = pd.to_numeric(res[time_col], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    frames_processed = int(res["frame"].nunique())

    if t.size == 0:
        return {"avg_fps": None, "duration_s": None, "frames_processed": frames_processed}

    t0 = float(np.min(t))
    t1 = float(np.max(t))
    duration_s = t1 - t0  # robust even if it doesn't start at 0

    if duration_s <= 0.0:
        # If timestamps are weird (all same), can't infer fps
        return {"avg_fps": None, "duration_s": float(duration_s), "frames_processed": frames_processed}

    avg_fps = frames_processed / duration_s
    return {"avg_fps": float(avg_fps), "duration_s": float(duration_s), "frames_processed": frames_processed}


def evaluate_one(gt: pd.DataFrame, res: pd.DataFrame, gate: float) -> dict:
    _require_columns(gt, "ground_truth.csv")
    _require_columns(res, "results")

    gt = gt.copy()
    res = res.copy()
    gt["frame"] = gt["frame"].astype(int)
    res["frame"] = res["frame"].astype(int)

    T = int(gt["frame"].max()) + 1
    gt_ids = sorted(gt["id"].unique())
    tr_ids = sorted(res["id"].unique())
    nG = len(gt_ids)
    nT = len(tr_ids)

    gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
    tr_id_to_idx = {tid: i for i, tid in enumerate(tr_ids)}

    gt_to_track = np.full((T, nG), np.nan)

    total_gt = 0
    total_tr = 0
    matches = 0
    FN = 0
    FP = 0
    sum_match_dist = 0.0

    contingency = np.zeros((nG, nT), dtype=np.int64)

    frames = sorted(set(gt["frame"]).intersection(res["frame"]))

    for f in frames:
        gtf = gt[gt["frame"] == f][["id", "x", "y", "z"]]
        rsf = res[res["frame"] == f][["id", "x", "y", "z"]]

        g_ids = gtf["id"].to_numpy()
        r_ids = rsf["id"].to_numpy()
        G = gtf[["x", "y", "z"]].to_numpy(dtype=float)
        R = rsf[["x", "y", "z"]].to_numpy(dtype=float)

        n_g = G.shape[0]
        n_r = R.shape[0]
        total_gt += n_g
        total_tr += n_r

        if n_g == 0 and n_r == 0:
            continue
        if n_g == 0:
            FP += n_r
            continue
        if n_r == 0:
            FN += n_g
            continue

        C = cdist(G, R, metric="euclidean")  # (n_gt, n_tr)
        gi, rj = linear_sum_assignment(C)

        keep = C[gi, rj] <= gate
        gi = gi[keep]
        rj = rj[keep]

        m = gi.size
        matches += m
        if m > 0:
            dists = C[gi, rj]
            sum_match_dist += float(dists.sum())

        FN += (n_g - m)
        FP += (n_r - m)

        for g_row, r_col in zip(gi, rj):
            gid = int(g_ids[g_row])
            tid = int(r_ids[r_col])
            g_global = gt_id_to_idx[gid]
            gt_to_track[f, g_global] = tid
            contingency[g_global, tr_id_to_idx[tid]] += 1

    recall = matches / total_gt if total_gt else 0.0
    precision = matches / total_tr if total_tr else 0.0

    total_id_switches = 0
    total_frags = 0
    for j in range(nG):
        s = gt_to_track[:, j]
        total_id_switches += count_id_switches_ignore_gaps(s)
        total_frags += count_frags(s)

    avg_swaps_per_gt = total_id_switches / nG if nG else 0.0
    avg_frags_per_gt = total_frags / nG if nG else 0.0

    IDTP = 0
    if nG > 0 and nT > 0 and contingency.sum() > 0:
        cost = -contingency
        gi2, tj2 = linear_sum_assignment(cost)
        IDTP = int(contingency[gi2, tj2].sum())

    IDFN = total_gt - IDTP
    IDFP = total_tr - IDTP
    denom = (2 * IDTP + IDFP + IDFN)
    IDF1 = (2 * IDTP / denom) if denom else 0.0

    MOTA = 1.0 - ((FN + FP + total_id_switches) / total_gt) if total_gt else 0.0
    MOTP = (sum_match_dist / matches) if matches else float("nan")

    gt_counts = np.zeros(T, dtype=int)
    tr_counts = np.zeros(T, dtype=int)
    for f, c in gt["frame"].value_counts().items():
        fi = int(f)
        if 0 <= fi < T:
            gt_counts[fi] = int(c)
    for f, c in res["frame"].value_counts().items():
        fi = int(f)
        if 0 <= fi < T:
            tr_counts[fi] = int(c)

    mean_abs_count_error = float(np.mean(np.abs(tr_counts - gt_counts))) if T > 0 else 0.0

    perf = _compute_avg_fps(res)

    return {
        "T": int(T),
        "n_gt_ids": int(nG),
        "n_track_ids": int(nT),
        "counts": {
            "total_gt": int(total_gt),
            "total_tr": int(total_tr),
            "matches": int(matches),
            "FN": int(FN),
            "FP": int(FP),
        },
        "coverage": {
            "recall": float(recall),
            "precision": float(precision),
        },
        "identity": {
            "total_id_switches": int(total_id_switches),
            "avg_swaps_per_gt": float(avg_swaps_per_gt),
            "total_frags": int(total_frags),
            "avg_frags_per_gt": float(avg_frags_per_gt),
            "IDF1": float(IDF1),
            "IDTP": int(IDTP),
            "IDFP": int(IDFP),
            "IDFN": int(IDFN),
        },
        "composite": {
            "MOTA": float(MOTA),
            "MOTP": float(MOTP) if np.isfinite(MOTP) else None,
        },
        "track_management": {
            "mean_abs_count_error": float(mean_abs_count_error),
        },
        "performance": perf,  # <-- added
    }


def _print_report(folder: Path, results_name: str, gate: float, metrics: dict) -> None:
    T = metrics["T"]
    nG = metrics["n_gt_ids"]
    nT = metrics["n_track_ids"]
    c = metrics["counts"]
    cov = metrics["coverage"]
    ident = metrics["identity"]
    comp = metrics["composite"]
    tm = metrics["track_management"]
    perf = metrics.get("performance", {})

    print("=== Tracking Evaluation ===")
    print(f"folder: {folder}")
    print(f"results: {results_name}")
    print(f"gate: {gate}")
    print(f"T: {T}  |  n_gt_ids: {nG}  |  n_track_ids: {nT}")
    print()

    print("Counts:")
    print(f"  total_gt: {c['total_gt']}")
    print(f"  total_tr: {c['total_tr']}")
    print(f"  matches:  {c['matches']}")
    print(f"  FN:       {c['FN']}")
    print(f"  FP:       {c['FP']}")
    print()

    print("Coverage:")
    print(f"  recall:    {cov['recall']:.4f}")
    print(f"  precision: {cov['precision']:.4f}")
    print()

    print("Identity:")
    print(f"  total_id_switches: {ident['total_id_switches']}")
    print(f"  avg_swaps_per_gt:  {ident['avg_swaps_per_gt']:.4f}")
    print(f"  total_frags:       {ident['total_frags']}")
    print(f"  avg_frags_per_gt:  {ident['avg_frags_per_gt']:.4f}")
    print(f"  IDF1:              {ident['IDF1']:.4f}")
    print()

    print("Composite:")
    print(f"  MOTA: {comp['MOTA']:.3f}")
    print(f"  MOTP: {comp['MOTP']}")
    print()

    print("Track management:")
    print(f"  mean_abs_count_error: {tm['mean_abs_count_error']:.4f}")
    print()

    print("Performance:")
    if perf.get("avg_fps") is None:
        print("  avg_fps: None (missing/invalid time(s) column or duration)")
    else:
        print(f"  avg_fps: {perf['avg_fps']:.3f}")
        print(f"  duration_s: {perf['duration_s']:.3f}")
        print(f"  frames_processed: {perf['frames_processed']}")
    print()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)

    gt_path = folder / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")

    gt = pd.read_csv(gt_path)
    _require_columns(gt, "ground_truth.csv")

    candidates = sorted([p for p in folder.iterdir() if p.is_file() and p.name.startswith("results")])
    if not candidates:
        raise FileNotFoundError(f"No results* files found in {folder}")

    out_dir = Path(args.json_out) if args.json_out else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for res_path in candidates:
        if res_path.name == "ground_truth.csv":
            continue

        try:
            res = pd.read_csv(res_path)
        except Exception as e:
            print(f"[skip] Could not read {res_path.name}: {e}")
            continue

        try:
            metrics = evaluate_one(gt, res, gate=args.gate)
        except Exception as e:
            print(f"[error] Evaluation failed for {res_path.name}: {e}")
            continue

        _print_report(folder, res_path.name, args.gate, metrics)

        stem = res_path.stem if res_path.suffix else res_path.name
        json_path = out_dir / f"metrics_{stem}.json"
        payload = {
            "folder": str(folder),
            "results_file": res_path.name,
            "gate": float(args.gate),
            "metrics": metrics,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[saved] {json_path}")
        print("-" * 60)


if __name__ == "__main__":
    main()