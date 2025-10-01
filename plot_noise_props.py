# -*- coding: utf-8 -*-
"""
Plot how proportions change with noise std for a fixed (rho, size).

- Reads JSONs from multiple noise result directories (e.g., data/results/noise1 ... noise7)
  File name pattern: rho{...}_size{...}_run{...}_std{...}.json
  where dots are replaced by 'p', e.g. rho1p0_size0p84_run3_std0p2.json

- Aggregates across runs for each std (same rho,size), computing mean ± error bars for:
    * prop_active
    * prop_grid_like_in_active

- Robustness (simple heuristic outlier handling):
    * "Majority-high → drop zeros" rule:
        If at a given (rho, size, std) at least q_major of runs are > tau_high,
        then values <= eps_zero are treated as structural zeros and removed
        BEFORE computing the mean and error bars. This is applied per-metric via switches.

- Saves figure to: data/results/plots/prop_vs_noise_rho{rho}_size{size}.png
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.transparent"] = True


# --------------------------- Config ---------------------------

# Multiple directories; add/remove as needed
NOISE_RESULTS_DIRS: List[str] = [
    os.path.join("data", "results", "noise6"),
    os.path.join("data", "results", "noise7"),
    os.path.join("data", "results", "noise8"),
    os.path.join("data", "results", "noise9"),
    os.path.join("data", "results", "noise10"),
    os.path.join("data", "results", "noise11"),
]

PLOTS_DIR = os.path.join("data", "results", "plots")

# Fixed (rho, size)
RHO_FIXED = 0.8
SIZE_FIXED = 0.5

# Filename pattern, e.g. "rho1p0_size0p84_run3_std0p2.json"
FILENAME_RE = re.compile(
    r"^rho([0-9p]+)_size([0-9p]+)_run(\d+)_std([0-9p]+)\.json$"
)

# Plot dpi
plt.rcParams["figure.dpi"] = 140

# ----- Heuristic filtering: "majority-high → drop zeros" -----
# Apply the rule to each metric? (PGI = prop_grid_like_in_active, PA = prop_active)
APPLY_FILTER_TO_PGI = True
APPLY_FILTER_TO_PA  = False  # Often you may not want to filter PA; set True if you do.

# Rule parameters
TAU_HIGH  = 0.7   # threshold for "high"
Q_MAJOR   = 0.1   # proportion that must exceed TAU_HIGH to trigger zero-dropping
EPS_ZERO  = 0.5  # values <= EPS_ZERO are treated as structural zeros
N_MIN_GRP = 5     # minimum group size to even consider filtering

# Error bar choice: use standard deviation (SD) or standard error of the mean (SEM)
ERRORBAR_SEM = False  # if True, show SEM; else show SD


# ------------------------ Utilities --------------------------

def pstr_to_float(s: str) -> float:
    """Convert a string like '0p995' to float 0.995."""
    return float(s.replace("p", "."))

def mean_std_safe(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std); if empty list, return (np.nan, np.nan)."""
    if len(values) == 0:
        return (np.nan, np.nan)
    arr = np.asarray(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr, ddof=0)))

def sem_from_std(std: float, n: int) -> float:
    """Return SEM given population-style std and sample size n."""
    if n <= 0 or not np.isfinite(std):
        return np.nan
    return float(std) / np.sqrt(float(n))

def filter_structural_zeros(
    vals: List[float],
    tau_high: float = TAU_HIGH,
    q_major: float = Q_MAJOR,
    eps: float = EPS_ZERO,
    n_min: int = N_MIN_GRP
):
    """
    If at least q_major of values are > tau_high AND group size >= n_min,
    drop all values <= eps (treat them as structural zeros).

    Returns:
        kept_values (np.ndarray),
        n_removed (int),
        n_total (int),
        rule_triggered (bool)
    """
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n_total = v.size
    if n_total < n_min:
        return v, 0, n_total, False

    prop_high = np.mean(v > tau_high) if n_total > 0 else 0.0
    if prop_high >= q_major:
        kept = v[v > eps]
        return kept, (n_total - kept.size), n_total, True
    else:
        return v, 0, n_total, False


# -------------------- Load & Collect Data ---------------------

def collect_noise_results(results_dirs: Iterable[str]) -> Dict[Tuple[float, float, float], dict]:
    """
    Read all JSONs from multiple directories and collect props per (rho, size, std).
    Returns: dict keyed by (rho, size, std) with lists of metrics across runs.

    Note:
    - Same filenames in different directories are fine; we read them separately.
    - Aggregation key ignores run_id by design (we aggregate across runs).
    """
    data = defaultdict(lambda: {
        "prop_active": [],
        "prop_grid_like_in_active": [],
    })

    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            print(f"[WARN] Results directory not found: {results_dir}")
            continue

        for fname in os.listdir(results_dir):
            if not fname.endswith(".json"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                # skip non-matching files
                continue

            rho_val = pstr_to_float(m.group(1))
            size_val = pstr_to_float(m.group(2))
            # run_id = int(m.group(3))  # not used in aggregation key
            std_val = pstr_to_float(m.group(4))

            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r") as f:
                    result = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
                continue

            key = (rho_val, size_val, std_val)

            pa = result.get("prop_active", None)
            pgi = result.get("prop_grid_like_in_active", None)

            if pa is not None and np.isfinite(pa):
                data[key]["prop_active"].append(float(pa))
            if pgi is not None and np.isfinite(pgi):
                data[key]["prop_grid_like_in_active"].append(float(pgi))

    return data


# --------------------- Aggregate with Filtering ---------------

def aggregate_by_std(
    data_by_key: Dict[Tuple[float, float, float], dict],
    rho_fixed: float,
    size_fixed: float
):
    """
    For a fixed (rho,size), aggregate across runs for each std.

    Returns:
        stds_sorted,
        (mean, err)_active,
        (mean, err)_gridlike,
        stats dict with removal counts for transparency
    """
    # filter keys for the fixed (rho,size)
    entries = [(std, metrics)
               for (rho, size, std), metrics in data_by_key.items()
               if abs(rho - rho_fixed) < 1e-12 and abs(size - size_fixed) < 1e-12]

    if not entries:
        return (
            np.array([]),
            (np.array([]), np.array([])),
            (np.array([]), np.array([])),
            {"pgi": {}, "pa": {}}
        )

    # sort by std
    entries.sort(key=lambda x: x[0])
    stds = np.array([e[0] for e in entries], dtype=float)

    pa_means, pa_errs = [], []
    pgi_means, pgi_errs = [], []

    # removal bookkeeping (per-std)
    pgi_removed, pgi_total, pgi_trigger = [], [], []
    pa_removed, pa_total, pa_trigger = [], [], []

    for std_val, metrics in entries:
        # --- PGI ---
        raw_pgi = metrics["prop_grid_like_in_active"]
        if APPLY_FILTER_TO_PGI:
            kept_pgi, removed, total, trig = filter_structural_zeros(raw_pgi)
        else:
            kept_pgi = np.asarray(raw_pgi, dtype=float)
            removed, total, trig = 0, len(kept_pgi), False

        m_pgi, s_pgi = mean_std_safe(list(kept_pgi))
        err_pgi = sem_from_std(s_pgi, len(kept_pgi)) if ERRORBAR_SEM else s_pgi

        pgi_means.append(m_pgi)
        pgi_errs.append(err_pgi)
        pgi_removed.append(removed)
        pgi_total.append(total)
        pgi_trigger.append(trig)

        # --- PA ---
        raw_pa = metrics["prop_active"]
        if APPLY_FILTER_TO_PA:
            kept_pa, removed_pa, total_pa, trig_pa = filter_structural_zeros(raw_pa)
        else:
            kept_pa = np.asarray(raw_pa, dtype=float)
            removed_pa, total_pa, trig_pa = 0, len(kept_pa), False

        m_pa, s_pa = mean_std_safe(list(kept_pa))
        err_pa = sem_from_std(s_pa, len(kept_pa)) if ERRORBAR_SEM else s_pa

        pa_means.append(m_pa)
        pa_errs.append(err_pa)
        pa_removed.append(removed_pa)
        pa_total.append(total_pa)
        pa_trigger.append(trig_pa)

        # Console transparency
        print(f"[GROUP std={std_val:.6g}] "
              f"PGI: kept={len(kept_pgi):3d} removed={removed:3d}/{total:3d} "
              f"(triggered={trig}) -> mean={m_pgi:.4f} ± {err_pgi:.4f} "
              f"|| PA: kept={len(kept_pa):3d} removed={removed_pa:3d}/{total_pa:3d} "
              f"(triggered={trig_pa}) -> mean={m_pa:.4f} ± {err_pa:.4f}")

    stats = {
        "pgi": {
            "removed": np.array(pgi_removed, dtype=int),
            "total":   np.array(pgi_total, dtype=int),
            "trigger": np.array(pgi_trigger, dtype=bool)
        },
        "pa": {
            "removed": np.array(pa_removed, dtype=int),
            "total":   np.array(pa_total, dtype=int),
            "trigger": np.array(pa_trigger, dtype=bool)
        }
    }

    return (
        stds,
        (np.array(pa_means),  np.array(pa_errs)),
        (np.array(pgi_means), np.array(pgi_errs)),
        stats
    )


# --------------------------- Plotting -------------------------

def plot_props_vs_noise(
    noise_results_dirs: Iterable[str],
    rho_fixed: float,
    size_fixed: float,
    out_dir: str
):
    """
    Make a clean errorbar plot:
      - Two series only: mean ± SD(or SEM) for prop_active and prop_grid_like_in_active
      - Optional heuristic filtering applied BEFORE aggregation
      - Transparent logging printed to console per-std
      - A concise method note in the figure corner (moved to top-left to avoid overlap)
    """
    data_by_key = collect_noise_results(noise_results_dirs)
    stds, (pa_mean, pa_err), (pgi_mean, pgi_err), stats = aggregate_by_std(
        data_by_key, rho_fixed=rho_fixed, size_fixed=size_fixed
    )

    if stds.size == 0:
        print(f"[WARN] No data found for rho={rho_fixed}, size={size_fixed} in {list(noise_results_dirs)}")
        return

    plt.figure()
    plt.errorbar(stds, pa_mean,  yerr=pa_err,  fmt='-o', capsize=4, label="prop_active", zorder=2)
    plt.errorbar(stds, pgi_mean, yerr=pgi_err, fmt='-s', capsize=4, label="prop_grid_like_in_active", zorder=2)

    plt.xlabel("noise std")
    plt.ylabel("Proportion")
    plt.title(f"Proportions vs noise (rho={rho_fixed}, size={size_fixed})")
    plt.grid(True, zorder=1)
    plt.legend(loc="best")

    err_label = "SEM" if ERRORBAR_SEM else "SD"
    pgi_removed_total = int(np.nansum(stats["pgi"]["removed"])) if "pgi" in stats else 0
    pgi_total_total   = int(np.nansum(stats["pgi"]["total"]))   if "pgi" in stats else 0
    pa_removed_total  = int(np.nansum(stats["pa"]["removed"]))  if "pa" in stats else 0
    pa_total_total    = int(np.nansum(stats["pa"]["total"]))    if "pa" in stats else 0

    method_note = (
        f"Error bars: {err_label}. "
        f"Filter (PGI={'on' if APPLY_FILTER_TO_PGI else 'off'}, "
        f"PA={'on' if APPLY_FILTER_TO_PA else 'off'}): "
        f"If ≥{int(Q_MAJOR * 100)}% runs > {TAU_HIGH}, drop values ≤ {EPS_ZERO}."
    )
    method_note2 = (
        f"Removed PGI {pgi_removed_total}/{pgi_total_total}"
        + (f"; PA {pa_removed_total}/{pa_total_total}" if APPLY_FILTER_TO_PA else "")
    )

    ax = plt.gca()
    ax.text(
        0.01, 0.98,
        method_note + "\n" + method_note2,
        transform=ax.transAxes,
        fontsize=8,
        ha='left',
        va='top',
        bbox=dict(boxstyle="round", facecolor='white', alpha=0.75, linewidth=0),
        zorder=3,
    )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(
        out_dir,
        f"prop_vs_noise_rho{str(rho_fixed).replace('.', 'p')}_size{str(size_fixed).replace('.', 'p')}"
    )

    # save PNG and SVG
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {base}.png")
    print(f"[SAVE] {base}.svg")



# ----------------------------- Main ---------------------------

def main():
    plot_props_vs_noise(
        noise_results_dirs=NOISE_RESULTS_DIRS,
        rho_fixed=RHO_FIXED,
        size_fixed=SIZE_FIXED,
        out_dir=PLOTS_DIR
    )

if __name__ == "__main__":
    main()
