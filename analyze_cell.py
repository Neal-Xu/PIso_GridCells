# -*- coding: utf-8 -*-
"""
Aggregate JSON results and plot:
1) mean_spacing_grid_like vs rho  (size fixed)  -> fit hyperbolic (1/x), 1/x^2, linear  [PNG]
2) mean_spacing_grid_like vs rho_true_mean (size fixed; PGI-filtered x) -> same fits    [PNG]
3) mean_spacing_grid_like vs size (rho fixed)  -> fit cubic and linear                   [PNG]
4) rho-corrected spacing vs size (rho fixed)   -> same fits (cubic, linear)              [PNG]
5) prop_active & prop_grid_like_in_active vs size (rho fixed)                            [PNG]

SVGs:
- combo_spacing_vs_rho_size{size_fixed}.svg     (spacing vs rho | spacing vs rhoTRUE)  [no grid]
- combo_spacing_vs_size_rho{rho_fixed}.svg      (spacing vs size | rho-corrected vs size) [no grid]
- combo_activity_vs_noise_rho{…}_size{…}.svg    (props vs noise | activity vs size)     [no grid]
- NEW: combo_rhoHandled_size{…}_rho{…}.svg      (spacing vs rhoTRUE | rho-corrected vs size)
       Left = ONLY hyperbolic (1/x); Right = ONLY cubic; legends show AdjR² only; no grid.

Notes:
- For spacing and true rho, PGI thresholding is applied where described.
"""

import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable

# --------------------------- link noise module ---------------------------
# for the left/right panel of "props vs noise || activity vs size" combined SVG
import plot_noise_props as noise_mod

# --------------------------- Config ---------------------------

# Where to read result JSONs
RESULTS_DIRS: List[str] = [
    os.path.join("data", "results", "batch7"),
    os.path.join("data", "results", "batch8"),
    os.path.join("data", "results", "batch9"),
    os.path.join("data", "results", "batch10"),
    # os.path.join("data", "results", "batch11"),
    # os.path.join("data", "results", "batch12"),
    # os.path.join("data", "results", "batch13"),
]

# Where to save plots
PLOTS_DIR = os.path.join("data", "results", "plots")

# Fixed params for figures
SIZE_FIXED_DEFAULT = 0.84
RHO_FIXED_DEFAULT  = 0.8  # also used as reference in rho-correction

# PGI threshold for filtered aggregations
PGI_THRESHOLD = 0.6

# File name pattern: "rho0p8_size0p995_run2.json" or "rho0p8_size0p995.json"
FILENAME_RE = re.compile(r"^rho([0-9p]+)_size([0-9p]+)(?:_run\d+)?\.json$")

# Matplotlib defaults (optional)
plt.rcParams["figure.dpi"] = 140

# Heuristic filtering settings for activity_vs_size
APPLY_FILTER_TO_PGI_ACTIVITY: bool = False
APPLY_FILTER_TO_PA_ACTIVITY:  bool = False
TAU_HIGH: float  = 0.5
Q_MAJOR: float   = 0.5
EPS_ZERO: float  = 0.1
N_MIN_GRP: int   = 5
ERRORBAR_SEM: bool = False  # if True, show SEM; else show SD

# ------------------------ Utilities --------------------------

def pstr_to_float(s: str) -> float:
    """Convert a string like '0p995' to float 0.995."""
    return float(s.replace("p", "."))

def mean_std_safe(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std); if empty list, return (np.nan, np.nan)."""
    if len(values) == 0:
        return (np.nan, np.nan)
    arr = np.asarray(values, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr)))

def sem_from_std(std: float, n: int) -> float:
    """Return SEM given population-style std and sample size n."""
    if n <= 0 or not np.isfinite(std):
        return np.nan
    return float(std) / np.sqrt(float(n))

def nearest_value(target: float, candidates: List[float]) -> float:
    """Pick the candidate value closest to target. If candidates empty, return NaN."""
    if not candidates:
        return float("nan")
    return min(candidates, key=lambda x: abs(x - target))

def design_matrix(x: np.ndarray, kind: str) -> np.ndarray:
    """
    Build design matrix for OLS.
    kind in {"linear", "hyperbolic", "cubic", "inverse_square"}:
      - linear:         y = b0 + b1 * x
      - hyperbolic:     y = b0 + b1 * (1/x)
      - cubic:          y = b0 + b1 * x + b2 * x^2 + b3 * x^3
      - inverse_square: y = b0 + b1 * (1/x^2)
    """
    x = np.asarray(x, dtype=float)

    if kind == "linear":
        return np.column_stack([np.ones_like(x), x])

    elif kind == "hyperbolic":
        return np.column_stack([np.ones_like(x), 1.0 / x])

    elif kind == "cubic":
        return np.column_stack([np.ones_like(x), x, x**2, x**3])

    elif kind == "inverse_square":
        return np.column_stack([np.ones_like(x), 1.0 / (x**2)])

    else:
        raise ValueError(f"Unknown model kind: {kind}")

def fit_ols(x: np.ndarray, y: np.ndarray, kind: str):
    """
    Fit OLS for a given model kind and return dict with:
      params, yhat, rss, aic, bic, r2_adj, n, k, mask
    """
    X = design_matrix(x, kind)
    # Drop rows with NaN in X or y
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X = X[mask]
    y = y[mask]
    n = len(y)
    if n == 0:
        return None
    # OLS
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(np.sum((y - yhat) ** 2))
    k = X.shape[1]  # number of parameters

    if n > k:
        sigma2 = rss / n
        aic = n * np.log(sigma2) + 2 * k
        bic = n * np.log(sigma2) + k * np.log(n)
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (rss / tss) if tss > 0 else np.nan
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if not np.isnan(r2) and n > k else np.nan
    else:
        aic = bic = r2_adj = np.nan

    return {
        "params": beta,
        "yhat": yhat,
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "r2_adj": r2_adj,
        "n": n,
        "k": k,
        "mask": mask,
    }

def eval_model_on_grid(x_grid: np.ndarray, params: np.ndarray, kind: str) -> np.ndarray:
    """Evaluate fitted model on a dense x_grid for plotting."""
    Xg = design_matrix(x_grid, kind)
    return Xg @ params

# -------------------- Load & Aggregate Data -------------------

def load_results_from_dir(results_dir: str) -> Dict[Tuple[float, float], dict]:
    """
    Read all JSON files in a single directory and collect metrics into lists per (rho, size) group.
    Returns a dict mapping (rho, size) -> metric lists.

    We build a PGI-filtered list for spacing and true rho.
    """
    data_dict = defaultdict(lambda: {
        "mean_spacing_grid_like": [],
        "mean_spacing_grid_like_filtered": [],
        "prop_active": [],
        "prop_grid_like_in_active": [],
        # true rho lists
        "rho_true_mean": [],
        "rho_true_mean_filtered": [],
    })

    if not os.path.isdir(results_dir):
        print(f"[WARN] Results directory not found: {results_dir}")
        return data_dict

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        m = FILENAME_RE.match(fname)
        if not m:
            continue

        rho_str, size_str = m.group(1), m.group(2)
        rho_val = pstr_to_float(rho_str)
        size_val = pstr_to_float(size_str)

        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r") as f:
                result = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
            continue

        key = (rho_val, size_val)

        # proportions
        pa  = float(result.get("prop_active", np.nan))
        pgi = float(result.get("prop_grid_like_in_active", np.nan))
        data_dict[key]["prop_active"].append(pa)
        data_dict[key]["prop_grid_like_in_active"].append(pgi)

        # spacing (raw)
        sp = float(result.get("mean_spacing_grid_like", np.nan))
        data_dict[key]["mean_spacing_grid_like"].append(sp)

        # spacing (filtered by PGI threshold)
        if (not math.isnan(pgi)) and pgi > PGI_THRESHOLD and (not math.isnan(sp)):
            data_dict[key]["mean_spacing_grid_like_filtered"].append(sp)

        # true rho (unfiltered)
        rtm = result.get("rho_true_mean", None)
        if rtm is not None and np.isfinite(rtm):
            data_dict[key]["rho_true_mean"].append(float(rtm))

        # true rho (filtered by PGI threshold)
        if (rtm is not None) and np.isfinite(rtm) and (not math.isnan(pgi)) and (pgi > PGI_THRESHOLD):
            data_dict[key]["rho_true_mean_filtered"].append(float(rtm))

    return data_dict

def merge_data_dicts(dicts: Iterable[Dict[Tuple[float, float], dict]]) -> Dict[Tuple[float, float], dict]:
    """Merge multiple data_dicts by concatenating lists for the same (rho, size) keys."""
    merged = defaultdict(lambda: {
        "mean_spacing_grid_like": [],
        "mean_spacing_grid_like_filtered": [],
        "prop_active": [],
        "prop_grid_like_in_active": [],
        "rho_true_mean": [],
        "rho_true_mean_filtered": [],
    })
    for d in dicts:
        for key, metrics in d.items():
            for k in merged[key].keys():
                merged[key][k].extend(metrics.get(k, []))
    return merged

def aggregate_data(data_dict: Dict[Tuple[float, float], dict]) -> Dict[Tuple[float, float], dict]:
    """
    Compute mean/std for each metric within each (rho, size) group.
    For spacing and true rho, use the FILTERED lists (PGI>threshold).
    Proportions keep their original lists.
    """
    agg = {}
    for key, metrics in data_dict.items():
        agg[key] = {
            # spacing uses filtered list
            "mean_spacing_grid_like": mean_std_safe(metrics["mean_spacing_grid_like_filtered"]),
            # proportions keep original lists
            "prop_active": mean_std_safe(metrics["prop_active"]),
            "prop_grid_like_in_active": mean_std_safe(metrics["prop_grid_like_in_active"]),
            # true rho (both, for transparency)
            "rho_true_mean": mean_std_safe(metrics["rho_true_mean"]),
            "rho_true_mean_filtered": mean_std_safe(metrics["rho_true_mean_filtered"]),
        }
    return agg

# -------------------- Filtering helper (for activity) -------------------

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

# --------------------------- Plotting (PNG originals) -------------------------

def plot_spacing_vs_rho_with_fits(agg_data: Dict[Tuple[float, float], dict],
                                  size_fixed: float,
                                  out_dir: str):
    """PNG: spacing vs nominal rho (size fixed): fits = hyperbolic, inverse_square, linear."""
    rhos = sorted({rho for (rho, size) in agg_data.keys() if abs(size - size_fixed) < 1e-12})
    means = [agg_data[(rho, size_fixed)]["mean_spacing_grid_like"][0] for rho in rhos]
    stds  = [agg_data[(rho, size_fixed)]["mean_spacing_grid_like"][1] for rho in rhos]

    rhos_arr  = np.asarray(rhos, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr  = np.asarray(stds, dtype=float)
    mask = ~np.isnan(means_arr)

    if mask.sum() < 3:
        print(f"[WARN] Not enough valid points to fit for size={size_fixed}. Need >=3.")
        return

    x = rhos_arr[mask]
    y = means_arr[mask]
    yerr = stds_arr[mask]

    fit_hyp   = fit_ols(x, y, kind="hyperbolic")
    fit_inv2  = fit_ols(x, y, kind="inverse_square")
    fit_lin   = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.min(x), np.max(x), 500)
    x_grid = x_grid[x_grid != 0]

    model_entries = []
    if fit_hyp is not None:   model_entries.append(("1/x", "hyperbolic", fit_hyp))
    if fit_inv2 is not None:  model_entries.append(("1/x²", "inverse_square", fit_inv2))
    if fit_lin is not None:   model_entries.append(("linear", "linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")

    linestyle_map = {"linear": "--", "hyperbolic": "-", "inverse_square": "-."}
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        plt.plot(x_grid, yg, linestyle=linestyle_map[kind],
                 label=f"{name:8s}  AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")

    plt.xlabel("rho")
    plt.ylabel("mean_spacing_grid_like")
    plt.title(f"mean_spacing_grid_like vs rho (size={size_fixed})")
    plt.grid(True)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"spacing_vs_rho_size{size_fixed}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")

def plot_spacing_vs_rhoTRUE_with_fits(agg_data: Dict[Tuple[float, float], dict],
                                      size_fixed: float,
                                      out_dir: str):
    """PNG: spacing vs rho_true_mean (filtered x), fits = hyperbolic, inverse_square, linear."""
    keys = [(rho, size) for (rho, size) in agg_data.keys() if abs(size - size_fixed) < 1e-12]
    if not keys:
        print(f"[WARN] No groups found for size={size_fixed}")
        return

    xs, ys, yerrs = [], [], []
    for (rho, size) in keys:
        rtm_mean, rtm_std = agg_data[(rho, size)]["rho_true_mean_filtered"]
        sp_mean,  sp_std  = agg_data[(rho, size)]["mean_spacing_grid_like"]
        if np.isnan(rtm_mean) or np.isnan(sp_mean):
            continue
        xs.append(float(rtm_mean))
        ys.append(float(sp_mean))
        yerrs.append(float(sp_std))

    if len(xs) < 3:
        print(f"[WARN] Not enough valid points to fit for size={size_fixed} (need >=3 with filtered rho_true_mean present).")
        return

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yerrs = np.asarray(yerrs, dtype=float)
    idx = np.argsort(xs)
    x = xs[idx]; y = ys[idx]; yerr = yerrs[idx]

    fit_hyp   = fit_ols(x, y, kind="hyperbolic")
    fit_inv2  = fit_ols(x, y, kind="inverse_square")
    fit_lin   = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 500)
    x_grid = x_grid[x_grid != 0]

    model_entries = []
    if fit_hyp   is not None: model_entries.append(("1/x",     "hyperbolic",     fit_hyp))
    if fit_inv2  is not None: model_entries.append(("1/x²",    "inverse_square", fit_inv2))
    if fit_lin   is not None: model_entries.append(("linear",  "linear",         fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")

    linestyle_map = {"linear": "--", "hyperbolic": "-", "inverse_square": "-."}
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        plt.plot(x_grid, yg, linestyle=linestyle_map[kind],
                 label=f"{name:8s}  AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")

    plt.xlabel("rho_true_mean")
    plt.ylabel("mean_spacing_grid_like")
    plt.title(f"mean_spacing_grid_like vs rho_true_mean (size={size_fixed})")
    plt.grid(True)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"spacing_vs_rhoTRUE_size{size_fixed}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")

def _rho_correct_spacing(y_mean: float, y_std: float, r_mean: float, r_std: float, rho_ref: float) -> Tuple[float, float]:
    """
    Apply rho normalization: y_adj = y_mean * (r_mean / rho_ref)
    Propagate std with first-order approximation assuming independence:
        var(y_adj) ≈ (r_mean/rho_ref)^2 * var(y) + (y_mean/rho_ref)^2 * var(r)
    """
    if any(map(lambda z: not np.isfinite(z), [y_mean, y_std, r_mean, r_std, rho_ref])) or rho_ref == 0:
        return (np.nan, np.nan)
    y_adj = y_mean * (r_mean / rho_ref)
    var_adj = (r_mean / rho_ref) ** 2 * (y_std ** 2) + (y_mean / rho_ref) ** 2 * (r_std ** 2)
    std_adj = float(np.sqrt(max(var_adj, 0.0)))
    return (float(y_adj), std_adj)

def plot_spacing_vs_size_with_fits(agg_data: Dict[Tuple[float, float], dict],
                                   rho_fixed: float,
                                   out_dir: str):
    """PNG: spacing vs size (rho fixed): fits = cubic, linear."""
    sizes = sorted({size for (rho, size) in agg_data.keys() if abs(rho - rho_fixed) < 1e-12})
    means = [agg_data[(rho_fixed, size)]["mean_spacing_grid_like"][0] for size in sizes]
    stds  = [agg_data[(rho_fixed, size)]["mean_spacing_grid_like"][1] for size in sizes]

    sizes_arr = np.asarray(sizes, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr  = np.asarray(stds, dtype=float)
    mask = ~np.isnan(means_arr)

    if mask.sum() < 3:
        print(f"[WARN] Not enough valid points to fit for rho={rho_fixed}. Need >=3.")
        return

    x = sizes_arr[mask]
    y = means_arr[mask]
    yerr = stds_arr[mask]

    fit_cubic = fit_ols(x, y, kind="cubic")
    fit_lin   = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.min(x), np.max(x), 300)

    model_entries = []
    if fit_cubic is not None: model_entries.append(("cubic",  "cubic",  fit_cubic))
    if fit_lin   is not None: model_entries.append(("linear", "linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")

    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        linestyle = {"linear": "--", "cubic": "-"}[kind]
        plt.plot(x_grid, yg, linestyle=linestyle,
                 label=f"{name:6s} AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")

    plt.xlabel("size")
    plt.ylabel("mean_spacing_grid_like")
    plt.title(f"mean_spacing_grid_like vs size (rho={rho_fixed})")
    plt.grid(True)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"spacing_vs_size_rho{rho_fixed}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")

def plot_spacing_vs_size_rho_corrected(agg_data: Dict[Tuple[float, float], dict],
                                       rho_fixed: float,
                                       out_dir: str):
    """PNG: rho-corrected spacing vs size (rho fixed): fits = cubic, linear."""
    rho_ref = float(rho_fixed)

    sizes = sorted({size for (rho, size) in agg_data.keys() if abs(rho - rho_fixed) < 1e-12})
    if not sizes:
        print(f"[WARN] No size groups found for nominal rho={rho_fixed}")
        return

    y_adj_means = []
    y_adj_stds  = []
    for size in sizes:
        key = (rho_fixed, size)
        sp_mean, sp_std = agg_data[key]["mean_spacing_grid_like"]
        r_mean,  r_std  = agg_data[key]["rho_true_mean_filtered"]
        y_adj, s_adj = _rho_correct_spacing(sp_mean, sp_std, r_mean, r_std, rho_ref)
        y_adj_means.append(y_adj)
        y_adj_stds.append(s_adj)

    sizes_arr = np.asarray(sizes, dtype=float)
    y_arr     = np.asarray(y_adj_means, dtype=float)
    yerr_arr  = np.asarray(y_adj_stds, dtype=float)
    mask = ~np.isnan(y_arr)

    if mask.sum() < 3:
        print(f"[WARN] Not enough valid points to fit rho-corrected spacing for rho={rho_fixed}. Need >=3.")
        return

    x = sizes_arr[mask]
    y = y_arr[mask]
    yerr = yerr_arr[mask]

    fit_cubic = fit_ols(x, y, kind="cubic")
    fit_lin   = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.min(x), np.max(x), 300)

    model_entries = []
    if fit_cubic is not None: model_entries.append(("cubic",  "cubic",  fit_cubic))
    if fit_lin   is not None: model_entries.append(("linear", "linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="rho-corrected data (mean±std)")

    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        linestyle = {"linear": "--", "cubic": "-"}[kind]
        plt.plot(x_grid, yg, linestyle=linestyle,
                 label=f"{name:6s} AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")

    plt.xlabel("size")
    plt.ylabel(f"mean_spacing_grid_like (rho-corrected to rho_ref={rho_ref})")
    plt.title(f"rho-corrected spacing vs size (nominal rho={rho_fixed})")
    plt.grid(True)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"spacing_vs_size_rho{rho_fixed}_rhoAdj.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")

def plot_activity_vs_size(merged_raw: Dict[Tuple[float, float], dict],
                          rho_fixed: float,
                          out_dir: str):
    """PNG: prop_active and prop_grid_like_in_active vs size (rho fixed)."""
    sizes = sorted({size for (rho, size) in merged_raw.keys() if abs(rho - rho_fixed) < 1e-12})
    if not sizes:
        print(f"[WARN] No size groups found for rho={rho_fixed}")
        return

    means_active, errs_active = [], []
    means_grid,  errs_grid  = [], []

    total_removed_pgi = 0
    total_runs_pgi    = 0
    total_removed_pa  = 0
    total_runs_pa     = 0

    for size in sizes:
        key = (rho_fixed, size)
        pa_vals  = merged_raw[key]["prop_active"]
        pgi_vals = merged_raw[key]["prop_grid_like_in_active"]

        # PGI filtering (default ON)
        if APPLY_FILTER_TO_PGI_ACTIVITY:
            kept_pgi, removed_pgi, total_pgi, trig_pgi = filter_structural_zeros(
                pgi_vals, tau_high=TAU_HIGH, q_major=Q_MAJOR, eps=EPS_ZERO, n_min=N_MIN_GRP
            )
        else:
            kept_pgi = np.asarray([v for v in pgi_vals if np.isfinite(v)], dtype=float)
            removed_pgi, total_pgi, trig_pgi = 0, len(kept_pgi), False

        # PA filtering (default OFF)
        if APPLY_FILTER_TO_PA_ACTIVITY:
            kept_pa, removed_pa, total_pa, trig_pa = filter_structural_zeros(
                pa_vals, tau_high=TAU_HIGH, q_major=Q_MAJOR, eps=EPS_ZERO, n_min=N_MIN_GRP
            )
        else:
            kept_pa = np.asarray([v for v in pa_vals if np.isfinite(v)], dtype=float)
            removed_pa, total_pa, trig_pa = 0, len(kept_pa), False

        m_pa, s_pa = mean_std_safe(kept_pa.tolist())
        m_pgi, s_pgi = mean_std_safe(kept_pgi.tolist())

        e_pa  = sem_from_std(s_pa,  len(kept_pa))  if ERRORBAR_SEM else s_pa
        e_pgi = sem_from_std(s_pgi, len(kept_pgi)) if ERRORBAR_SEM else s_pgi

        means_active.append(m_pa)
        errs_active.append(e_pa)
        means_grid.append(m_pgi)
        errs_grid.append(e_pgi)

        total_removed_pgi += removed_pgi
        total_runs_pgi    += total_pgi
        total_removed_pa  += removed_pa
        total_runs_pa     += total_pa

        print(f"[GROUP size={size:.6g}] "
              f"PGI kept={len(kept_pgi):3d} removed={removed_pgi:3d}/{total_pgi:3d} "
              f"-> mean={m_pgi:.4f} ± {e_pgi:.4f} || "
              f"PA kept={len(kept_pa):3d} removed={removed_pa:3d}/{total_pa:3d} "
              f"-> mean={m_pa:.4f} ± {e_pa:.4f}")

    plt.figure()
    plt.errorbar(sizes, means_active, yerr=errs_active, fmt='-o', capsize=4, label="prop_active", zorder=2)
    plt.errorbar(sizes, means_grid,  yerr=errs_grid,  fmt='-s', capsize=4, label="prop_grid_like_in_active", zorder=2)
    plt.xlabel("size")
    plt.ylabel("Proportion")
    plt.title(f"Cell activity vs size (rho={rho_fixed})")
    plt.grid(True, zorder=1)
    plt.legend(loc="best")

    err_label = "SEM" if ERRORBAR_SEM else "SD"
    method_note = (
        f"Error bars: {err_label}. "
        f"Filter (PGI={'on' if APPLY_FILTER_TO_PGI_ACTIVITY else 'off'}, "
        f"PA={'on' if APPLY_FILTER_TO_PA_ACTIVITY else 'off'}): "
        f"If ≥{int(Q_MAJOR*100)}% runs > {TAU_HIGH}, drop values ≤ {EPS_ZERO}."
    )
    method_note2 = (
        f"Removed PGI {int(total_removed_pgi)}/{int(total_runs_pgi)}"
        + (f"; PA {int(total_removed_pa)}/{int(total_runs_pa)}" if APPLY_FILTER_TO_PA_ACTIVITY else "")
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
    save_path = os.path.join(out_dir, f"activity_vs_size_rho{rho_fixed}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_path}")

# --------------------------- Combined SVG (existing) ---------------------------
# (Same curves as before, but grids removed for SVGs)

def _plot_spacing_vs_rho_on_ax(ax, agg_data, size_fixed: float):
    """SVG helper: spacing vs nominal rho (size fixed). Same model set; NO grid."""
    rhos = sorted({rho for (rho, size) in agg_data.keys() if abs(size - size_fixed) < 1e-12})
    means = [agg_data[(rho, size_fixed)]["mean_spacing_grid_like"][0] for rho in rhos]
    stds  = [agg_data[(rho, size_fixed)]["mean_spacing_grid_like"][1] for rho in rhos]

    rhos_arr  = np.asarray(rhos, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr  = np.asarray(stds, dtype=float)
    mask = ~np.isnan(means_arr)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("rho")
        ax.set_title(f"mean_spacing_grid_like vs rho (size={size_fixed})")
        ax.grid(False)
        return

    x = rhos_arr[mask]; y = means_arr[mask]; yerr = stds_arr[mask]
    fit_hyp  = fit_ols(x, y, kind="hyperbolic")
    fit_inv2 = fit_ols(x, y, kind="inverse_square")
    fit_lin  = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.min(x), np.max(x), 500)
    x_grid = x_grid[x_grid != 0]

    model_entries = []
    if fit_hyp  is not None:  model_entries.append(("1/x", "hyperbolic", fit_hyp))
    if fit_inv2 is not None:  model_entries.append(("1/x²", "inverse_square", fit_inv2))
    if fit_lin  is not None:  model_entries.append(("linear", "linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")
    linestyle_map = {"linear": "--", "hyperbolic": "-", "inverse_square": "-."}
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        ax.plot(x_grid, yg, linestyle=linestyle_map[kind],
                label=f"{name:8s}  AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")
    ax.set_xlabel("rho")
    ax.set_title(f"mean_spacing_grid_like vs rho (size={size_fixed})")
    ax.grid(False)
    ax.legend()

def _plot_spacing_vs_rhoTRUE_on_ax(ax, agg_data, size_fixed: float):
    """SVG helper: spacing vs filtered true rho (size fixed). Same model set; NO grid."""
    keys = [(rho, size) for (rho, size) in agg_data.keys() if abs(size - size_fixed) < 1e-12]
    xs, ys, yerrs = [], [], []
    for (rho, size) in keys:
        rtm_mean, rtm_std = agg_data[(rho, size)]["rho_true_mean_filtered"]
        sp_mean,  sp_std  = agg_data[(rho, size)]["mean_spacing_grid_like"]
        if np.isnan(rtm_mean) or np.isnan(sp_mean):
            continue
        xs.append(float(rtm_mean)); ys.append(float(sp_mean)); yerrs.append(float(sp_std))

    if len(xs) < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("rho_true_mean (filtered by PGI)")
        ax.set_title(f"mean_spacing_grid_like vs rho_true_mean (filtered, size={size_fixed})")
        ax.grid(False)
        return

    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float); yerrs = np.asarray(yerrs, dtype=float)
    idx = np.argsort(xs); x = xs[idx]; y = ys[idx]; yerr = yerrs[idx]

    fit_hyp  = fit_ols(x, y, kind="hyperbolic")
    fit_inv2 = fit_ols(x, y, kind="inverse_square")
    fit_lin  = fit_ols(x, y, kind="linear")

    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 500)
    x_grid = x_grid[x_grid != 0]

    model_entries = []
    if fit_hyp  is not None: model_entries.append(("1/x", "hyperbolic", fit_hyp))
    if fit_inv2 is not None: model_entries.append(("1/x²","inverse_square", fit_inv2))
    if fit_lin  is not None: model_entries.append(("linear","linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])

    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")
    linestyle_map = {"linear": "--", "hyperbolic": "-", "inverse_square": "-."}
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        ax.plot(x_grid, yg, linestyle=linestyle_map[kind],
                label=f"{name:8s}  AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")
    ax.set_xlabel("rho_true_mean")
    ax.set_title(f"mean_spacing_grid_like vs rho_true_mean (size={size_fixed})")
    ax.grid(False)
    ax.legend()

def create_combined_svg_spacing_vs_rho(agg_data, size_fixed: float, out_dir: str):
    """Create combined SVG: [spacing vs rho] || [spacing vs rhoTRUE], shared y removed (independent); NO grids."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    _plot_spacing_vs_rho_on_ax(axs[0], agg_data, size_fixed)
    _plot_spacing_vs_rhoTRUE_on_ax(axs[1], agg_data, size_fixed)
    axs[0].set_ylabel("mean_spacing_grid_like")
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"combo_spacing_vs_rho_size{size_fixed}.svg")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")

def _plot_spacing_vs_size_on_ax(ax, agg_data, rho_fixed: float):
    """SVG helper: uncorrected spacing vs size (rho fixed). Same model set; NO grid."""
    sizes = sorted({size for (rho, size) in agg_data.keys() if abs(rho - rho_fixed) < 1e-12})
    means = [agg_data[(rho_fixed, size)]["mean_spacing_grid_like"][0] for size in sizes]
    stds  = [agg_data[(rho_fixed, size)]["mean_spacing_grid_like"][1] for size in sizes]
    sizes_arr = np.asarray(sizes, dtype=float); means_arr = np.asarray(means, dtype=float); stds_arr = np.asarray(stds, dtype=float)
    mask = ~np.isnan(means_arr)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size"); ax.set_title(f"mean_spacing_grid_like vs size (rho={rho_fixed})"); ax.grid(False)
        return
    x = sizes_arr[mask]; y = means_arr[mask]; yerr = stds_arr[mask]
    fit_cubic = fit_ols(x, y, kind="cubic"); fit_lin = fit_ols(x, y, kind="linear")
    x_grid = np.linspace(np.min(x), np.max(x), 300)
    model_entries = []
    if fit_cubic is not None: model_entries.append(("cubic", "cubic", fit_cubic))
    if fit_lin   is not None: model_entries.append(("linear","linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)")
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        linestyle = {"linear": "--", "cubic": "-"}[kind]
        ax.plot(x_grid, yg, linestyle=linestyle,
                label=f"{name:6s} AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")
    ax.set_xlabel("size"); ax.set_title(f"mean_spacing_grid_like vs size (rho={rho_fixed})"); ax.grid(False); ax.legend()

def _plot_spacing_vs_size_rhoAdj_on_ax(ax, agg_data, rho_fixed: float):
    """SVG helper: rho-corrected spacing vs size (rho fixed). Same model set; NO grid."""
    rho_ref = float(rho_fixed)
    sizes = sorted({size for (rho, size) in agg_data.keys() if abs(rho - rho_fixed) < 1e-12})
    if not sizes:
        ax.text(0.5, 0.5, "No size groups", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size"); ax.set_title(f"rho-corrected spacing vs size (nominal rho={rho_fixed})"); ax.grid(False)
        return
    y_adj_means, y_adj_stds = [], []
    for size in sizes:
        sp_mean, sp_std = agg_data[(rho_fixed, size)]["mean_spacing_grid_like"]
        r_mean,  r_std  = agg_data[(rho_fixed, size)]["rho_true_mean_filtered"]
        y_adj, s_adj = _rho_correct_spacing(sp_mean, sp_std, r_mean, r_std, rho_ref)
        y_adj_means.append(y_adj); y_adj_stds.append(s_adj)
    sizes_arr = np.asarray(sizes, dtype=float); y_arr = np.asarray(y_adj_means, dtype=float); yerr_arr = np.asarray(y_adj_stds, dtype=float)
    mask = ~np.isnan(y_arr)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size"); ax.set_title(f"rho-corrected spacing vs size (nominal rho={rho_fixed})"); ax.grid(False)
        return
    x = sizes_arr[mask]; y = y_arr[mask]; yerr = yerr_arr[mask]
    fit_cubic = fit_ols(x, y, kind="cubic"); fit_lin  = fit_ols(x, y, kind="linear")
    x_grid = np.linspace(np.min(x), np.max(x), 300)
    model_entries = []
    if fit_cubic is not None: model_entries.append(("cubic", "cubic", fit_cubic))
    if fit_lin   is not None: model_entries.append(("linear","linear", fit_lin))
    model_entries = [m for m in model_entries if not np.isnan(m[2]["aic"])]
    model_entries.sort(key=lambda m: m[2]["aic"])
    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="rho-corrected data (mean±std)")
    for name, kind, fit in model_entries:
        yg = eval_model_on_grid(x_grid, fit["params"], kind=kind)
        linestyle = {"linear": "--", "cubic": "-"}[kind]
        ax.plot(x_grid, yg, linestyle=linestyle,
                label=f"{name:6s} AIC={fit['aic']:.2f}, BIC={fit['bic']:.2f}, AdjR²={fit['r2_adj']:.3f}")
    ax.set_xlabel("size"); ax.set_title(f"rho-corrected spacing vs size (nominal rho={rho_fixed})")
    ax.grid(False); ax.legend()

def create_combined_svg_spacing_vs_size(agg_data, rho_fixed: float, out_dir: str):
    """Create combined SVG: [spacing vs size] || [rho-corrected spacing vs size], NO grids."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    _plot_spacing_vs_size_on_ax(axs[0], agg_data, rho_fixed)
    _plot_spacing_vs_size_rhoAdj_on_ax(axs[1], agg_data, rho_fixed)
    axs[0].set_ylabel("mean_spacing_grid_like")
    fig.tight_layout()
    save_path = os.path.join(out_dir, f"combo_spacing_vs_size_rho{rho_fixed}.svg")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")

# --------------------------- NEW: rho-handled combined (only 1/x and cubic) ---------------------------

def _plot_rhoTRUE_hyperbolic_only(ax, agg_data, size_fixed: float):
    """
    Left panel: mean_spacing_grid_like vs rho_true_mean (filtered), ONLY hyperbolic (1/x).
    No grid; legend shows AdjR² only (if available).
    """
    keys = [(rho, size) for (rho, size) in agg_data.keys() if abs(size - size_fixed) < 1e-12]
    xs, ys, yerrs = [], [], []
    for (rho, size) in keys:
        rtm_mean, rtm_std = agg_data[(rho, size)]["rho_true_mean_filtered"]
        sp_mean,  sp_std  = agg_data[(rho, size)]["mean_spacing_grid_like"]
        if np.isnan(rtm_mean) or np.isnan(sp_mean):
            continue
        xs.append(float(rtm_mean)); ys.append(float(sp_mean)); yerrs.append(float(sp_std))

    if len(xs) < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("rho_true_mean (filtered by PGI)")
        ax.set_ylabel("mean_spacing_grid_like")
        ax.set_title(f"mean_spacing_grid_like vs rho_true_mean (size={size_fixed})")
        ax.grid(False)
        return

    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float); yerrs = np.asarray(yerrs, dtype=float)
    idx = np.argsort(xs)
    x = xs[idx]; y = ys[idx]; yerr = yerrs[idx]

    fit_hyp = fit_ols(x, y, kind="hyperbolic")  # y = b0 + b1*(1/x)

    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="data (mean±std)", zorder=2)

    if fit_hyp is not None:
        x_grid = np.linspace(np.nanmin(x[x!=0]), np.nanmax(x), 500)
        x_grid = x_grid[x_grid != 0]
        yg = eval_model_on_grid(x_grid, fit_hyp["params"], kind="hyperbolic")
        if np.isfinite(fit_hyp.get("r2_adj", np.nan)):
            lbl = f"1/x   AdjR²={fit_hyp['r2_adj']:.3f}"
        else:
            lbl = "1/x"
        ax.plot(x_grid, yg, linestyle='-', label=lbl, zorder=3)

    ax.set_xlabel("rho_true_mean")
    ax.set_ylabel("mean_spacing_grid_like")
    ax.set_title(f"mean_spacing_grid_like vs rho_true_mean (size={size_fixed})")
    ax.grid(False)
    ax.legend(loc="best")

def _plot_size_cubic_only(ax, agg_data, rho_fixed: float):
    """
    Right panel: rho-corrected spacing vs size (rho fixed), ONLY cubic.
    No grid; legend shows AdjR² only (if available).
    """
    rho_ref = float(rho_fixed)
    sizes = sorted({size for (rho, size) in agg_data.keys() if abs(rho - rho_fixed) < 1e-12})
    if not sizes:
        ax.text(0.5, 0.5, "No size groups", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size")
        ax.set_ylabel(f"rho-corrected mean_spacing_grid_like (rho_ref={rho_ref})")
        ax.set_title(f"rho-corrected spacing vs size (rho={rho_fixed})")
        ax.grid(False)
        return

    y_adj_means, y_adj_stds = [], []
    for size in sizes:
        sp_mean, sp_std = agg_data[(rho_fixed, size)]["mean_spacing_grid_like"]
        r_mean,  r_std  = agg_data[(rho_fixed, size)]["rho_true_mean_filtered"]
        y_adj, s_adj = _rho_correct_spacing(sp_mean, sp_std, r_mean, r_std, rho_ref)
        y_adj_means.append(y_adj); y_adj_stds.append(s_adj)

    sizes_arr = np.asarray(sizes, dtype=float)
    y_arr     = np.asarray(y_adj_means, dtype=float)
    yerr_arr  = np.asarray(y_adj_stds, dtype=float)
    mask = ~np.isnan(y_arr)

    if mask.sum() < 3:
        ax.text(0.5, 0.5, "Insufficient points (>=3) for fit", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("size")
        ax.set_ylabel(f"rho-corrected mean_spacing_grid_like (rho_ref={rho_ref})")
        ax.set_title(f"rho-corrected spacing vs size (rho={rho_fixed})")
        ax.grid(False)
        return

    x = sizes_arr[mask]; y = y_arr[mask]; yerr = yerr_arr[mask]

    fit_cubic = fit_ols(x, y, kind="cubic")

    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, label="rho-corrected data (mean±std)", zorder=2)

    if fit_cubic is not None:
        x_grid = np.linspace(np.min(x), np.max(x), 300)
        yg = eval_model_on_grid(x_grid, fit_cubic["params"], kind="cubic")
        if np.isfinite(fit_cubic.get("r2_adj", np.nan)):
            lbl = f"cubic   AdjR²={fit_cubic['r2_adj']:.3f}"
        else:
            lbl = "cubic"
        ax.plot(x_grid, yg, linestyle='-', label=lbl, zorder=3)

    ax.set_xlabel("size")
    ax.set_ylabel(f"rho-corrected mean_spacing_grid_like (rho_ref={rho_ref})")
    ax.set_title(f"rho-corrected spacing vs size (rho={rho_fixed})")
    ax.grid(False)
    ax.legend(loc="best")

def create_combined_svg_rho_handled(agg_data, size_fixed: float, rho_fixed: float, out_dir: str):
    """
    NEW combined SVG (rho-handled):
      Left : spacing vs rho_true_mean (filtered), ONLY hyperbolic (1/x).
      Right: rho-corrected spacing vs size (rho fixed), ONLY cubic.
      NOTE: Different y-axes; NO grids.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    _plot_rhoTRUE_hyperbolic_only(axs[0], agg_data, size_fixed)
    _plot_size_cubic_only(axs[1], agg_data, rho_fixed)

    fig.tight_layout()
    save_path = os.path.join(out_dir, f"combo_rhoHandled_size{size_fixed}_rho{rho_fixed}.svg")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")

# --------------------------- Combined SVG: props vs noise || activity vs size ---------------------------

def _plot_activity_vs_size_on_ax(ax, merged_raw, rho_fixed: float):
    """SVG helper: activity vs size (clean), NO grid."""
    sizes = sorted({size for (rho, size) in merged_raw.keys() if abs(rho - rho_fixed) < 1e-12})
    if not sizes:
        ax.text(0.5, 0.5, "No size groups", ha="center", va="center", transform=ax.transAxes)
        return

    means_active, errs_active = [], []
    means_grid,  errs_grid  = [], []

    for size in sizes:
        key = (rho_fixed, size)
        pa_vals  = merged_raw[key]["prop_active"]
        pgi_vals = merged_raw[key]["prop_grid_like_in_active"]

        if APPLY_FILTER_TO_PGI_ACTIVITY:
            kept_pgi, _, _, _ = filter_structural_zeros(
                pgi_vals, tau_high=TAU_HIGH, q_major=Q_MAJOR, eps=EPS_ZERO, n_min=N_MIN_GRP
            )
        else:
            kept_pgi = np.asarray([v for v in pgi_vals if np.isfinite(v)], dtype=float)

        if APPLY_FILTER_TO_PA_ACTIVITY:
            kept_pa, _, _, _ = filter_structural_zeros(
                pa_vals, tau_high=TAU_HIGH, q_major=Q_MAJOR, eps=EPS_ZERO, n_min=N_MIN_GRP
            )
        else:
            kept_pa = np.asarray([v for v in pa_vals if np.isfinite(v)], dtype=float)

        m_pa, s_pa = mean_std_safe(kept_pa.tolist())
        m_pgi, s_pgi = mean_std_safe(kept_pgi.tolist())

        e_pa  = sem_from_std(s_pa,  len(kept_pa))  if ERRORBAR_SEM else s_pa
        e_pgi = sem_from_std(s_pgi, len(kept_pgi)) if ERRORBAR_SEM else s_pgi

        means_active.append(m_pa); errs_active.append(e_pa)
        means_grid.append(m_pgi);  errs_grid.append(e_pgi)

    ax.errorbar(sizes, means_active, yerr=errs_active, fmt='-o', capsize=4, label="prop_active", zorder=2)
    ax.errorbar(sizes, means_grid,  yerr=errs_grid,  fmt='-s', capsize=4, label="prop_grid_like_in_active", zorder=2)
    ax.set_xlabel("size")
    ax.set_title(f"Cell activity vs size (rho={rho_fixed})")
    ax.grid(False)
    ax.legend(loc="best")

def create_combined_svg_activity_vs_noise(merged_raw, rho_fixed: float, out_dir: str):
    """
    Combined SVG (order adjusted): [proportions vs noise std] || [activity vs size], NO grids.
    Left : props vs noise std (noise_mod), same rho_fixed and noise_mod.SIZE_FIXED.
    Right: activity vs size (this module).
    """
    # collect noise results via the noise module
    data_by_key = noise_mod.collect_noise_results(noise_mod.NOISE_RESULTS_DIRS)
    stds, (pa_mean, pa_err), (pgi_mean, pgi_err), _stats = noise_mod.aggregate_by_std(
        data_by_key, rho_fixed=rho_fixed, size_fixed=noise_mod.SIZE_FIXED
    )

    if stds.size == 0:
        print(f"[WARN] No noise data found for rho={rho_fixed}, size={noise_mod.SIZE_FIXED} in {list(noise_mod.NOISE_RESULTS_DIRS)}")
        return

    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # Left: props vs noise std (match style; NO grid)
    axs[0].errorbar(stds, pa_mean,  yerr=pa_err,  fmt='-o', capsize=4, label="prop_active", zorder=2)
    axs[0].errorbar(stds, pgi_mean, yerr=pgi_err, fmt='-s', capsize=4, label="prop_grid_like_in_active", zorder=2)
    axs[0].set_xlabel("noise std")
    axs[0].set_ylabel("Proportion")  # shared semantic on the left now
    axs[0].set_title(f"Proportions vs noise (rho={rho_fixed}, size={noise_mod.SIZE_FIXED})")
    axs[0].grid(False)
    axs[0].legend(loc="best")

    # Right: activity vs size
    _plot_activity_vs_size_on_ax(axs[1], merged_raw, rho_fixed)

    fig.tight_layout()

    # build filename with p-style floats (e.g., 0.8 -> 0p8)
    def _pfloat(x: float) -> str:
        return str(x).replace(".", "p")

    save_path = os.path.join(
        out_dir,
        f"combo_activity_vs_noise_rho{_pfloat(rho_fixed)}_size{_pfloat(noise_mod.SIZE_FIXED)}.svg"
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {save_path}")

# ----------------------------- Main ---------------------------

def load_and_aggregate():
    """Load from multiple directories, merge, and aggregate. Returns (merged_raw_lists, aggregated_means_stds)."""
    all_data_dicts = []
    for d in RESULTS_DIRS:
        dd = load_results_from_dir(d)
        all_data_dicts.append(dd)

    merged_data = merge_data_dicts(all_data_dicts)
    if not merged_data:
        raise RuntimeError(f"No JSON results found in: {RESULTS_DIRS}")

    agg_data = aggregate_data(merged_data)
    return merged_data, agg_data

def main():
    # Load and aggregate
    merged_raw, agg_data = load_and_aggregate()

    # Determine available unique rhos and sizes
    unique_rhos  = sorted({rho for (rho, _) in agg_data.keys()})
    unique_sizes = sorted({size for (_, size) in agg_data.keys()})

    # Pick fixed params (use nearest if requested value does not exist)
    size_fixed = SIZE_FIXED_DEFAULT
    if size_fixed not in unique_sizes:
        nearest = nearest_value(SIZE_FIXED_DEFAULT, unique_sizes)
        print(f"[INFO] size={SIZE_FIXED_DEFAULT} not found. Using nearest available size={nearest}")
        size_fixed = nearest

    rho_fixed = RHO_FIXED_DEFAULT
    if rho_fixed not in unique_rhos:
        nearest = nearest_value(RHO_FIXED_DEFAULT, unique_rhos)
        print(f"[INFO] rho={RHO_FIXED_DEFAULT} not found. Using nearest available rho={nearest}")
        rho_fixed = nearest

    # ---------------- PNG (unchanged model sets except cubic) ----------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_spacing_vs_rho_with_fits(agg_data, size_fixed=size_fixed, out_dir=PLOTS_DIR)
    plot_spacing_vs_rhoTRUE_with_fits(agg_data, size_fixed=size_fixed, out_dir=PLOTS_DIR)  # x uses filtered true rho

    # Uncorrected spacing vs size (baseline)
    plot_spacing_vs_size_with_fits(agg_data, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)

    # rho-corrected spacing vs size (remove true-rho influence)
    plot_spacing_vs_size_rho_corrected(agg_data, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)

    # Activity plot uses merged_raw (raw lists) so the heuristic can operate per-run
    plot_activity_vs_size(merged_raw, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)

    # ---------------- SVG (combined) ----------------
    create_combined_svg_spacing_vs_rho(agg_data, size_fixed=size_fixed, out_dir=PLOTS_DIR)          # no grid (kept models)
    create_combined_svg_spacing_vs_size(agg_data, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)          # no grid (cubic+linear)
    # create_svg_activity_vs_size(...)  # DISABLED per request: do not generate activity_vs_size_rho*.svg

    # NEW combined SVG: true-rho panel (ONLY 1/x) || rho-corrected vs size (ONLY cubic)
    create_combined_svg_rho_handled(agg_data, size_fixed=size_fixed, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)

    # Combined SVG: props vs noise (left) || activity vs size (right) — order adjusted
    create_combined_svg_activity_vs_noise(merged_raw, rho_fixed=rho_fixed, out_dir=PLOTS_DIR)

    print("[DONE] All plots saved.")

if __name__ == "__main__":
    main()
