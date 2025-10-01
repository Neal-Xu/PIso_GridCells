import os
import matplotlib
matplotlib.use("Agg")  # headless: no Tk, no GUI windows

import scipy
import spatial_maps as sm
from ripser import ripser
from persim import plot_diagrams
import plotly.express as px
import plotly.io as pio
from Tools.plotting_functions import *
from Tools.utils import *
import json
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3d projection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def result_ratemap(ratemaps, gs, file_appendix):
    """
    Save three outputs:
      1) {file_appendix}_rate.png       —— 25 example rate-maps (unchanged)
      2) {file_appendix}_grid_dist.png  —— Grid & Square Score + Spacing + Orientation (unchanged)
      3) {file_appendix}_combined.svg   —— Top: 4x4 square rate-maps; Bottom: 3 square histograms
    """

    # ---------- PNG 1: 25 example rate-maps (original helper) ----------
    fig_rate, _, _ = multiimshow(ratemaps[:25], figsize=(5, 5))
    fig_rate.savefig(file_appendix + "_rate.png", dpi=300)
    plt.close(fig_rate)
    print("saved rate-map PNG")

    # ---------- PNG 2: distributions (original logic preserved) ----------
    fig_dist, axes = plt.subplots(1, 3, figsize=(15, 4))

    grid_score   = np.asarray(gs.get("grid_score", []),   dtype=float)
    square_score = np.asarray(gs.get("square_score", []), dtype=float)
    spacing      = np.asarray(gs.get("spacings", []),     dtype=float)
    orientation  = np.asarray(gs.get("orientations", []), dtype=float)

    grid_score   = grid_score[np.isfinite(grid_score)]
    square_score = square_score[np.isfinite(square_score)]
    spacing      = spacing[np.isfinite(spacing)]
    orientation  = orientation[np.isfinite(orientation)]

    # (1) Grid / Square Score histogram
    if grid_score.size > 0 or square_score.size > 0:
        lo_candidates, hi_candidates = [], []
        if grid_score.size > 0:
            lo_candidates.append(float(np.min(grid_score)))
            hi_candidates.append(float(np.max(grid_score)))
        if square_score.size > 0:
            lo_candidates.append(float(np.min(square_score)))
            hi_candidates.append(float(np.max(square_score)))
        bin_range = (min(lo_candidates), max(hi_candidates)) if lo_candidates else (0.0, 1.0)
        axes[0].hist(grid_score,   bins=32, range=bin_range, alpha=0.9, label="Grid Score")
        axes[0].hist(square_score, bins=32, range=bin_range, alpha=0.9, label="Square Score")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("Score")

    # (2) Spacing
    if spacing.size > 0:
        axes[1].hist(spacing, bins=32)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("Spacing")

    # (3) Orientation
    if orientation.size > 0:
        axes[2].hist(orientation, bins=32)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_xlabel("Orientation")

    fig_dist.tight_layout()
    fig_dist.savefig(f"{file_appendix}_grid_dist.png", dpi=300)
    plt.close(fig_dist)
    print("saved distribution PNG (grid & square, spacing, orientation)")

    # ---------- SVG: precise manual layout (all square panels) ----------
    mpl.rcParams['svg.fonttype'] = 'none'  # keep text as editable text in SVG

    # -- Layout in figure-normalized coordinates --
    left_pad   = 0.06   # outer left margin
    right_pad  = 0.06   # outer right margin
    top_pad    = 0.05   # outer top margin
    bottom_pad = 0.07   # outer bottom margin
    v_gap      = 0.05   # vertical gap between top and bottom panels
    inner_pad  = 0.015  # gap between subplots inside a panel (in figure units)

    # Panel width (normalized): full width minus outer margins
    panel_w = 1.0 - left_pad - right_pad

    # --- TOP: 4x4 square images (each cell is a square) ---
    # cell size is set by width; 4 cells + 3 inner gaps must fit panel_w
    cell = (panel_w - 3 * inner_pad) / 4.0
    panel_h_top = 4 * cell + 3 * inner_pad

    # --- BOTTOM: 1x3 square histograms ---
    cell_bot = (panel_w - 2 * inner_pad) / 3.0
    panel_h_bot = cell_bot

    # Ensure total height fits [0,1]; if not, scale down paddings proportionally
    total_h_need = top_pad + panel_h_top + v_gap + panel_h_bot + bottom_pad
    if total_h_need > 1.0:
        scale = 1.0 / total_h_need
        top_pad *= scale; v_gap *= scale; bottom_pad *= scale
        cell *= scale; cell_bot *= scale
        panel_h_top = 4 * cell + 3 * inner_pad
        panel_h_bot = cell_bot

    # Create figure with a reasonable physical size following aspect
    W_inch = 8.0
    H_inch = W_inch * (top_pad + panel_h_top + v_gap + panel_h_bot + bottom_pad)
    fig_svg = plt.figure(figsize=(W_inch, H_inch))

    # ---------- TOP PANEL: place 16 square axes by hand ----------
    top_y0 = bottom_pad + panel_h_bot + v_gap  # bottom of the top panel
    n_show = min(16, len(ratemaps))
    for r in range(4):         # row: 0..3 (from bottom to top)
        for c in range(4):     # col: 0..3 (from left to right)
            idx = r * 4 + c
            # compute left-bottom corner for (r, c)
            ax_left = left_pad + c * (cell + inner_pad)
            ax_bottom = top_y0 + r * (cell + inner_pad)
            ax = fig_svg.add_axes([ax_left, ax_bottom, cell, cell])  # [l, b, w, h] normalized
            ax.axis('off')
            if idx < n_show:
                ax.imshow(ratemaps[idx], aspect='equal', origin='upper')

    # ---------- BOTTOM PANEL: 3 square histograms with extra horizontal gap ----------
    inner_pad_x = 0.035  # larger horizontal gap for bottom plots
    cell_bot = (panel_w - 2 * inner_pad_x) / 3.0
    panel_h_bot = cell_bot

    bot_y0 = bottom_pad
    ax0_left = left_pad
    ax1_left = left_pad + cell_bot + inner_pad_x
    ax2_left = left_pad + 2 * cell_bot + 2 * inner_pad_x

    ax0 = fig_svg.add_axes([ax0_left, bot_y0, cell_bot, cell_bot])
    ax1 = fig_svg.add_axes([ax1_left, bot_y0, cell_bot, cell_bot])
    ax2 = fig_svg.add_axes([ax2_left, bot_y0, cell_bot, cell_bot])

    # Adjust tick label style to reduce crowding
    for ax in (ax0, ax1, ax2):
        ax.tick_params(axis='y', pad=1, labelsize=8)
        ax.tick_params(axis='x', labelsize=8)

    # Reuse filtered data; set fixed/robust x-limits per your spec
    GRID_SCORE_XLIM = (0.0, 1.5)

    if "spacing_range" in gs and isinstance(gs["spacing_range"], (list, tuple)) and len(gs["spacing_range"]) == 2:
        SPACING_XLIM = (float(gs["spacing_range"][0]), float(gs["spacing_range"][1]))
    else:
        if spacing.size > 0:
            sp_hi = float(np.percentile(spacing, 99.5)) * 1.25
            SPACING_XLIM = (0.0, max(sp_hi, float(np.nanmax(spacing)) * 1.05, 1.0))
        else:
            SPACING_XLIM = (0.0, 1.0)

    if "orientation_range" in gs and isinstance(gs["orientation_range"], (list, tuple)) and len(gs["orientation_range"]) == 2:
        ORI_XLIM = (float(gs["orientation_range"][0]), float(gs["orientation_range"][1]))
    else:
        if orientation.size > 0 and float(np.nanmax(orientation)) <= 3.5:
            ORI_XLIM = (0.0, float(np.pi / 3))  # radians, 6-fold symmetry
        else:
            ORI_XLIM = (0.0, 60.0)              # degrees

    # (1) Grid Score
    if grid_score.size > 0:
        ax0.hist(grid_score, bins=32, range=GRID_SCORE_XLIM, alpha=0.9)
    else:
        ax0.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_xlabel("Score")
    ax0.set_xlim(GRID_SCORE_XLIM)
    ax0.grid(True, alpha=0.3)

    # (2) Spacing
    if spacing.size > 0:
        ax1.hist(spacing, bins=32, range=SPACING_XLIM)
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("Spacing")
    ax1.set_xlim(SPACING_XLIM)
    ax1.grid(True, alpha=0.3)

    # (3) Orientation
    if orientation.size > 0:
        ax2.hist(orientation, bins=32, range=ORI_XLIM)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Orientation")
    ax2.set_xlim(ORI_XLIM)
    ax2.grid(True, alpha=0.3)

    # ---------- Export SVG (exact manual layout) ----------
    svg_path = f"{file_appendix}_combined.svg"
    fig_svg.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig_svg)
    print(f"saved combined SVG (square 4x4 top + 3 square histograms bottom) -> {svg_path}")




def result_homology(ratemaps, file_appendix):
    """
    Compute persistent homology on population activity (cells as features),
    save PNG + (standalone) SVG of the persistence diagrams,
    and return the computed diagrams for later composition.
    """
    activity = ratemaps.reshape(ratemaps.shape[0], -1)  # cells × pixels
    dgms = ripser(activity.T, maxdim=2, n_perm=300)["dgms"]

    # PNG
    fig = plt.figure(figsize=(5, 5))
    plot_diagrams(dgms)
    fig.savefig(f"{file_appendix}_homology.png", dpi=300)
    plt.close(fig)
    print("saved homology diagram PNG")

    # Standalone SVG
    mpl.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure(figsize=(5, 5))
    plot_diagrams(dgms)
    fig.savefig(f"{file_appendix}_homology.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("saved homology diagram SVG")

    return dgms


def result_projection(ratemaps, file_appendix, show_interactive=False):
    """
    UMAP 3D embedding; save PNG + JSON (+ HTML).
    By default do NOT open interactive window to avoid Tk/thread issues.
    """
    activity = ratemaps.reshape(ratemaps.shape[0], -1)
    embedding = proj_umap(activity)
    cols_flat = def_cols(activity)

    fig = px.scatter_3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        color=cols_flat, color_continuous_scale="Viridis"
    )
    # static exports (safe in headless/batch)
    fig.write_image(f"{file_appendix}_topology.png", scale=3)
    pio.write_json(fig, f"{file_appendix}_topology.json")
    fig.write_html(f"{file_appendix}_topology.html", include_plotlyjs="cdn")

    # only show if explicitly requested and in an environment that supports it
    if show_interactive:
        try:
            pio.renderers.default = "browser"  # open in system browser (no Tk)
            fig.show()
        except Exception as e:
            print(f"[WARN] fig.show() skipped: {e}")

    print("saved topology of population activity (PNG + JSON + HTML)")
    return embedding, cols_flat


# ----------------------- helpers-----------------------


def save_combined_svg_homology_and_views_plotly(
    embedding, colors, file_appendix,
    colorscale=None,                 # None -> cyclic HSV
    marker_size=None,                # None -> auto by N
    marker_opacity=0.95,
    out_svg=False, out_png=False,
    gap=0.02,                       # gap between subplots (0~0.02)
    panel_pad=0.01,                 # outer padding (0~0.01)
    auto_marker_size=True,           # scale size by N if marker_size is None
    width=1800, height=1300          # canvas size (controls subplot size)
):
    """
    Interactive 2×2 Plotly panel (transparent, no axes, orthographic).
    - No PCA alignment (removed).
    - Center & scale data to fill [-1,1]^3, and fix each scene axis range to [-1,1],
      so the point cloud fully occupies the subplot.
    - Tight layout via manual scene domains; minimal gaps/borders.
    - Use the modebar in the HTML to export the EXACT view as SVG/PNG.
    """

    # ---------- shape check ----------
    X = np.asarray(embedding)
    if X.ndim != 2:
        raise ValueError(f"'embedding' must be 2D, got {X.shape}")
    if X.shape[1] == 3:
        pass
    elif X.shape[0] == 3:
        X = X.T
    else:
        raise ValueError(f"'embedding' must have 3 coords; got {X.shape}")
    X = X.astype(float, copy=False)
    N = len(X)

    # ---------- center to origin & scale to fill [-1,1]^3 ----------
    Xc = X - X.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(Xc))
    if np.isfinite(max_abs) and max_abs > 0:
        Xc = Xc / max_abs  # now spans roughly [-1,1] in its largest dimension
    X = Xc

    # ---------- colors (cyclic by default) ----------
    use_colors, colvec = False, None
    try:
        if colors is not None and len(colors) == N:
            colvec = np.asarray(colors); use_colors = True
    except Exception:
        pass

    if not use_colors:
        # cyclic HSV by angle in first two coordinates (already centered & scaled)
        ang = (np.arctan2(X[:,1], X[:,0]) + np.pi) / (2*np.pi)  # [0,1]
        colvec = ang
        use_colors = True

    if colorscale is None:
        import colorsys
        def hsv_color(h):
            r,g,b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        colorscale = [[i/8.0, hsv_color(i/8.0)] for i in range(9)]  # cyclic

    # ---------- auto marker size (pixel) ----------
    if marker_size is None and auto_marker_size:
        # heuristic: ~sqrt-density; clamp to reasonable range
        ms = 65.0 / (N**0.5)
        marker_size = float(np.clip(ms, 1.1, 2.6))
    elif marker_size is None:
        marker_size = 1.6

    # ---------- figure ----------
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=gap, vertical_spacing=gap
    )

    def add_points(r, c):
        mk = dict(size=marker_size, opacity=marker_opacity,
                  colorscale=colorscale, showscale=False)
        if use_colors:
            mk["color"] = colvec
        fig.add_trace(
            go.Scatter3d(
                x=X[:,0], y=X[:,1], z=X[:,2],
                mode="markers", marker=mk, showlegend=False
            ),
            row=r, col=c
        )

    add_points(1, 1)
    add_points(1, 2)
    add_points(2, 1)
    add_points(2, 2)

    # ---------- tight, almost full-bleed domains ----------
    x_mid = 0.5
    y_mid = 0.5
    left  = panel_pad
    right = 1.0 - panel_pad
    top   = 1.0 - panel_pad
    bot   = panel_pad

    def dom(lo, hi, mid, g):
        return [lo, mid - g/2.0], [mid + g/2.0, hi]

    (xL, xR) = dom(left, right, x_mid, gap)
    (yB, yT) = dom(bot,  top,   y_mid, gap)

    # ---------- common scene options: fixed ranges fill the subplot ----------
    axis_hidden = dict(visible=False, range=[-1, 1])  # range fixed to fill panel
    scene_common = dict(
        xaxis=axis_hidden, yaxis=axis_hidden, zaxis=axis_hidden,
        bgcolor="rgba(0,0,0,0)",
        aspectmode="cube",
        camera=dict(projection=dict(type="orthographic"))
    )

    fig.update_layout(
        scene = {**scene_common, "domain": {"x": xL, "y": yT}},  # TL
        scene2= {**scene_common, "domain": {"x": xR, "y": yT}},  # TR
        scene3= {**scene_common, "domain": {"x": xL, "y": yB}},  # BL
        scene4= {**scene_common, "domain": {"x": xR, "y": yB}},  # BR
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        width=width, height=height
    )

    # ---------- interactive HTML (modebar supports SVG/PNG export) ----------
    html_path = f"{file_appendix}_manifold_panel.html"
    cfg = dict(
        displaylogo=False,
        toImageButtonOptions=dict(format="svg",
                                  filename=f"{file_appendix}_manifold_panel",
                                  scale=1)
    )
    pio.write_html(fig, file=html_path, include_plotlyjs="cdn", config=cfg, full_html=True)
    print(f"saved interactive HTML -> {html_path} (use modebar to download SVG/PNG)")

    # optional static exports (initial camera only; kaleido required)
    if out_svg:
        fig.write_image(f"{file_appendix}_manifold_panel.svg")
    if out_png:
        fig.write_image(f"{file_appendix}_manifold_panel.png", scale=3)

    return fig

def analyze_grid_cells(
    ratemaps,
    gs,
    grid_score_threshold,
    activity_threshold=0.5,
    file_appendix=None
):

    max_rates = ratemaps.max(axis=(1, 2))
    is_active = max_rates > activity_threshold

    grid_scores = np.array(gs["grid_score"])
    spacings = np.array(gs["spacings"])

    grid_scores_active = grid_scores[is_active]
    spacings_active = spacings[is_active]

    is_grid_like = grid_scores_active > grid_score_threshold
    spacings_grid_like = spacings_active[is_grid_like]

    num_total = len(ratemaps)
    num_active = is_active.sum()
    num_grid_like = is_grid_like.sum()

    result = {
        "num_total": int(num_total),
        "num_active": int(num_active),
        "prop_active": float(num_active / num_total) if num_total > 0 else 0.0,
        "num_grid_like": int(num_grid_like),
        "prop_grid_like_in_active": float(num_grid_like / num_active) if num_active > 0 else 0.0,
        "mean_spacing_grid_like": float(spacings_grid_like.mean()) if num_grid_like > 0 else 0.0,
        "rho_true_mean": float(gs.get("rho_true_mean", float('nan'))),
        "rho_true_std": float(gs.get("rho_true_std", float('nan'))),
    }

    if file_appendix is not None:
        # Extract tag from file_appendix: looks like "data/figures/model_rhoXpY_sizeZqW_runN/FC"
        parts = file_appendix.split("model_")[-1].split(os.sep)[0]  # rhoXpY_sizeZqW_runN
        tag = parts  # e.g. rho0p8_size0p995_run2

        # Save to central directory
        results_dir = os.path.join("data", "results")
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, f"{tag}.json")

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result