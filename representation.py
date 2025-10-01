from config import *
from Network.model import *
from Network.datamaker import *
from Tools.statistic import *
from Tools.utils import grid_feature, filter_ratemap

import argparse
import os
import torch


@torch.no_grad()
def compute_true_rho_diff(model, r, v, device, dx=1e-8, along_dim=0):
    """
    ρ_true ≈ ||g(x+dx/2) - g(x-dx/2)|| / dx
    """
    r = r.to(device)
    v = v.to(device) if v is not None else None

    r_plus  = r.clone()
    r_minus = r.clone()
    r_plus[...,  along_dim] += dx * 0.5
    r_minus[..., along_dim] -= dx * 0.5

    g_plus,  _ = model(rs=r_plus,  vs=v)
    g_minus, _ = model(rs=r_minus, vs=v)

    dg  = torch.linalg.norm(g_plus - g_minus, dim=-1)   # (N,)
    rho = dg / dx                                       # (N,)
    return rho, rho.mean().item(), rho.std(unbiased=False).item()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=float, default=1.0, help='dg / dr')
    parser.add_argument('--size', type=float, default=0.995, help='size of torus, range: [0,1)')
    parser.add_argument('--run', type=int, default=0, help='repeat index')  # NEW
    args = parser.parse_args()

    # Set up paths
    currentFolder = os.path.dirname(os.path.abspath(__file__))
    modelFolder = os.path.join(currentFolder, 'data', 'saved-models')
    file_appendix_str = f"rho{args.rho}_size{args.size}_run{args.run}".replace('.', 'p')  # NEW
    model_file = os.path.join(modelFolder, f"model_{file_appendix_str}.pth")
    figureFolder = os.path.join(currentFolder, 'data', 'figures', f"model_{file_appendix_str}")
    os.makedirs(figureFolder, exist_ok=True)

    # Generate dataset
    dataConfig = DatasetConfig()
    dataset = DatasetMakerRandom(dataConfig)

    # Initialize model
    modelConfig = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelConfig.device = device
    model = FFGC(modelConfig)

    # Load model weights
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Forward pass to get activations
    activations = []
    r, v = dataset.generate_data()
    with torch.no_grad():
        norm_rep, _ = model(rs=r.to(device), vs=v.to(device))
    activations.append(norm_rep)

    # Compute ratemaps
    bins = dataConfig.bins
    x, y = r[..., 0].ravel().cpu().numpy(), r[..., 1].ravel().cpu().numpy()
    rep = activations[-1]
    g = rep.detach().cpu().numpy()
    file_appendix = os.path.join(figureFolder, "FC")

    ratemaps, scores, spacings, orientations = grid_feature(x, y, g, bins=bins)


    rho_per_sample, rho_mean, rho_std = compute_true_rho_diff(
        model=model, r=r, v=v, device=device, dx=1e-3, along_dim=0
    )
    print(rho_per_sample)

    gs = {
        'grid_score': scores[0],
        'spacings': spacings,
        'orientations': orientations,
        'square_score': scores[1],
        'rho_true_mean': float(rho_mean),
        'rho_true_std': float(rho_std),
    }

    return ratemaps, gs, file_appendix


if __name__ == '__main__':
    ratemaps, gs, file_appendix = main()

    # some analysis and visualization of the ratemaps---------------------------------

    result_ratemap(ratemaps, gs, file_appendix)
    dgms = result_homology(ratemaps, file_appendix)
    embedding, cols_flat = result_projection(ratemaps, file_appendix)
    save_combined_svg_homology_and_views_plotly(embedding, colors=None, file_appendix=file_appendix, colorscale=None)
    analyze_grid_cells(ratemaps, gs, grid_score_threshold=0.7, file_appendix=file_appendix)

