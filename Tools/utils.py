import numpy as np
import scipy
import spatial_maps as sm
from spatial_maps.gridcells import peak_to_peak_distance, masked_corrcoef2d
import umap
from sklearn.decomposition import PCA


def def_cols(activity):
    xcol = np.arange(0, int(np.sqrt(activity.shape[-1])))
    ycol = np.arange(0, int(np.sqrt(activity.shape[-1])))

    xx, yy = np.meshgrid(xcol, ycol)
    cols = xx + yy
    cols_flat = cols.flatten()
    return cols_flat


def proj_umap(activity):
    n_neighbors = 2000
    n_components = 3
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.8)
    reducer.fit(activity.T)
    embedding = reducer.transform(activity.T)
    return embedding

def proj_pca(activity):
    n_components = 3
    reducer = PCA(n_components=n_components)
    reducer.fit(activity.T)
    embedding = reducer.transform(activity.T)
    er = 100 * np.sum(reducer.explained_variance_ratio_)  # Explained variance
    return embedding, er


def grid_feature(x, y, g, bins):
    ratemaps = scipy.stats.binned_statistic_2d(x, y, g.reshape(-1, g.shape[-1]).T, bins=bins)[0]

    grid_score = np.array([sm.gridness(ratemaps[i]) for i in range(len(ratemaps))])
    square_score = np.array([squareness(ratemaps[i]) for i in range(len(ratemaps))])

    valid_mask = np.sum(ratemaps, axis=(-2, -1)) > 0
    valid_ratemaps = ratemaps[valid_mask]

    acorrs = np.array([sm.autocorrelation(g) for g in valid_ratemaps])
    peaks = [sm.find_peaks(acorr) for acorr in acorrs]

    valid_spacings = np.array([
        sm.spacing_and_orientation(peak, acorrs.shape[-1])[0]
        for peak in peaks
    ])
    valid_orientations = np.array([
        sm.spacing_and_orientation(peak, acorrs.shape[-1])[1]
        for peak in peaks
    ])

    spacings = np.full(len(ratemaps), np.nan)
    orientations = np.full(len(ratemaps), np.nan)

    spacings[valid_mask] = valid_spacings
    orientations[valid_mask] = valid_orientations

    return ratemaps, [grid_score, square_score], spacings, orientations



def filter_ratemap(ratemaps, gs):
    grid_score, spacings, orientations = gs["grid_score"], gs["spacings"], gs["orientations"]
    # gs_big_o = ratemaps[orientations > 0.45]
    # gs_mid_o = ratemaps[(orientations > 0.4) & (orientations < 0.8)]
    # gs_low_o = ratemaps[(orientations < 0.05) + (orientations > 0.96)]
    ratemap_big_o = ratemaps[grid_score > 0.75]
    ratemap_mid_o = ratemaps[(grid_score > 0) & (grid_score < 0.75)]
    ratemap_low_o = ratemaps[grid_score < 0]

    grid_score_big_o = grid_score[grid_score > 0.75]
    grid_score_mid_o = grid_score[(grid_score > 0) & (grid_score < 0.75)]
    grid_score_low_o = grid_score[grid_score < 0]

    spacings_big_o = spacings[grid_score > 0.75]
    spacings_mid_o = spacings[(grid_score > 0) & (grid_score < 0.75)]
    spacings_low_o = spacings[grid_score < 0]

    orientations_big_o = orientations[grid_score > 0.75]
    orientations_mid_o = orientations[(grid_score > 0) & (grid_score < 0.75)]
    orientations_low_o = orientations[grid_score < 0]

    gs_big_o = {'grid_score': grid_score_big_o, 'spacings': spacings_big_o, 'orientations': orientations_big_o}
    gs_mid_o = {'grid_score': grid_score_mid_o, 'spacings': spacings_mid_o, 'orientations': orientations_mid_o}
    gs_low_o = {'grid_score': grid_score_low_o, 'spacings': spacings_low_o, 'orientations': orientations_low_o}


    return [ratemap_big_o, ratemap_mid_o, ratemap_low_o], [gs_big_o, gs_mid_o, gs_low_o]


def filter_ratemap_high_gridscore(ratemaps, gs, threshold=0.1):

    grid_scores = gs["grid_score"]
    spacings = gs["spacings"]
    orientations = gs["orientations"]
    square_scores = gs["square_score"]

    mask = grid_scores > threshold

    ratemaps_selected = ratemaps[mask]

    grid_metrics_selected = {
        "grid_score": grid_scores[mask],
        "spacings": spacings[mask],
        "orientations": orientations[mask],
        "square_score": square_scores[mask]
    }

    return ratemaps_selected, grid_metrics_selected


def squareness(rate_map, return_mask=False):
    import numpy.ma as ma

    rate_map = rate_map.copy()
    rate_map[~np.isfinite(rate_map)] = 0
    acorr = sm.autocorrelation(rate_map, mode="full", normalize=True)

    acorr_maxima = sm.find_peaks(acorr)
    inner_radius = 0.5 * peak_to_peak_distance(acorr_maxima, 0, 1)
    outer_radius = inner_radius + peak_to_peak_distance(acorr_maxima, 0, 6)

    # limit radius to smallest side of map and ensure inner < outer
    outer_radius = np.clip(outer_radius, 0.0, min(acorr.shape) / 2)
    inner_radius = np.clip(inner_radius, 0.0, outer_radius)

    # Speed up the calculation by limiting the autocorr map to the outer area
    center = np.array(acorr.shape) / 2
    lower = (center - outer_radius).astype(int)
    upper = (center + outer_radius).astype(int)
    acorr = acorr[lower[0]: upper[0], lower[1]: upper[1]]

    # create a mask
    ylen, xlen = acorr.shape  # ylen, xlen is the correct order for meshgrid
    x = np.linspace(-xlen / 2.0, xlen / 2.0, xlen)
    y = np.linspace(-ylen / 2.0, ylen / 2.0, ylen)
    X, Y = np.meshgrid(x, y)
    distance_map = np.sqrt(X ** 2 + Y ** 2)
    mask = (distance_map < inner_radius) | (distance_map > outer_radius)

    # calculate the correlation with the rotated maps
    r45, r90 = rotate_corr(acorr, mask=mask)
    squrescore = float(np.min(r90) - np.max(r45))

    if return_mask:
        return squrescore, ma.masked_array(acorr, mask=mask)

    return squrescore


def rotate_corr(acorr, mask):
    import numpy.ma as ma
    from scipy.ndimage import rotate

    m_acorr = ma.masked_array(acorr, mask=mask)
    angles = range(45, 180 + 45, 45)
    corr = []
    # Rotate and compute correlation coefficient
    for angle in angles:
        rot_acorr = rotate(acorr, angle, reshape=False)
        rot_acorr = ma.masked_array(rot_acorr, mask=mask)
        corr.append(masked_corrcoef2d(rot_acorr, m_acorr)[0, 1])
    r90 = corr[1::2]
    r45 = corr[::2]
    return r45, r90

