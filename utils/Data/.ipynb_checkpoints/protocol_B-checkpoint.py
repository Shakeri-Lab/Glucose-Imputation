import numpy as np
from scipy.signal import find_peaks

def _cluster_indices(indices, merge_gap_pts):
    """Group indices that are closer than merge_gap_pts."""
    if indices.size == 0: return []
    clusters = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev <= merge_gap_pts: 
            prev = idx
        else:
            clusters.append((start, prev))
            start = prev = idx
    clusters.append((start, prev))
    return clusters

def _smooth_signal(cgm, window_pts):
    """Apply simple moving average smoothing if window > 1."""
    if window_pts > 1:
        kernel = np.ones(window_pts, dtype=float) / window_pts
        return np.convolve(cgm, kernel, mode="same")
    return cgm

def _get_peak_in_window(cgm, all_peaks, search_start, search_end):
    """
    Finds the first valid peak in a specific window.
    """
    if search_end <= search_start + 1: return None
    candidates = all_peaks[(all_peaks >= search_start) & (all_peaks < search_end)]
    if candidates.size > 0: return int(candidates[0])
    
    window_slice = cgm[search_start:search_end]
    if len(window_slice) == 0: return None
    return int(search_start + np.nanargmax(window_slice))

def _map_clusters_to_peaks(cgm, clusters, all_peaks, missing_config, POINTS_PER_HOUR):
    """
    Iterates over all clusters and finds the associated post-prandial peak.
    Returns list of (meal_start, peak_idx).
    """
    T = len(cgm)
    lag_pts = int((int(missing_config.get("min_lag_minutes", 15)) / 60) * POINTS_PER_HOUR)
    horizon_pts = int(float(missing_config.get("horizon_hours", 4)) * POINTS_PER_HOUR)
    
    pairs = []
    for i, (meal_start, _) in enumerate(clusters):
        search_start = meal_start + lag_pts
        search_end = min(T, meal_start + horizon_pts)

        if i + 1 < len(clusters):
            next_cluster_start = clusters[i + 1][0]
            search_end = min(search_end, next_cluster_start)

        peak_idx = _get_peak_in_window(cgm, all_peaks, search_start, search_end)
        
        if peak_idx is not None: pairs.append((int(meal_start), peak_idx))
            
    return pairs

def _calculate_mask_bounds(meal_start, peak_idx, T, missing_config, POINTS_PER_HOUR):
    """
    Calculates mask bounds to CENTER the peak within the window.
    Maximize difficulty for linear interpolation.
    """
    min_wlen = int(float(missing_config.get("min_mask_hours", 3.5)) * POINTS_PER_HOUR)
    max_wlen = int(float(missing_config.get("max_mask_hours", 4)) * POINTS_PER_HOUR)

    target_wlen = np.random.randint(min_wlen, max_wlen + 1)    
    wlen = min(target_wlen, T - meal_start)
    
    half_window = wlen // 2
    ideal_start = peak_idx - half_window
    start = max(meal_start, ideal_start)
    
    if start + wlen > T: start = max(meal_start, T - wlen)
    end = start + wlen
    return int(start), int(end)

def _apply_masks(mask, pairs, num_to_hide, missing_config, POINTS_PER_HOUR):
    """Selects random pairs and updates the mask in-place."""
    T = len(mask)
    selected_pairs = list(pairs)
    np.random.shuffle(selected_pairs)
    
    count = 0
    for meal_start, peak_idx in selected_pairs:
        if count >= num_to_hide: break  
        start, end = _calculate_mask_bounds(meal_start, peak_idx, T, missing_config, POINTS_PER_HOUR)
        mask[start:end] = True
        count += 1
        
    return mask


def apply_protocol_B_hidden_peak_engine(drop_mask, day_df, missing_config, POINTS_PER_HOUR=12):
    """
    Main entry point.  
    """
    num_to_hide = int(missing_config.get("num_meal_hide", 0))
    if num_to_hide <= 0: return drop_mask

    cgm_raw = day_df["cgm"].to_numpy().astype(float)
    meal_indices = np.flatnonzero(day_df["meal"].to_numpy() > 0)
    if meal_indices.size == 0: return drop_mask
    meal_indices.sort()

    working_mask = drop_mask.copy()

    cluster_gap_mins = int(missing_config.get("cluster_gap_minutes", 60))
    merge_pts = int((cluster_gap_mins / 60) * POINTS_PER_HOUR)
    clusters = _cluster_indices(meal_indices, merge_pts)

    if len(clusters) < num_to_hide: return drop_mask
    smooth_pts = int(missing_config.get("smooth_window_pts", 3))
    cgm_s = _smooth_signal(cgm_raw, smooth_pts)
    
    prominence = float(missing_config.get("peak_prominence_mgdl", 10))
    dist_mins = int(missing_config.get("peak_distance_minutes", 30))
    dist_pts = max(1, int((dist_mins / 60) * POINTS_PER_HOUR))
    
    all_peaks, _ = find_peaks(cgm_s, prominence=prominence, distance=dist_pts)
    pairs = _map_clusters_to_peaks(cgm_s, clusters, all_peaks, missing_config, POINTS_PER_HOUR)

    if len(pairs) < num_to_hide: return drop_mask
    final_mask = _apply_masks(working_mask, pairs, num_to_hide, missing_config, POINTS_PER_HOUR)

    return final_mask