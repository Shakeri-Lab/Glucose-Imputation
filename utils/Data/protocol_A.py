import numpy as np
import pandas as pd

def apply_protocol_A_homestatic_engine(drop_mask, day_df, missing_config, BASE_WINDOW_LEN=6, LOOKBACK=12):
    """
    Protocol A: Steady-State Mask.
    Fills stable areas with fixed window lengths, avoiding excursion periods.
    """
    temp_mask = drop_mask.copy() 
    T = len(day_df)
    
    cgm_values = day_df['cgm'].values
    meal_values = day_df['meal'].values
    bolus_values = day_df['bolus'].values
    
    target_total_points = np.random.randint(
        int(T * missing_config['min']), 
        int(T * missing_config['max']) + 1
    )
    
    gradients = np.abs(np.gradient(cgm_values))
    is_stable = gradients < 3.0
    
    valid_starts = _find_valid_windows(
        cgm_values, meal_values, bolus_values, is_stable,
        T, BASE_WINDOW_LEN, LOOKBACK
    )
    
    if len(valid_starts) * BASE_WINDOW_LEN < target_total_points:
        return drop_mask
    
    temp_mask = _apply_mask_windows(
        temp_mask, valid_starts, target_total_points, BASE_WINDOW_LEN
    )
    
    points_added = np.sum(temp_mask) - np.sum(drop_mask)
    if points_added < target_total_points:
        return drop_mask
    
    return temp_mask


def _find_valid_windows(cgm_values, meal_values, bolus_values, is_stable, 
                        T, BASE_WINDOW_LEN, LOOKBACK):
    """
    Identifies all valid steady-state windows that meet homeostatic criteria.
    """
    valid_starts = []
    
    for t in range(LOOKBACK, T - BASE_WINDOW_LEN + 1):
        lookback_meal = meal_values[t - LOOKBACK : t]
        lookback_bolus = bolus_values[t - LOOKBACK : t]
        
        window_meal = meal_values[t : t + BASE_WINDOW_LEN]
        window_bolus = bolus_values[t : t + BASE_WINDOW_LEN]
        window_cgm = cgm_values[t : t + BASE_WINDOW_LEN]
        window_stable = is_stable[t : t + BASE_WINDOW_LEN]
        
        no_recent_events = (np.sum(lookback_meal) == 0) and (np.sum(lookback_bolus) == 0)
        no_current_events = (np.sum(window_meal) == 0) and (np.sum(window_bolus) == 0)
        stable_gradient = np.mean(window_stable) >= 0.85
        cgm_range = np.ptp(window_cgm) < 25
        cgm_in_normal_range = (window_cgm >= 70).all() and (window_cgm <= 140).all()
        
        if no_recent_events and no_current_events and stable_gradient and cgm_range and cgm_in_normal_range:
            valid_starts.append(t)
    
    return valid_starts


def _apply_mask_windows(temp_mask, valid_starts, target_total_points, BASE_WINDOW_LEN):
    """
    Applies missing data windows to the mask array from valid candidate positions.
    """
    np.random.shuffle(valid_starts)
    points_added = 0
    
    for start_idx in valid_starts:
        if points_added >= target_total_points:
            break
        
        points_remaining = target_total_points - points_added
        current_len = min(BASE_WINDOW_LEN, points_remaining)
        end_idx = start_idx + current_len
        
        if not temp_mask[start_idx : end_idx].any():
            temp_mask[start_idx : end_idx] = True
            points_added += current_len
    
    return temp_mask