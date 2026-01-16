import numpy as np
import pandas as pd

def apply_protocol_A_homeostatic(drop_mask, day_df, missing_config, POINTS_PER_HOUR=12):
    """
    Protocol A: Steady-State Mask.
    Fills stable areas with RANDOM window lengths (within a range)
    until the Total Target is met.
    """
    T = len(day_df)
    cgm_values, meal_values = day_df['cgm'].values, day_df['meal'].values
    
    MIN_WINDOW_LEN, MAX_WINDOW_LEN = int(1.0 * POINTS_PER_HOUR), int(2.0 * POINTS_PER_HOUR) 
    min_total_points, max_total_points = int(T * missing_config['min']), int(T * missing_config['max'])
    target_total_points = np.random.randint(min_total_points, max_total_points + 1)

    valid_starts = []
    gradients = np.abs(np.gradient(cgm_values))
    is_stable = gradients < 2.0 
    
    for t in range(T - MAX_WINDOW_LEN):
        window_meal = meal_values[t : t + MAX_WINDOW_LEN]
        window_stable = is_stable[t : t + MAX_WINDOW_LEN]
        if (np.sum(window_meal) == 0) and (np.mean(window_stable) > 0.8): valid_starts.append(t)
            
    if not valid_starts: return drop_mask 
    points_added = 0
    np.random.shuffle(valid_starts)
    for start_idx in valid_starts:
        if points_added >= target_total_points: break
        current_window_len = np.random.randint(MIN_WINDOW_LEN, MAX_WINDOW_LEN + 1)        
        end_idx = start_idx + current_window_len
        
        if not drop_mask[start_idx : end_idx].any():
            if (points_added + current_window_len) <= target_total_points:
                drop_mask[start_idx : end_idx] = True
                points_added += current_window_len
                
    return drop_mask

def apply_protocol_B_hidden_peak(drop_mask, day_df, missing_config, POINTS_PER_HOUR=12):
    """
    Protocol B: Hidden Peak Mask.
    Masks post-prandial peaks with RANDOM window lengths centered on the peak.
    """
    T, cgm_values = len(day_df), day_df['cgm'].values
    
    MIN_WINDOW_LEN, MAX_WINDOW_LEN = int(2.5 * POINTS_PER_HOUR), int(4.0 * POINTS_PER_HOUR) 
    min_total_points, max_total_points = int(T * missing_config['min']), int(T * missing_config['max'])
    target_total_points = np.random.randint(min_total_points, max_total_points + 1)

    peak_candidates = []
    for meal_idx in np.where(day_df['meal'] > 0)[0]:
        search_end = min(meal_idx + 24, T)
        if search_end > meal_idx: peak_candidates.append(meal_idx + np.argmax(cgm_values[meal_idx : search_end]))

    if not peak_candidates: return drop_mask
    points_added = 0
    np.random.shuffle(peak_candidates)
    for peak_idx in peak_candidates:
        if points_added >= target_total_points: break

        current_window_len = np.random.randint(MIN_WINDOW_LEN, MAX_WINDOW_LEN + 1)
        start_idx = max(0, peak_idx - (current_window_len // 2))
        end_idx = min(T, start_idx + current_window_len)
        actual_len = end_idx - start_idx

        if not drop_mask[start_idx : end_idx].any():
            if (points_added + actual_len) <= target_total_points:
                drop_mask[start_idx : end_idx] = True
                points_added += actual_len

    return drop_mask

def process_single_day_experiment(day_df, experiment_mode='A'):
    """
    Master function to apply specific experimental protocols.
    """
    df_day = day_df.copy()
    total_points = len(df_day)
    drop_mask = np.zeros(total_points, dtype=bool)
    
    config_A = {'min': 0.10, 'max': 0.15}
    config_B = {'min': 0.20, 'max': 0.30}
    
    if experiment_mode == 'A':
        drop_mask = apply_protocol_A_homeostatic(drop_mask, df_day, config_A)
    elif experiment_mode == 'B':
        drop_mask = apply_protocol_B_hidden_peak(drop_mask, df_day, config_B)
    elif experiment_mode == 'Mixed':
        drop_mask = apply_protocol_A_homeostatic(drop_mask, df_day, config_A)
        drop_mask = apply_protocol_B_hidden_peak(drop_mask, df_day, config_B)

    if 'cgm_simulated' in df_day.columns:
        col_loc = df_day.columns.get_loc('cgm_simulated')
        df_day.iloc[drop_mask, col_loc] = np.nan
    
    return df_day

def simulate_experiment_pipeline(df, mode='Mixed'):
    df = df.copy()
    if 'cgm_simulated' not in df.columns:
        df['cgm_simulated'] = df['cgm'].copy()
    return process_single_day_experiment(df, experiment_mode=mode)