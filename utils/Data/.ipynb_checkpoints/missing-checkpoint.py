import torch
import math
import numpy as np
import pandas as pd

def apply_mnar_values(drop_mask, day_df, max_points):
    """
    Simulates MNAR by masking ONLY contiguous Hyper/Hypo values.
    """
    cgm_values = day_df['cgm'].values
    is_extreme = (cgm_values > 150) | (cgm_values < 70)
    
    triggers = np.where(is_extreme)[0].tolist()
    np.random.shuffle(triggers)

    T = len(day_df)
    remaining = max_points - drop_mask.sum()
    if remaining <= 0: return drop_mask

    while remaining > 0 and len(triggers) > 0:
        loc_idx = triggers.pop()       
        
        if drop_mask[loc_idx]: continue
        L_max = np.random.randint(1, remaining + 1)        
        points_masked_in_this_loop = 0
        
        for i in range(L_max):
            curr = loc_idx + i
            
            if curr >= T: break
            if not is_extreme[curr]: break
            if drop_mask[curr]: break

            drop_mask[curr] = True
            points_masked_in_this_loop += 1
        
        remaining -= points_masked_in_this_loop

    return drop_mask

def apply_mar_meal(drop_mask, day_df, max_points):
    """ 
    Meal simulation: Drops data segments starting at meal times.
    """
    triggers = np.where(day_df['meal'] == 1)[0]
    np.random.shuffle(triggers)    
    triggers = list(triggers) 

    T = len(day_df)
    remaining = max_points - drop_mask.sum()
    if remaining <= 0: return drop_mask

    while remaining > 0 and len(triggers) > 0:
        start_idx = triggers.pop()
        L = np.random.randint(1, remaining + 1)
        
        if start_idx + L > T: continue
        if drop_mask[start_idx : start_idx + L].any(): continue
        drop_mask[start_idx : start_idx + L] = True
        remaining -= L

    return drop_mask


def apply_mcar_random_noise(drop_mask, max_points):
    """
    MCAR with exact total missing = max_points
    """
    current = drop_mask.sum()
    remaining = max_points - current
    if remaining <= 0: return drop_mask

    safe_pos = np.where(~drop_mask)[0]
    remaining = min(remaining, len(safe_pos))

    if remaining > 0:
        mcar_pos = np.random.choice(safe_pos, size=remaining, replace=False)
        drop_mask[mcar_pos] = True

    return drop_mask


def process_single_day(day_df, max_pct, probs):
    df_day = day_df.copy()
    total_points = len(df_day)
    drop_mask = np.zeros(total_points, dtype=bool)
    
    max_points = int(total_points * max_pct)
    
    active_mnar     = np.random.rand() < probs.get('mnar', 0.5)
    active_meal     = np.random.rand() < probs.get('meal', 0.3)
    active_mcar     = np.random.rand() < probs.get('mcar', 0.8)
    
    if not (active_mnar or active_meal or active_mcar): return df_day

    # Apply Logic
    if active_mnar:
        drop_mask = apply_mnar_values(drop_mask, df_day, max_points)
    if active_meal:
        drop_mask = apply_mar_meal(drop_mask, df_day, max_points)
    if active_mcar:
        drop_mask = apply_mcar_random_noise(drop_mask, max_points)

    if drop_mask.sum() != max_points: return None
    
    col_loc = df_day.columns.get_loc('cgm_simulated')
    df_day.iloc[drop_mask, col_loc] = np.nan
    return df_day


def simulate_missingness_pipeline(df, max_pct, probs):
    df = df.copy()
    df['cgm_simulated'] = df['cgm'].copy()    
    return process_single_day(df, max_pct, probs)
