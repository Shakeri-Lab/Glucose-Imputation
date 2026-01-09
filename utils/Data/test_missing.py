import torch
import math
import numpy as np
import pandas as pd


def apply_mnar_values(drop_mask, day_df, max_points):
    """
    Simulates NMAR (Not Missing At Random) by masking values based on their 
    actual glucose level (High > 150 or Low < 70).
    """
    triggers = day_df[(day_df['cgm'] > 150) | (day_df['cgm'] < 70)].index.tolist()
    np.random.shuffle(triggers)

    T = len(day_df)
    remaining = max_points - drop_mask.sum()
    if remaining <= 0: return drop_mask

    while remaining > 0 and len(triggers) > 0:

        trigger_idx = triggers.pop()
        loc_idx = day_df.index.get_loc(trigger_idx)

        L = np.random.randint(1, remaining + 1)
        if loc_idx + L > T: continue
        if not (~drop_mask[loc_idx:loc_idx+L]).all(): continue

        drop_mask[loc_idx:loc_idx+L] = True
        remaining -= L

    return drop_mask


# def apply_mar_pisa(drop_mask, day_df, max_points):
#     """PISA simulation. """
#     is_sleep = (day_df['date'].dt.hour <= 7) | (day_df['date'].dt.hour >= 23)
#     T = len(day_df)

#     remaining = max_points - drop_mask.sum()
#     if remaining <= 0: return drop_mask

#     while remaining > 0:
#         L = np.random.randint(1, remaining + 1)
#         valid_starts = []
#         for i in np.where(is_sleep)[0]:
#             if i + L <= T:
#                 if is_sleep.iloc[i:i+L].all() and (~drop_mask[i:i+L]).all():
#                     valid_starts.append(i)

#         if len(valid_starts) == 0: break

#         pisa_start = np.random.choice(valid_starts)
#         drop_mask[pisa_start:pisa_start + L] = True
#         remaining -= L

#     return drop_mask


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
    if remaining <= 0:
        return drop_mask

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
    
    if not (active_mnar or active_meal or active_mcar):
        return df_day

    drop_mask_backup = drop_mask.copy()
    # Apply Logic
    if active_mnar:
        drop_mask = apply_mnar_values(drop_mask, df_day, max_points)
    if active_meal:
        drop_mask = apply_mar_meal(drop_mask, df_day, max_points)
    if active_mcar:
        drop_mask = apply_mcar_random_noise(drop_mask, max_points)

    if drop_mask.sum() != max_points: drop_mask = drop_mask_backup
    
    col_loc = df_day.columns.get_loc('cgm_simulated')
    df_day.iloc[drop_mask, col_loc] = np.nan
    return df_day

def simulate_missingness_test_pipeline(df, max_pct, probs):
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    df['cgm_simulated'] = df['cgm'].copy()
    
    def process_patient(patient_df):
        day_groups = patient_df.groupby(patient_df['date'].dt.date)
        results = []
        for date, day_data in day_groups:
            processed_day = process_single_day(day_data, max_pct, probs)
            results.append(processed_day)
        return pd.concat(results)

    if 'pid' in df.columns:
        final_df = df.groupby('pid').apply(process_patient).reset_index(drop=True)
    else:
        final_df = process_patient(df)
    return final_df
