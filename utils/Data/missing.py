import numpy as np
import pandas as pd
from .protocol_B import apply_protocol_B_hidden_peak_engine
from .protocol_A import apply_protocol_A_homestatic_engine


def apply_protocol_A_homeostatic(drop_mask, day_df, missing_config, BASE_WINDOW_LEN=6, LOOKBACK=12):
    """
    Protocol A: Steady-State Mask.
    Fills stable areas with fixed window lengths, avoiding excursion periods.
    """
    drop_mask = apply_protocol_A_homestatic_engine(drop_mask, day_df, missing_config, BASE_WINDOW_LEN=6, LOOKBACK=12)
    return drop_mask

def apply_protocol_B_hidden_peak(drop_mask, day_df, missing_config, POINTS_PER_HOUR=12):
    """
    Protocol B: Hidden Peak Mask.
    Masks post-prandial peaks centered on the peak location.
    """
    drop_mask = apply_protocol_B_hidden_peak_engine(drop_mask, day_df, missing_config)
    return drop_mask


def apply_protocol_C_TCR(drop_mask, day_df):
    """
    Protocol C: Masks 1-hour window around hypoglycemia during TCR activations.
    """
    cgm = day_df['cgm'].values
    tcr_flag = day_df['tcr_flag'].values.astype('bool')
    
    low_during_tcr = tcr_flag & (cgm < 70)
    
    if low_during_tcr.any():
        low_indices = np.where(low_during_tcr)[0]
        window_size = 6 
        
        for idx in low_indices:
            start = max(0, idx - window_size)
            end = min(len(cgm), idx + window_size + 1)
            drop_mask[start:end] = True
    
    return drop_mask


def process_single_day_experiment(day_df, miss_config):
    """
    Master function to apply specific experimental protocols.
    """
    experiment_mode, protocol_mask_ratio =  miss_config['type'], miss_config['protocol_mask_ratio']
    df_day = day_df.copy()
    total_points = len(df_day)
    drop_mask = np.zeros(total_points, dtype=bool)
    
    config_A = {'min': protocol_mask_ratio, 'max': protocol_mask_ratio}
    
    if experiment_mode == 'A':
        drop_mask = apply_protocol_A_homeostatic(drop_mask, df_day, config_A)
    elif experiment_mode == 'B':
        drop_mask = apply_protocol_B_hidden_peak(drop_mask, df_day, miss_config)
    elif experiment_mode == 'C':
        drop_mask = apply_protocol_C_TCR(drop_mask, df_day)
    elif experiment_mode == 'Mixed':
        choice_prob = np.random.rand()
        if choice_prob < 0.2:
            drop_mask = apply_protocol_A_homeostatic(drop_mask, df_day, config_A)
        elif choice_prob < 0.5:
            drop_mask = apply_protocol_B_hidden_peak(drop_mask, df_day, miss_config)
        else:
            drop_mask = apply_protocol_A_homeostatic(drop_mask, df_day, config_A)
            drop_mask = apply_protocol_B_hidden_peak(drop_mask, df_day, miss_config)
            
    if 'cgm_simulated' in df_day.columns:
        col_loc = df_day.columns.get_loc('cgm_simulated')
        df_day.iloc[drop_mask, col_loc] = np.nan
    
    return df_day

def simulate_experiment_pipeline(df, miss_config):
    df = df.copy()
    if 'cgm_simulated' not in df.columns:
        df['cgm_simulated'] = df['cgm'].copy()
    return process_single_day_experiment(df, miss_config)