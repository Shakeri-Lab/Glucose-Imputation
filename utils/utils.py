import numpy as np
import torch.nn as nn
import os, torch, argparse, logging, random, yaml, argparse, json
from .Data.cgm_dataset import CGMDataset
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

def parse_args_from_yml(config_path):
    """ Read .yml file. """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


def set_seed(seed=42):
    random.seed(seed)                      # Python RNG
    np.random.seed(seed)                   # NumPy RNG
    torch.manual_seed(seed)                # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)           # PyTorch single-GPU RNG
    torch.cuda.manual_seed_all(seed)       # PyTorch multi-GPU RNG

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: for full determinism in certain PyTorch versions
    os.environ["PYTHONHASHSEED"] = str(seed)



def load_data(args):
    """ Load dataset. """
    logging.info(f"Loading dataset from {args.data_path}")    
    all_dataset_dict = {'org': {}, 'missed': {}}    
    for splt in ['train', 'val', 'test']:
        missing_enabled = True if (args.is_pedap and not args.missing_enabled and splt in ['val', 'test']) else args.missing_enabled
        raw_dataset = CGMDataset(data_path=os.path.join(args.data_path, splt + '.csv'), seq_len=args.seq_len, stride=args.stride, missing_enabled=missing_enabled, miss_cfg=args.miss_config, is_pedap=args.is_pedap)
        
        exg_intrv = raw_dataset[:, :, 1:4]
        time_embed = raw_dataset[:, :, 4:6]

        for key in ['org', 'missed']:
            if key == 'missed':
                raw_signal = raw_dataset[:, :, 0]
            else:
                raw_signal = raw_dataset[:, :, -1]

            signal = raw_signal.unsqueeze(-1)
            processed_data = torch.cat([signal, exg_intrv, time_embed], dim=2)            
            all_dataset_dict[key][splt] = processed_data

    return all_dataset_dict


def skew_norm(x, skew, min_val, max_val):
    """Normalize the input x to the range [0, 1] using a skewed function."""
    x = np.array(x) # Ensure input is numpy array
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    x = np.maximum(np.minimum(x, max_val), min_val)
    return np.power((x - min_val) / (max_val - min_val), skew)

def unskew_norm(x, skew, min_val, max_val):
    """ Unnormalize the input x from the range [0, 1]. """
    x = np.array(x)
    if x.ndim == 1:
        x = np.expand_dims(x, 0)        
    x = np.maximum(np.minimum(x, 1.0), 0.0)
    return min_val + (max_val - min_val) * np.power(x, 1.0 / skew)

def visualize_imputation(truth, truth_org, imp, mask, s_idx=0, f_idx=0, title="Imp_Quality", plot_dir='./', max_plot=32):
    os.makedirs(os.path.join(plot_dir, "plots"), exist_ok=True)
    
    for i in range(len(truth)):
        if i > max_plot: break
        m = mask[i].astype(bool)
        x = np.arange(len(truth[i]))

        fig, ax = plt.subplots(figsize=(12, 5), dpi=120, layout='constrained')

        ax.plot(x, truth_org[i], color='green', alpha=0.2, lw=3, label="Truth (Full)", zorder=1)
        ax.plot(x, np.where(~m, truth[i], np.nan), 'k-', lw=2, label="Observed", zorder=2)
        ax.plot(x, np.where(m, truth_org[i], np.nan), color='green', linestyle='-', 
                marker='.', markersize=6, alpha=0.7, label="Truth (Hidden)", zorder=3)

        ax.plot(x, np.where(m, imp[i], np.nan), color='red', linestyle='--', 
                marker='x', markersize=6, label="Imputed", zorder=4)

        ax.set_title(f"{title} (S{s_idx}, F{f_idx})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(plot_dir, f"{title}_{s_idx}_{f_idx}_i{i}.png"))
        plt.close()



def eval_metrics(tr_ts, pre_ts, mask):
    """
    Evaluates imputation performance using both statistical error metrics 
    and clinical metrics defined in the paper.
    """
    mask_bool = mask.astype(bool)
    y_true_flat = tr_ts[mask_bool]
    y_pred_flat = pre_ts[mask_bool]

    if len(y_true_flat) == 0:
        return {k: 0.0 for k in ["MSE", "MAE", "RMSE", "Bias", "emp_SE", "mard", "PCR", "hypo_sens"]}

    err = y_pred_flat - y_true_flat
    
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(mse)
    bias = np.mean(err)
    emp_SE = np.std(err, ddof=1) # Empirical Standard Error
    
    mard = np.mean(np.abs(err / (y_true_flat + 1e-6))) * 100

    dtw_score = calc_dtw(tr_ts, pre_ts, mask)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Bias": float(bias),
        "emp_SE": float(emp_SE),
        "mard": float(mard),
        'dtw': float(dtw_score),
    }

def calc_dtw(ground_truth, imputation, mask):
    """ DTW to measure shape. """
    distances = []
    for i, (gr_ts, pr_ts) in enumerate(zip(ground_truth, imputation)):
        flag_tuples = tuple_flags(mask[i])
        for tpl in flag_tuples:
            gr_slice = gr_ts[tpl[0]:tpl[1] + 1]
            pr_slice = pr_ts[tpl[0]:tpl[1] + 1]
            distances.append(dtw(gr_slice, pr_slice))
    return np.mean(np.array(distances))

def tuple_flags(flgs):
    """Give list of contiguous True-index intervals."""
    indices = np.where(flgs)[0]
    if len(indices) == 0: return []

    indices_tuples = []
    strt = indices[0]
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > 1:
            indices_tuples.append([strt, indices[i]])
            strt = indices[i + 1]

    indices_tuples.append([strt, indices[-1]])
    return indices_tuples
