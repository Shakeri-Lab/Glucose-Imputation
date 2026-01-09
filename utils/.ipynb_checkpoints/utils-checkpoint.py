import numpy as np
import torch.nn as nn
import os, torch, argparse, logging, random, yaml, argparse, json
from .Data.cgm_dataset import CGMDataset
import matplotlib.pyplot as plt

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

# def load_data(args):
#     """ Load dataset. """
#     logging.info(f"Loading dataset from {args.data_path}")
#     all_dataset_dict = {'org': None, 'missed': None}
#     for keys in all_dataset_dict.keys():       
#         data_dict = {}
#         dataset = CGMDataset(data_path=args.data_path, seq_len=args.seq_len, stride=args.stride, 
#                              missing_enabled=(keys == 'missed'), miss_cfg=args.miss_config, is_evaluate=args.is_evaluate)
#         num_samples = len(dataset)
#         for splt in ['train', 'val', 'test']:
#             if splt == 'train':
#                 start, end = 0, int(num_samples * 0.8)
#                 sub_dataset = dataset[start:end]
#             elif splt == 'val':
#                 start, end = int(num_samples * 0.8), int(num_samples * 0.9)
#                 sub_dataset = dataset[start:end]
#             else:
#                 start = int(num_samples * 0.9)
#                 sub_dataset = dataset[start:]

#             signal = sub_dataset[:, :, 0] if keys == 'missed' else sub_dataset[:, :, 3]                
#             signal, time_embed = signal.unsqueeze(-1), sub_dataset[:, :, 1:3] 
            
#             sub_dataset = torch.cat([signal, time_embed], dim=2)
#             data_dict[splt] = sub_dataset
            
#         all_dataset_dict[keys] = data_dict

#     return all_dataset_dict




def load_data(args):
    """ Load dataset. """
    logging.info(f"Loading dataset from {args.data_path}")
    all_dataset_dict = {'org': None, 'missed': None}
    
    for keys in all_dataset_dict.keys():       
        data_dict = {}
        for splt in ['train', 'val', 'test']:
            sub_dataset = CGMDataset(data_path=os.path.join(args.data_path, splt + '.csv'), seq_len=args.seq_len, stride=args.stride, missing_enabled=True, miss_cfg=args.miss_config, is_dclp3=args.is_dclp3)
            
            signal = sub_dataset[:, :, 0] if keys == 'missed' else sub_dataset[:, :, -1]                
            signal, meal, time_embed = signal.unsqueeze(-1), sub_dataset[:, :, 1].unsqueeze(-1), sub_dataset[:, :, 2:4] 
            
            sub_dataset = torch.cat([signal, meal, time_embed], dim=2)
            data_dict[splt] = sub_dataset

        all_dataset_dict[keys] = data_dict

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
        
        plt.savefig(os.path.join(plot_dir, f"plots/{title}_{s_idx}_{f_idx}_i{i}.png"))
        plt.close()


def eval_metrics(tr_ts, pre_ts):
    """ Eval metrics. """
    err = pre_ts - tr_ts
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(mse)
    bias = np.mean(err)
    emp_SE = np.std(err, ddof=1)  # empirical standard error (sample std)
    mard = np.mean(np.abs((pre_ts - tr_ts) / tr_ts)) * 100
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Bias": float(bias),
        "emp_SE": float(emp_SE),
        "mard": float(mard)
    }
    