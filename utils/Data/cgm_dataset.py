import torch, math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from .missing import simulate_missingness_pipeline
from .missing import simulate_experiment_pipeline

class CGMDataset(Dataset):
    def __init__(self, data_path, seq_len=288, stride=12, 
                 missing_enabled=True, miss_cfg=None, is_pedap=False):
        self.seq_len = seq_len
        self.stride = stride
        self.missing_enabled = missing_enabled
        self.miss_cfg = miss_cfg or {}
        
        self.df = pd.read_csv(data_path)

        if is_pedap:
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d %H:%M:%S')
        
        if not self.missing_enabled: self.df['cgm_simulated'] = self.df['cgm'].copy() 

        self.samples = self._build_samples_dclp3() if is_pedap else self._build_samples()

    def _absolute_time_encoding(self, indices, T_day=288):
        t = indices.float() / float(T_day)
        return torch.stack([torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1)

    def _skew_norm(self, x, skew=1, min_val=40, max_val=400):
        x = np.array(x)
        x = np.maximum(np.minimum(x, max_val), min_val)
        return np.power((x - min_val) / (max_val - min_val), skew)

    def _build_samples(self):
        samples = []
        for pid, group in self.df.groupby('pid'):
            samples.extend(self._generate_samples(group))
        return np.array(samples)
        
    def _build_samples_dclp3(self):
        samples = []
        for pid, group in self.df.groupby('pat_id'):
            for seq_id, episode in group.groupby('seq_id'):
                samples.extend(self._generate_samples(episode))
        return np.array(samples)
        
    def _generate_samples(self, group):
        samples = []
        num_points = len(group)
        for i in range(0, num_points - self.seq_len + 1, self.stride):

            if self.missing_enabled:
                slice_df = simulate_experiment_pipeline(
                    group.iloc[i : i + self.seq_len],
                    self.miss_cfg
                )
            else:
                slice_df = group.iloc[i : i + self.seq_len]

            if slice_df is None: continue
                
            slice_in = slice_df['cgm_simulated'].values
            slice_gt = slice_df['cgm'].values
            slice_meal = np.where(slice_df['meal'].values > 0, 1, 0)

            if np.isnan(slice_gt).any(): continue
                
            dates = pd.to_datetime(slice_df['date'])
            time_indices = (dates.dt.hour * 60 + dates.dt.minute) // 5
            time_indices = torch.tensor(time_indices.values)
            time_embeds = self._absolute_time_encoding(time_indices).numpy()
    
            norm_in = self._skew_norm(slice_in) 
            norm_gt = self._skew_norm(slice_gt)
            
            sample = np.column_stack([norm_in, slice_meal, time_embeds, norm_gt])
            samples.append(sample)

        return samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx])

