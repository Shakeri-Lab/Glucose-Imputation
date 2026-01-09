import torch
import pandas as pd
import numpy as np
import os

class RiskLookup:
    def __init__(self, npy_path="risk_matrix_seg.npy", device='cpu'):
        self.device = device
        self.table = torch.tensor(np.load(npy_path), device=device, dtype=torch.float32)
        
        self.max_ref = self.table.shape[0] - 1
        self.max_bgm = self.table.shape[1] - 1

    def get_risk(self, ref_tensor, bgm_tensor):
        if ref_tensor.shape != bgm_tensor.shape:
            raise ValueError(f"Shape mismatch: REF {ref_tensor.shape} vs BGM {bgm_tensor.shape}")

        ref_tensor = ref_tensor.to(self.device, dtype=torch.float32)
        bgm_tensor = bgm_tensor.to(self.device, dtype=torch.float32)

        ref_idx = torch.clamp(torch.round(ref_tensor), 0, self.max_ref).long()
        bgm_idx = torch.clamp(torch.round(bgm_tensor), 0, self.max_bgm).long()

        return self.table[ref_idx, bgm_idx]

    def count_zones(self, risk_tensor):
        """
        Calculate Zone percentage efficiently.
        """
        total = risk_tensor.numel()
        if total == 0: return [0]*5

        count_a = (risk_tensor < 0.5).sum().item()
        count_b = ((risk_tensor >= 0.5) & (risk_tensor < 1.5)).sum().item()
        count_c = ((risk_tensor >= 1.5) & (risk_tensor < 2.5)).sum().item()
        count_d = ((risk_tensor >= 2.5) & (risk_tensor < 3.5)).sum().item()
        count_e = (risk_tensor >= 3.5).sum().item()

        return [c / total for c in [count_a, count_b, count_c, count_d, count_e]]


def risk_calc(true_ts, pred_ts, result_dir, risk_dir):
    """ Calc Risk. """
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    risk_model = RiskLookup(risk_dir, device=device) 
    
    for (tr_ts, pr_ts) in zip(true_ts, pred_ts):
        tr_ts, pr_ts = torch.tensor(tr_ts), torch.tensor(pr_ts)
        risk_map = risk_model.get_risk(tr_ts, pr_ts)    
        mean_risk = risk_map.mean().item()
        zone_list = risk_model.count_zones(risk_map)    
        results.append({
            "risk": mean_risk, 
            "A": zone_list[0],
            "B": zone_list[1],
            "C": zone_list[2],
            "D": zone_list[3],
            "E": zone_list[4],
        })

    
    df = pd.DataFrame(results)
    averages = pd.DataFrame([df.mean(numeric_only=True)])    
    averages.to_csv(result_dir, index=False)