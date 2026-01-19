import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import curve_fit

class MISSINGNESS_PATTERN:
    def __init__(self, datapath_list, target_col='Value'):
        self.df = pd.concat([pd.read_csv(csv) for csv in datapath_list], axis=0, ignore_index=True)
        
        self.df['DT_Index'] = pd.to_datetime(self.df['DT_Index'])
        self.target_col = target_col
        self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors='coerce')
        
    def resampling(self):
        def enforce_5min(group):
            group = group.drop_duplicates(subset='DT_Index')
            group = group.set_index('DT_Index')
            group = group.sort_index()

            if group.empty: return group

            start_time = group.index.min().floor('D')
            end_time = group.index.max().ceil('D') - pd.Timedelta(minutes=5)
            full_grid = pd.date_range(start=start_time, end=end_time, freq='5min', name='DT_Index')
            
            return group.reindex(full_grid)

        df_regular = self.df.groupby('SID').apply(enforce_5min)        
        
        if 'SID' in df_regular.columns: df_regular = df_regular.drop(columns=['SID'])
        df_regular = df_regular.reset_index()
        if 'DT_Index' not in df_regular.columns and 'level_1' in df_regular.columns:
            df_regular = df_regular.rename(columns={'level_1': 'DT_Index'})
            
        return df_regular

    def filter_valid_days(self, df, threshold=0.50):
        """
        Removes 'Ghost Days' (low adherence) so they don't corrupt the gap analysis.
        Standard requirement: >70% data (approx 200/288 points).
        """
        expected_points = 288
        min_required = expected_points * threshold
        
        df_clean = df.copy()
        df_clean['Date'] = df_clean['DT_Index'].dt.date
        
        daily_counts = df_clean.groupby(['SID', 'Date'])[self.target_col].count()
        valid_days = daily_counts[daily_counts >= min_required].index
        
        df_clean = df_clean.set_index(['SID', 'Date'])
        df_filtered = df_clean.loc[df_clean.index.isin(valid_days)].reset_index()
        
        return df_filtered

    def analyze_gaps(self, rs_df):
        gap_data = []
        for sid, grp in rs_df.groupby('SID'):
            mask = grp[self.target_col].isna()
            blocks = (mask != mask.shift()).cumsum()
            
            gap_summary = grp[mask].groupby(blocks).agg(
                Start_Time=('DT_Index', 'first'),
                Count=('DT_Index', 'size')
            )
            
            gap_summary['Duration_Min'] = gap_summary['Count'] * 5
            gap_summary['SID'] = sid
            gap_summary['Hour_of_Day'] = gap_summary['Start_Time'].dt.hour
            gap_summary['Date'] = gap_summary['Start_Time'].dt.date
            
            gap_data.append(gap_summary)

        if gap_data:
            return pd.concat(gap_data).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def analyze_hourly_profile(self, gap_stats):
        """
        Calculates the 'When': Probability of a gap starting at Hour X.
        """
        counts = gap_stats['Hour_of_Day'].value_counts().sort_index()
        probs = counts / counts.sum()
        return probs.reindex(range(24), fill_value=0)


    def fit_global_distribution(self, gaps):
        """
        Fits the distribution in two parts:
        1. Exact probability of 'Single Drops' (5 min).
        2. Mixture Model (Exp + Gauss + Uniform Offset) for 'Real Gaps' (>5 min).
        """
        all_data = gaps['Duration_Min'].values
        
        n_total = len(all_data)
        n_single = np.sum(all_data <= 5)
        prob_single = n_single / n_total if n_total > 0 else 0
        tail_data = all_data[all_data > 5]
        
        if len(tail_data) == 0: return prob_single, None

        counts, bin_edges = np.histogram(tail_data, bins=47, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def mixture_func(x, A, k, B, mu, sigma, C):
            exp_part = A * np.exp(-k * x)
            gauss_part = B * np.exp(-((x - mu)**2) / (2 * sigma**2))
            return exp_part + gauss_part + C

        p0 = [np.max(counts), 0.05, np.max(counts)/10, 120, 20, 0.005]
        
        bounds = ([0, 0.001, 0, 60, 5, 0], [np.inf, 0.2, np.inf, 240, 60, 0.01])

        try:
            mix_params, _ = curve_fit(mixture_func, bin_centers, counts, 
                                    p0=p0, bounds=bounds, maxfev=10000)
            print("Tail Mixture Model (with Offset) Fitted Successfully.")
        except RuntimeError:
            print("Mixture fit failed.")
            mix_params = None

        return prob_single, mix_params


class RealisticMaskGenerator:
    def __init__(self, hourly_rate, prob_single, mix_params):
        self.hourly_probs = hourly_rate.values if hasattr(hourly_rate, 'values') else hourly_rate
        self.prob_single = prob_single
        self.use_mixture = (mix_params is not None)
        
        if self.use_mixture:
            self.A, self.k, self.B, self.mu, self.sigma, self.C = mix_params
            area_exp = (self.A / self.k) * np.exp(-self.k * 5)            
            area_gauss = self.B * self.sigma * np.sqrt(2 * np.pi)            
            area_unif = self.C * (240 - 5)
            
            total_area = area_exp + area_gauss + area_unif
            
            self.p_exp = area_exp / total_area
            self.p_gauss = area_gauss / total_area

    def sample_duration(self):
        """Decides 'How Long' using the Two-Stage logic."""
        if np.random.rand() < self.prob_single: return 5.0
        if self.use_mixture:
            r = np.random.rand()
            if r < self.p_exp:
                duration = np.random.exponential(scale=(1 / self.k)) + 5
            elif r < (self.p_exp + self.p_gauss):
                duration = np.random.normal(loc=self.mu, scale=self.sigma)                
            else:
                duration = np.random.uniform(10, 240)
            return max(10, min(duration, 1440))
        return 10.0

    def generate_mask(self, df_slice, points_per_hour=12):
        """Creates a boolean mask (0=Missing, 1=Observed)."""  
        df_slice = df_slice.copy()
        if 'cgm_simulated' not in df_slice.columns: df_slice['cgm_simulated'] = df_slice['cgm'].copy()
        dt_series = pd.to_datetime(df_slice['date'])
        hour_list = dt_series.dt.hour.unique()
        start_minute = dt_series.dt.minute.iloc[0]        
        total_points = len(df_slice)
        drop_mask = np.zeros(total_points, dtype=bool)
        
        for i, hour in enumerate(hour_list):
            if np.random.rand() < self.hourly_probs[hour]:
                duration_mins = self.sample_duration()
                duration_points = int(round(duration_mins / 5))
                
                if duration_points > 0:
                    low_limit = int(start_minute / 5) if i == 0 else 0                    
                    if low_limit >= points_per_hour: continue

                    start_offset = np.random.randint(low_limit, points_per_hour)
                    hour_indices = np.where(dt_series.dt.hour == hour)[0]
                    
                    if len(hour_indices) > 0:
                        base_idx = hour_indices[0]
                        abs_start = base_idx + start_offset
                        abs_end = min(abs_start + duration_points, total_points)
    
                        drop_mask[abs_start : abs_end] = True

        if 'cgm_simulated' in df_slice.columns: df_slice.loc[drop_mask, 'cgm_simulated'] = np.nan  
        return df_slice


def init_train_masking(csv_list, threshold):
    """ Init training real-world masking. """
    
    pipeline = MISSINGNESS_PATTERN(csv_list)
    df_resampled = pipeline.resampling()
    df_clean = pipeline.filter_valid_days(df_resampled, threshold=threshold)
    gaps = pipeline.analyze_gaps(df_clean)
    hourly_rate = pipeline.analyze_hourly_profile(gaps)
    
    prob_single, mix_p = pipeline.fit_global_distribution(gaps)
    
    gen = RealisticMaskGenerator(hourly_rate, prob_single, mix_p)
    return gen
