import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MISSINGNESS_PATTERN:
    def __init__(self, data_path, target_col='Value'):
        """
        Initializes the class by loading the data.
        """
        self.df = pd.read_csv(data_path)
        self.df['DT_Index'] = pd.to_datetime(self.df['DT_Index'])
        self.target_col = target_col
        self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors='coerce')

    def resampling(self):
        """ 
        Enforces a strict 5-minute grid for each Subject (SID).
        """
        def enforce_5min(group):
            group = group.drop_duplicates(subset='DT_Index')
            group = group.set_index('DT_Index')
            return group.resample('5min').asfreq()

        df_regular = self.df.groupby('SID').apply(enforce_5min)        
        df_regular = df_regular.drop(columns=['SID'], errors='ignore').reset_index()
        return df_regular

    def analyze_gaps(self, rs_df):
        """
        Detects every missing data episode (gap).
        """
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
            
            gap_data.append(gap_summary)

        if gap_data:
            return pd.concat(gap_data).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def plot_patterns(self, gap_stats, max_gap_threshold=240):
        """
        Plots the distribution with filtering to make it readable.
        """
        if gap_stats.empty:
            print("No gaps found to plot.")
            return

        routine_gaps = gap_stats[gap_stats['Duration_Min'] <= max_gap_threshold]
        long_gaps = gap_stats[gap_stats['Duration_Min'] > max_gap_threshold]
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        sns.histplot(data=routine_gaps, x='Duration_Min', bins=48, kde=True, ax=ax[0], color='teal')
        ax[0].set_title(f'Distribution of Gaps (Under {max_gap_threshold} Minutes)')
        ax[0].set_xlabel('Length of Gap (Minutes)')
        ax[0].set_ylabel('Frequency')

        sns.boxplot(
            data=routine_gaps, 
            x='Hour_of_Day', 
            y='Duration_Min', 
            color='lightseagreen', 
            ax=ax[1], 
            showfliers=False # Hides extreme outliers to keep the view clean
        )
        
        ax[1].set_title('Time of Missingness vs. Length (Routine Gaps)')
        ax[1].set_xlabel('Hour of Day (0-23)')
        ax[1].set_ylabel('Length of Gap (Minutes)')
        ax[1].set_xticks(range(0, 25, 2))
        ax[1].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

# --- HOW TO RUN IT ---
pipeline = MISSINGNESS_PATTERN('RawData/dclp3_cgm_plus_features.csv')

df_clean = pipeline.resampling()
gaps = pipeline.analyze_gaps(df_clean)
pipeline.plot_patterns(gaps)