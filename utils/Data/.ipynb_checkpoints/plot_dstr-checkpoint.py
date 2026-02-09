import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from train_missing import MISSINGNESS_PATTERN

DPI = 300
FIGSIZE = (10, 6)

def mkdir(p):
    os.makedirs(p, exist_ok=True)

def plot_cdf(x, reg, out):
    if len(x) == 0: return
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    p5 = np.mean(x == 5)

    plt.figure(figsize=FIGSIZE)
    plt.step(x, y, where='post')
    plt.axvline(5, ls=':', c='r', alpha=.6)
    plt.axhline(p5, ls=':', c='r', alpha=.6)
    plt.text(10, p5, f'{p5*100:.1f}% ≤ 5 min', c='r')
    plt.xlabel('Gap Duration (min)')
    plt.ylabel('CDF')
    plt.title(f'{reg.capitalize()} – Gap Duration CDF')
    plt.grid(alpha=.3)
    plt.xlim(0, 300)

    plt.savefig(f'{out}/cdf.png', dpi=DPI, bbox_inches='tight')
    plt.close()


def mix_pdf(x, p):
    A,k,B,mu,sig,C = p
    return A*np.exp(-k*(x-10)) + B*np.exp(-(x-mu)**2/(2*sig**2)) + C


def plot_tail(df, params, reg, out):
    x = df[(df.Duration_Min>=10)&(df.Duration_Min<=240)].Duration_Min.values
    if len(x) == 0: return

    bins = np.arange(10, 245, 5)
    plt.figure(figsize=FIGSIZE)
    h, b, _ = plt.hist(x, bins=bins, density=True, alpha=.7)

    if params is not None:
        xc = (b[:-1]+b[1:])/2
        plt.plot(xc, mix_pdf(xc, params), 'r', lw=2)
        mse = mean_squared_error(h, mix_pdf(xc, params))
        plt.text(.95,.8,f'MSE={mse:.1e}',transform=plt.gca().transAxes,ha='right')

    plt.xlabel('Gap Duration (min)')
    plt.ylabel('PDF')
    plt.title(f'{reg.capitalize()} – Tail Distribution')
    plt.grid(alpha=.3)

    plt.savefig(f'{out}/tail.png', dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_hourly(density_dicts, out):
    density = density_dicts['day'] + density_dicts['night']
    hours, vals = density.index, density.values
    colors = ['tab:blue' if h < 6 else 'tab:orange' for h in hours]

    plt.figure(figsize=FIGSIZE)
    plt.bar(hours, vals, color=colors)

    plt.xlabel('Hour')
    plt.ylabel('P(gap start)')
    plt.title('Hourly Missingness')

    plt.xticks(range(24))
    plt.grid(axis='y', alpha=.3)

    plt.bar(0, 0, color='tab:blue', label='Night (0–5)')
    plt.bar(0, 0, color='tab:orange', label='Day (6–23)')
    plt.legend()

    plt.savefig(f'{out}/hourly.png', dpi=DPI, bbox_inches='tight')
    plt.close()



def calc_hourly_prob(gaps, df_clean):
    M = df_clean[['SID','Date']].drop_duplicates().shape[0]
    h = (gaps[['SID','Date','Hour_of_Day']]
         .drop_duplicates()['Hour_of_Day']
         .value_counts()
         .sort_index() / M)

    h = h.reindex(range(24), fill_value=0).clip(0,1)
    return h 
    

def run(csvs, thr=.5):
    pipe = MISSINGNESS_PATTERN(csvs)
    df = pipe.filter_valid_days(pipe.resampling(), thr)
    gaps = pipe.analyze_gaps(df)
    hourly_prob_dict = {'day': None, 'night': None}
    
    for reg in ['night','day']:
        sub = gaps[(gaps.Hour_of_Day<6)] if reg=='night' else gaps[gaps.Hour_of_Day>=6]
        _, params = pipe.fit_regime_distribution(gaps, reg)

        out = f'real_mask_distribution/{reg}'
        mkdir(out)

        plot_cdf(sub.Duration_Min.values, reg, out)
        plot_tail(sub, params, reg, out)

        hourly_prob_dict[reg] = calc_hourly_prob(sub, df)
    plot_hourly(hourly_prob_dict, out)
    

if __name__ == '__main__':
    run([
        '/project/shakeri-lab/Amir/CGM_Imputation/RawData/dclp3_cgm_plus_features.csv',
        '/project/shakeri-lab/Amir/CGM_Imputation/RawData/dclp5_cgm_plus_features.csv'
    ])
