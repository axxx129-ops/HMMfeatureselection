"""
experiments.py

Empirical study of feature selection for HMM regime detection.
Multi-stock, multi-experiment, CSV-output version.

For each of 24 stocks (8 large cap, 8 mid cap, 8 small/distressed cap):
    Exp 1: 6 univariate HMMs (one per candidate feature)
    Exp 2 vol stacking : 1D -> 2D -> 3D over the volatility-group features
    Exp 2 bb  stacking : 1D -> 2D -> 3D over the trend-group features
    Order experiment   : all 6 permutations of the 3D vol-group input
                         AND all 6 permutations of the 3D bb-group input,
                         with the full 6x6 agreement matrix per group

CSV outputs (placed in same directory as this script):
    exp1_results.csv      one row per (ticker, feature, state)
    exp2_stacking.csv     one row per (ticker, group, step, state, feature)
    order_experiment.csv  one row per (ticker, group, perm_i, perm_j)
    summary.csv           one row per ticker, cross-cut metrics

Plots (PNG, only for tickers in PLOT_TICKERS):
    {ticker}_exp1.png             6 stacked univariate panels
    {ticker}_exp2_vol.png         1D -> 2D -> 3D vol stacking
    {ticker}_exp2_bb.png          1D -> 2D -> 3D bb  stacking
    {ticker}_order_bb.png         3 random bb-group permutations stacked

Model: 2-state Gaussian HMM via hmmlearn (full covariance, 5 random restarts).
"""

import os
import csv
import warnings
import logging
from itertools import permutations

import numpy as np
try:
    import yfinance as yf  # no longer used at runtime (data comes from snapshot)
except ImportError:
    yf = None              # fine — reproduction reads frozen CSVs, not the network
from hmmlearn import hmm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

plt.rcParams.update({
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'font.size':        16,
    'axes.titlesize':   17,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'axes.linewidth':   1.5,
    'axes.titleweight': 'bold',
    'savefig.bbox':     'tight',
})

COLOR_STATE_0 = '#3B6BAA'
COLOR_STATE_1 = '#B23A3C'
STRIPE_ALPHA  = 0.35


# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

STOCKS_BY_TIER = {
    # ── Large cap (20) ────────────────────────────────────────────────────
    # Covers all major GICS sectors with blue-chip names that have
    # continuous price history from 2005 to 2024.
    #
    #   IT             : AAPL  MSFT  IBM   GOOGL INTC  CSCO
    #   Financials     : JPM   BAC
    #   Energy         : XOM   CVX
    #   Cons Staples   : WMT   KO    PG
    #   Cons Disc      : AMZN  HD
    #   Industrials    : GE    CAT
    #   Health Care    : JNJ
    #   Comm Services  : VZ
    #   Utilities      : NEE
    'large': [
        'AAPL', 'MSFT', 'JPM',  'XOM',  'WMT',  'KO',   'IBM',  'GE',
        'GOOGL','AMZN', 'BAC',  'PG',   'CVX',  'HD',   'INTC', 'CSCO',
        'JNJ',  'VZ',   'NEE',  'CAT',
    ],

    # ── Mid cap (20) ──────────────────────────────────────────────────────
    # Primarily Consumer Discretionary (retail / apparel) with additions
    # in Health Care, Energy, and Materials to broaden sector coverage.
    #
    #   Cons Disc      : DECK  SIG   MAT   HOG   ANF   RRGB  SKX   GT
    #                    GPS   M     JWN   KSS   AEO   RL    FL    BBY  URBN
    #   Health Care    : XRAY
    #   Energy         : EQT
    #   Materials/Ind  : SON
    'mid':   [
        'DECK', 'SIG',  'MAT',  'HOG',  'ANF',  'RRGB', 'SKX',  'GT',
        'GPS',  'M',    'JWN',  'KSS',  'AEO',  'RL',   'FL',   'BBY',
        'URBN', 'XRAY', 'EQT',  'SON',
    ],

    # ── Small / distressed cap (20) ───────────────────────────────────────
    # Volatile, trend-absent, or structurally declining names that stress-test
    # Observation 2 (trend-group regime conditionality). Sector additions
    # in Health Care, Financials, and Materials fill coverage gaps.
    #
    #   Comm Equipment : BB    NOK
    #   IT / Consumer  : GRMN  IMAX  SIRI  XRX   NTGR  HIMX  VOXX
    #   Energy         : PLUG  RIG
    #   Cons Disc      : GME   BIG   DDS   BGFV  SCVL  CATO
    #   Health Care    : HCSG
    #   Financials     : NBTB
    #   Materials      : MTRN
    'small': [
        'BB',   'GRMN', 'PLUG', 'RIG',  'IMAX', 'NOK',  'SIRI', 'XRX',
        'GME',  'BIG',  'DDS',  'NTGR', 'HIMX', 'VOXX', 'BGFV', 'SCVL',
        'CATO', 'HCSG', 'NBTB', 'MTRN',
    ],
}

# Tickers for which we produce PNG figures.  Keep small — every entry
# multiplies the figure count by ~4.
PLOT_TICKERS = ['AAPL']

# Timeframes per experiment
EXP1_START,     EXP1_END     = '2010-01-01', '2024-01-01'
EXP2_VOL_START, EXP2_VOL_END = '2005-01-01', '2024-01-01'
EXP2_BB_START,  EXP2_BB_END  = '2018-01-01', '2024-01-01'

# Feature groups
VOL_FEATURES = ['daily return', 'rolling volatility', 'absolute return']
BB_FEATURES  = ['20-day return', '60-day return', '100-day return']
ALL_FEATURES = VOL_FEATURES + BB_FEATURES   # for Exp 1

# Order experiment: which 3 permutations of the bb group to plot
PERM_PLOT_SEED  = 42
N_PERMS_TO_PLOT = 3

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────
# 1. DATA AND FEATURES
# ─────────────────────────────────────────────────────────────────────────

import pandas as pd  # for reading the frozen snapshot CSVs

# Directory holding the frozen snapshot produced by the snapshot script.
# Expected files: data_snapshot/<TICKER>__<START>__<END>.csv  (columns: Date, Close)
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data_snapshot')


def get_data(ticker, start, end):
    """Read frozen close prices from the local snapshot instead of yfinance.

    Returns (close_array, date_index) — identical signature to the original
    network version, so nothing downstream changes. Raises FileNotFoundError
    if the snapshot file is missing (a loud failure beats a silent re-download
    that would reintroduce data drift)."""
    path = os.path.join(SNAPSHOT_DIR, f'{ticker}__{start}__{end}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No snapshot for {ticker} [{start}..{end}] at {path}. '
            f'Run the snapshot script first, or check the ticker/timeframe.')
    # Read robustly: yfinance-saved CSVs can carry extra header rows
    # (e.g. a 'Ticker,AAPL' line), which would otherwise make the Close
    # column parse as text. Coerce to numeric and drop any non-data rows.
    df = pd.read_csv(path, index_col=0)
    if 'Close' not in df.columns:
        raise ValueError(f'{path} has no Close column (cols: {list(df.columns)})')
    df = df[['Close']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()].dropna(subset=['Close'])
    return df['Close'].values.flatten(), df.index


def make_features(close, vol_window=20, roll_short=20,
                  roll_long=60, roll_extralong=100):
    n = len(close)

    daily_return     = np.full(n, np.nan)
    daily_return[1:] = (close[1:] - close[:-1]) / close[:-1]
    abs_return       = np.abs(daily_return)

    rolling_vol = np.full(n, np.nan)
    for t in range(vol_window, n):
        rolling_vol[t] = np.std(daily_return[t - vol_window : t])

    roll_return_20 = np.full(n, np.nan)
    for t in range(roll_short, n):
        roll_return_20[t] = ((close[t] - close[t - roll_short])
                              / close[t - roll_short])

    roll_return_60 = np.full(n, np.nan)
    for t in range(roll_long, n):
        roll_return_60[t] = ((close[t] - close[t - roll_long])
                              / close[t - roll_long])

    roll_return_100 = np.full(n, np.nan)
    for t in range(roll_extralong, n):
        roll_return_100[t] = ((close[t] - close[t - roll_extralong])
                              / close[t - roll_extralong])

    features = {
        'daily return':       daily_return,
        'absolute return':    abs_return,
        'rolling volatility': rolling_vol,
        '20-day return':      roll_return_20,
        '60-day return':      roll_return_60,
        '100-day return':     roll_return_100,
    }

    valid = np.ones(n, dtype=bool)
    for arr in features.values():
        valid &= ~np.isnan(arr)
    return features, valid


def load_data(ticker, start, end):
    """Download and feature-align one stock over one timeframe.
    Returns (close, dates, daily_returns, feature_getter) or None on failure.
    """
    try:
        close, dates = get_data(ticker, start, end)
    except Exception as e:
        print(f"    DOWNLOAD FAILED for {ticker}: {e}")
        return None
    if len(close) == 0:
        print(f"    NO DATA for {ticker} in {start}–{end}")
        return None

    features, valid = make_features(close)
    aligned_close   = close[valid]
    aligned_dates   = dates[valid]
    daily_returns   = features['daily return'][valid]

    if len(aligned_close) < 200:
        print(f"    TOO FEW DAYS for {ticker} after alignment "
              f"({len(aligned_close)} rows)")
        return None

    def f(name):
        return features[name][valid]

    return aligned_close, aligned_dates, daily_returns, f


# ─────────────────────────────────────────────────────────────────────────
# 2. HMM FIT / DECODE / NORMALISE
# ─────────────────────────────────────────────────────────────────────────

def fit_hmm(obs, n_restarts=5):
    best_score = -np.inf
    best_model = None
    for seed in range(n_restarts):
        try:
            model = hmm.GaussianHMM(
                n_components    = 2,
                covariance_type = 'full',
                n_iter          = 200,
                tol             = 1e-4,
                random_state    = seed,
            )
            model.fit(obs)
            score = model.score(obs)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue
    return best_model


def decode_states(model, obs):
    return model.predict(obs)


def normalise_state_ordering(states, model, feature_to_sort_by=0,
                             sort_by='mean'):
    """Reorder states so that state 0 has the lower value of the chosen
    statistic ('mean' or 'sigma') on the chosen feature dimension."""
    if sort_by == 'mean':
        values = model.means_[:, feature_to_sort_by]
    elif sort_by == 'sigma':
        values = np.sqrt(model.covars_[:, feature_to_sort_by,
                                        feature_to_sort_by])
    else:
        raise ValueError(f"sort_by must be 'mean' or 'sigma', got {sort_by!r}")

    if values[0] > values[1]:
        states     = 1 - states
        means_full = model.means_[::-1]
        covs_full  = model.covars_[::-1]
        trans      = model.transmat_[::-1, ::-1]
    else:
        means_full = model.means_
        covs_full  = model.covars_
        trans      = model.transmat_
    return states, {
        'means':    means_full,
        'covars':   covs_full,
        'transmat': trans,
    }


def state_agreement(a, b):
    eq = (a == b).mean()
    return max(eq, 1 - eq) * 100


def fit_decode(obs, feature_names, sort_by='mean'):
    """Fit + decode + normalise, return (states, info, sigmas) or
    (None, None, None).  sort_by='mean' (default) puts the lower-mean
    state at index 0; 'sigma' puts the lower-σ state at index 0."""
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    model = fit_hmm(obs, n_restarts=5)
    if model is None:
        return None, None, None
    raw_states   = decode_states(model, obs)
    states, info = normalise_state_ordering(raw_states, model, sort_by=sort_by)
    sigma_0 = np.sqrt(np.diag(info['covars'][0]))
    sigma_1 = np.sqrt(np.diag(info['covars'][1]))
    return states, info, (sigma_0, sigma_1)


# ─────────────────────────────────────────────────────────────────────────
# 3. PLOTTING (stacked panels)
# ─────────────────────────────────────────────────────────────────────────

def plot_experiment(panels, filename):
    n = len(panels)
    fig = plt.figure(figsize=(14, 5.5 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.55)
    ax_pairs = []
    for i in range(n):
        inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_pairs.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))

    for i, panel in enumerate(panels):
        ax_price, ax_ret = ax_pairs[i]
        close, dates   = panel['close'], panel['dates']
        states, info   = panel['states'], panel['info']
        returns, title = panel['returns'], panel['title']

        means = info['means']
        if means.shape[1] == 1:
            sigma = np.sqrt(info['covars'][:, 0, 0])
            info_text = (f"state 0  μ={means[0,0]:+.5f}  σ={sigma[0]:.5f}  "
                         f"({(states==0).mean()*100:.1f}%)   |   "
                         f"state 1  μ={means[1,0]:+.5f}  σ={sigma[1]:.5f}  "
                         f"({(states==1).mean()*100:.1f}%)")
        else:
            info_text = (f"state 0 means={np.round(means[0], 5)}  "
                         f"({(states==0).mean()*100:.1f}%)   |   "
                         f"state 1 means={np.round(means[1], 5)}  "
                         f"({(states==1).mean()*100:.1f}%)")

        j = 0
        while j < len(states):
            k = j + 1
            while k < len(states) and states[k] == states[j]:
                k += 1
            color = COLOR_STATE_0 if states[j] == 0 else COLOR_STATE_1
            for ax in (ax_price, ax_ret):
                ax.axvspan(dates[j], dates[min(k, len(dates) - 1)],
                           alpha=STRIPE_ALPHA, color=color, lw=0)
            j = k

        ax_price.plot(dates, close, lw=2.5, color='#111', zorder=3)
        ax_price.set_title(f"{title}\n{info_text}", pad=10)
        ax_price.set_ylabel('Close price')
        ax_price.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax_price.legend(handles=[
            mpatches.Patch(color=COLOR_STATE_0, alpha=STRIPE_ALPHA + 0.25,
                           label='state 0 (lower mean)'),
            mpatches.Patch(color=COLOR_STATE_1, alpha=STRIPE_ALPHA + 0.25,
                           label='state 1 (higher mean)'),
        ], loc='upper left', framealpha=0.95)

        ax_ret.plot(dates, returns * 100, lw=1.0, color='#222',
                    alpha=0.85, zorder=3)
        ax_ret.axhline(0, color='#888', lw=1.2)
        ax_ret.set_ylabel('Daily return (%)')
        ax_ret.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax_ret.set_xlabel('Date')

    plt.savefig(filename, facecolor='white')
    plt.close(fig)
    print(f"      saved {os.path.basename(filename)}")


def _gaussian_pdf(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * sig * sig)) / (sig * np.sqrt(2 * np.pi))


def plot_exp1_densities(ticker, rows, filename):
    """2×3 grid: fitted emission Gaussians for each Exp-1 feature.
    Row 1 = volatility group, row 2 = trend group."""
    by_feature = {}
    for r in rows:
        by_feature.setdefault(r['feature'], {})[r['state']] = r

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for idx, name in enumerate(ALL_FEATURES):
        ax = axes[idx // 3, idx % 3]
        if name not in by_feature or 0 not in by_feature[name] \
                or 1 not in by_feature[name]:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='#888')
            ax.set_title(name, fontsize=13, weight='bold')
            continue

        s0, s1 = by_feature[name][0], by_feature[name][1]
        x_min = min(s0['mu'] - 3.5 * s0['sigma'],
                    s1['mu'] - 3.5 * s1['sigma'])
        x_max = max(s0['mu'] + 3.5 * s0['sigma'],
                    s1['mu'] + 3.5 * s1['sigma'])
        if name == 'absolute return':
            x_min = max(x_min, 0.0)
        xs = np.linspace(x_min, x_max, 400)
        y0 = _gaussian_pdf(xs, s0['mu'], s0['sigma'])
        y1 = _gaussian_pdf(xs, s1['mu'], s1['sigma'])

        ax.fill_between(xs, y0, alpha=0.25, color=COLOR_STATE_0, zorder=2)
        ax.plot(xs, y0, color=COLOR_STATE_0, lw=2.2, zorder=3,
                label=(f"s0  μ={s0['mu']:+.4f}  σ={s0['sigma']:.4f}  "
                       f"({s0['occupancy_pct']:.0f}%)"))
        ax.fill_between(xs, y1, alpha=0.25, color=COLOR_STATE_1, zorder=2)
        ax.plot(xs, y1, color=COLOR_STATE_1, lw=2.2, linestyle='--',
                zorder=3,
                label=(f"s1  μ={s1['mu']:+.4f}  σ={s1['sigma']:.4f}  "
                       f"({s1['occupancy_pct']:.0f}%)"))

        ax.set_title(name, fontsize=13, weight='bold')
        ax.set_xlabel('feature value', fontsize=11)
        ax.set_ylabel('density', fontsize=11)
        ax.legend(fontsize=9, loc='best', framealpha=0.92)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

    fig.text(0.005, 0.74, 'Volatility group', rotation=90,
             fontsize=14, weight='bold', va='center')
    fig.text(0.005, 0.27, 'Trend group', rotation=90,
             fontsize=14, weight='bold', va='center')
    fig.suptitle(f"{ticker} — Exp 1: emission distributions per feature",
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    plt.savefig(filename, facecolor='white')
    plt.close(fig)
    print(f"      saved {os.path.basename(filename)}")


def plot_exp1_scatter(ticker, rows, filename):
    """Summary scatter showing the vol/trend dichotomy:
    each feature plotted at (|Δμ|, σ-ratio).  Vol features
    should cluster in low-Δμ / high-ratio, trend features in the opposite."""
    from matplotlib.lines import Line2D
    by_feature = {}
    for r in rows:
        by_feature.setdefault(r['feature'], {})[r['state']] = r

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in ALL_FEATURES:
        if name not in by_feature or 0 not in by_feature[name] \
                or 1 not in by_feature[name]:
            continue
        s0, s1 = by_feature[name][0], by_feature[name][1]
        delta_mu = abs(s1['mu'] - s0['mu'])
        denom = max(min(s0['sigma'], s1['sigma']), 1e-12)
        sigma_ratio = max(s0['sigma'], s1['sigma']) / denom

        is_vol = name in VOL_FEATURES
        ax.scatter(delta_mu, sigma_ratio, s=220,
                   c=COLOR_STATE_0 if is_vol else COLOR_STATE_1,
                   marker='o' if is_vol else 's',
                   edgecolors='black', linewidths=1.6, zorder=4)
        ax.annotate(name, xy=(delta_mu, sigma_ratio),
                    xytext=(11, 9), textcoords='offset points',
                    fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markersize=12,
               markerfacecolor=COLOR_STATE_0, markeredgecolor='black',
               markeredgewidth=1.6, label='Volatility group'),
        Line2D([0], [0], marker='s', color='w', markersize=12,
               markerfacecolor=COLOR_STATE_1, markeredgecolor='black',
               markeredgewidth=1.6, label='Trend group'),
    ]
    ax.legend(handles=legend_handles, loc='best', fontsize=12,
              framealpha=0.92)

    ax.axhline(1, color='gray', linestyle=':', alpha=0.7, zorder=1)
    ax.set_xlabel('Mean spread  |μ₁ − μ₀|', fontsize=13)
    ax.set_ylabel('σ-ratio  max(σ₀, σ₁) / min(σ₀, σ₁)', fontsize=13)
    ax.set_title(f'{ticker} — Exp 1: feature dichotomy',
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

    plt.tight_layout()
    plt.savefig(filename, facecolor='white')
    plt.close(fig)
    print(f"      saved {os.path.basename(filename)}")


# ─────────────────────────────────────────────────────────────────────────
# 4. PER-STOCK EXPERIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def run_exp1(ticker, cap_tier, data, do_plot):
    """Six univariate HMMs.  Returns list of result dicts for CSV.

    For volatility-group features (daily return, abs return, rolling vol)
    we sort states by σ so that state 0 = lower σ = calm regime, giving
    consistent visual encoding across these three features.  For
    trend-group features we keep the default mean-sort so state 0 = bear.
    """
    close, dates, returns, f = data
    rows  = []
    panels = []
    for name in ALL_FEATURES:
        obs = f(name).reshape(-1, 1)
        sort_by = 'sigma' if name in VOL_FEATURES else 'mean'
        states, info, sigmas = fit_decode(obs, [name], sort_by=sort_by)
        if states is None:
            continue
        sig0, sig1 = sigmas
        occ0 = (states == 0).mean() * 100
        occ1 = (states == 1).mean() * 100
        rows.append({
            'ticker': ticker, 'cap_tier': cap_tier, 'feature': name,
            'state': 0,
            'mu':    float(info['means'][0, 0]),
            'sigma': float(sig0[0]),
            'occupancy_pct': float(occ0),
            'sort_by': sort_by,
        })
        rows.append({
            'ticker': ticker, 'cap_tier': cap_tier, 'feature': name,
            'state': 1,
            'mu':    float(info['means'][1, 0]),
            'sigma': float(sig1[0]),
            'occupancy_pct': float(occ1),
            'sort_by': sort_by,
        })
        if do_plot:
            panels.append({
                'close': close, 'dates': dates, 'states': states,
                'info': info, 'returns': returns,
                'title': f"{ticker} — Exp 1: feature = {name}",
            })

    if do_plot and panels:
        plot_experiment(panels,
                        os.path.join(OUTPUT_DIR, f'{ticker}_exp1.png'))
    if do_plot and rows:
        plot_exp1_densities(
            ticker, rows,
            os.path.join(OUTPUT_DIR, f'{ticker}_exp1_densities.png'))
        plot_exp1_scatter(
            ticker, rows,
            os.path.join(OUTPUT_DIR, f'{ticker}_exp1_scatter.png'))
    return rows


def run_stacking(ticker, cap_tier, group_name, features, data, do_plot,
                 plot_filename):
    """Cumulative 1D -> 2D -> 3D stacking on the given feature group."""
    close, dates, returns, f = data
    rows = []
    panels = []
    state_seqs_per_step = []   # for the 1D-vs-2D-vs-3D agreement matrix

    for k in range(1, len(features) + 1):
        active = features[:k]
        obs = np.column_stack([f(name) for name in active])
        states, info, sigmas = fit_decode(obs, active)
        if states is None:
            continue
        sig0, sig1 = sigmas
        occ0 = (states == 0).mean() * 100
        occ1 = (states == 1).mean() * 100
        for state, sig in [(0, sig0), (1, sig1)]:
            for feat_idx, feat_name in enumerate(active):
                rows.append({
                    'ticker': ticker, 'cap_tier': cap_tier,
                    'group': group_name,
                    'step_n_features': k,
                    'feature_combo': ' + '.join(active),
                    'state': state,
                    'feature_index': feat_idx,
                    'feature_name':  feat_name,
                    'mu':    float(info['means'][state, feat_idx]),
                    'sigma': float(sig[feat_idx]),
                    'occupancy_pct': float(occ0 if state == 0 else occ1),
                })
        state_seqs_per_step.append((k, states))
        if do_plot:
            panels.append({
                'close': close, 'dates': dates, 'states': states,
                'info': info, 'returns': returns,
                'title': (f"{ticker} — Exp 2 {group_name} group, "
                          f"{k}D: {' + '.join(active)}"),
            })

    if do_plot and panels:
        plot_experiment(panels, plot_filename)

    # Pairwise agreement across stacking steps (1D vs 2D, 1D vs 3D, 2D vs 3D)
    step_agreements = {}
    for i in range(len(state_seqs_per_step)):
        for j in range(i + 1, len(state_seqs_per_step)):
            ki, si = state_seqs_per_step[i]
            kj, sj = state_seqs_per_step[j]
            step_agreements[f'{ki}D_vs_{kj}D'] = state_agreement(si, sj)

    return rows, step_agreements


def run_order_experiment(ticker, cap_tier, group_name, features, data,
                         do_plot, plot_filename):
    """All 6 permutations of the 3-feature input.  Returns:
        rows           list of dicts for the agreement-matrix CSV
        max_disagree   the largest (100 - agreement) seen across permutation pairs
    """
    close, dates, returns, f = data
    perms = list(permutations(range(len(features))))
    state_seqs   = []
    perm_strs    = []
    perm_panels  = []

    for perm in perms:
        ordered = [features[i] for i in perm]
        obs = np.column_stack([f(name) for name in ordered])
        states, info, _ = fit_decode(obs, ordered)
        if states is None:
            state_seqs.append(None)
            perm_strs.append(' + '.join(ordered))
            continue
        state_seqs.append(states)
        perm_strs.append(' + '.join(ordered))
        perm_panels.append({
            'close': close, 'dates': dates, 'states': states,
            'info': info, 'returns': returns,
            'title': (f"{ticker} — Exp 2 {group_name} order perm: "
                      f"{' + '.join(ordered)}"),
        })

    # Pairwise agreement matrix
    rows = []
    max_disagree = 0.0
    n = len(perms)
    for i in range(n):
        for j in range(n):
            if state_seqs[i] is None or state_seqs[j] is None:
                agr = float('nan')
            elif i == j:
                agr = 100.0
            else:
                agr = state_agreement(state_seqs[i], state_seqs[j])
                if not np.isnan(agr):
                    max_disagree = max(max_disagree, 100.0 - agr)
            rows.append({
                'ticker': ticker, 'cap_tier': cap_tier,
                'group': group_name,
                'perm_i_index': i, 'perm_j_index': j,
                'perm_i_features': perm_strs[i],
                'perm_j_features': perm_strs[j],
                'agreement_pct': float(agr) if not np.isnan(agr) else None,
            })

    # Pick N_PERMS_TO_PLOT random permutations and stack them in one figure
    if do_plot and len(perm_panels) >= N_PERMS_TO_PLOT:
        rng = np.random.RandomState(PERM_PLOT_SEED)
        choice = sorted(rng.choice(len(perm_panels),
                                   size=N_PERMS_TO_PLOT, replace=False))
        chosen_panels = [perm_panels[c] for c in choice]
        plot_experiment(chosen_panels, plot_filename)

    return rows, max_disagree


# ─────────────────────────────────────────────────────────────────────────
# 5. CSV WRITING
# ─────────────────────────────────────────────────────────────────────────

def write_csv(rows, path, fieldnames):
    if not rows:
        return
    with open(path, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows):>5} rows -> {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────
# 6. STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, ci=95, seed=0):
    """Bootstrap confidence interval for a statistic over a 1D array.

    Resamples `data` with replacement `n_boot` times.  Returns (lower, upper)
    percentile bounds at the requested CI level.  The sampling unit is one
    stock, so each resample draws 24 stocks with replacement — this quantifies
    how stable the panel-mean estimate is to the specific set of 24 stocks
    chosen.
    """
    rng = np.random.RandomState(seed)
    n   = len(data)
    boot_stats = [stat_fn(rng.choice(data, size=n, replace=True))
                  for _ in range(n_boot)]
    alpha = (100 - ci) / 2
    return (float(np.percentile(boot_stats, alpha)),
            float(np.percentile(boot_stats, 100 - alpha)))


def permutation_test(group_a, group_b, n_perm=10000, seed=0):
    """Two-sample permutation test for difference in means (a − b).

    Procedure:
      1. Compute the observed mean difference.
      2. Pool group_a and group_b, shuffle 10 000 times, split at len(group_a),
         and record the mean difference each time.
      3. p-value = fraction of permuted differences whose absolute value is at
         least as extreme as the observed absolute difference (two-tailed).

    Returns (observed_diff, p_value).
    """
    rng     = np.random.RandomState(seed)
    a       = np.asarray(group_a, dtype=float)
    b       = np.asarray(group_b, dtype=float)
    observed = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a      = len(a)
    count = 0
    for _ in range(n_perm):
        perm     = rng.permutation(len(combined))
        perm_diff = np.mean(combined[perm[:n_a]]) - np.mean(combined[perm[n_a:]])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return float(observed), float(count / n_perm)


def run_statistical_tests(exp1_rows, exp2_rows, n_boot=1000, n_perm=10000):
    """Compute bootstrap CIs and permutation tests over the collected rows.

    Bootstrap CIs (sampling unit = one stock, 24 total):
      - Exp 1 : σ-ratio and |Δμ| per feature
      - Exp 2 : minority-state occupancy at each stacking step per group

    Permutation tests (two-tailed, 10 000 shuffles):
      1. Vol 3D vs BB 3D minority occupancy       — main asymmetry / collapse claim
      2. Vol 1D vs Vol 3D minority occupancy      — collapse within vol group
      3. BB  1D vs BB  3D minority occupancy      — preservation within bb group
      4. Vol σ-ratio vs BB σ-ratio (Exp 1)        — variance-vs-direction dichotomy

    Returns (bootstrap_rows, permutation_rows) as lists of dicts.
    """
    from collections import defaultdict

    bootstrap_rows   = []
    permutation_rows = []

    # ── Exp 1: σ-ratio and |Δμ| per feature, bootstrapped ────────────────
    exp1_by = defaultdict(dict)   # {(ticker, feature): {state: row}}
    for r in exp1_rows:
        exp1_by[(r['ticker'], r['feature'])][r['state']] = r

    feature_sigma_ratios = defaultdict(list)
    feature_delta_mus    = defaultdict(list)
    for (ticker, feature), states in exp1_by.items():
        if 0 not in states or 1 not in states:
            continue
        s0, s1 = states[0], states[1]
        if s0['sigma'] > 0 and s1['sigma'] > 0:
            feature_sigma_ratios[feature].append(
                max(s0['sigma'], s1['sigma']) / min(s0['sigma'], s1['sigma']))
        feature_delta_mus[feature].append(abs(s1['mu'] - s0['mu']))

    for feature in ALL_FEATURES:
        for stat_name, values in [('sigma_ratio',   feature_sigma_ratios[feature]),
                                   ('abs_delta_mu', feature_delta_mus[feature])]:
            if len(values) < 2:
                continue
            arr     = np.array(values)
            lo, hi  = bootstrap_ci(arr, n_boot=n_boot)
            bootstrap_rows.append({
                'experiment': 'exp1',
                'group':      'vol' if feature in VOL_FEATURES else 'bb',
                'feature':    feature,
                'statistic':  stat_name,
                'n_stocks':   len(arr),
                'mean':       float(np.mean(arr)),
                'ci_lower':   lo,
                'ci_upper':   hi,
            })

    # ── Exp 2: minority occupancy per (group, step), bootstrapped ─────────
    # One row per (ticker, group, step) — use feature_index==0 to avoid
    # double-counting the multi-feature rows that share the same occupancy.
    occ_by = defaultdict(dict)   # {(ticker, group, step): {state: occ}}
    for r in exp2_rows:
        if r['feature_index'] == 0:
            key = (r['ticker'], r['group'], r['step_n_features'])
            occ_by[key][r['state']] = r['occupancy_pct']

    step_occ = defaultdict(list)  # {(group, step): [minority_occ per ticker]}
    for (ticker, group, step), states in occ_by.items():
        if 0 in states and 1 in states:
            step_occ[(group, step)].append(min(states[0], states[1]))

    for (group, step), values in sorted(step_occ.items()):
        if len(values) < 2:
            continue
        arr    = np.array(values)
        lo, hi = bootstrap_ci(arr, n_boot=n_boot)
        bootstrap_rows.append({
            'experiment': 'exp2_stacking',
            'group':      group,
            'feature':    f'{step}D',
            'statistic':  'minority_occupancy_pct',
            'n_stocks':   len(arr),
            'mean':       float(np.mean(arr)),
            'ci_lower':   lo,
            'ci_upper':   hi,
        })

    # ── Permutation tests ─────────────────────────────────────────────────

    def _perm_row(comparison, description, label_a, label_b, a, b):
        """Helper: run permutation test and return a result dict."""
        if len(a) < 2 or len(b) < 2:
            return None
        diff, pval = permutation_test(a, b, n_perm=n_perm)
        return {
            'comparison':    comparison,
            'description':   description,
            'group_a':       label_a,
            'group_b':       label_b,
            'mean_a':        float(np.mean(a)),
            'mean_b':        float(np.mean(b)),
            'observed_diff': float(diff),
            'p_value':       float(pval),
            'n_a':           len(a),
            'n_b':           len(b),
            'significant_at_0.05': float(pval) < 0.05,
        }

    # 1. Main asymmetry: vol 3D vs bb 3D minority occupancy
    row = _perm_row(
        'vol_3D_vs_bb_3D_minority_occ',
        'Vol-group 3D occupancy higher than bb-group 3D (collapse vs preservation)',
        'vol_3D', 'bb_3D',
        np.array(step_occ.get(('vol', 3), [])),
        np.array(step_occ.get(('bb',  3), [])),
    )
    if row: permutation_rows.append(row)

    # 2. Vol 1D vs Vol 3D: occupancy rises (collapse within vol group)
    row = _perm_row(
        'vol_1D_vs_vol_3D_minority_occ',
        'Vol-group minority occupancy rises from 1D to 3D (regime collapse)',
        'vol_3D', 'vol_1D',
        np.array(step_occ.get(('vol', 3), [])),
        np.array(step_occ.get(('vol', 1), [])),
    )
    if row: permutation_rows.append(row)

    # 3. BB 1D vs BB 3D: occupancy stable (preservation within bb group)
    row = _perm_row(
        'bb_1D_vs_bb_3D_minority_occ',
        'BB-group minority occupancy stable from 1D to 3D (regime preservation)',
        'bb_3D', 'bb_1D',
        np.array(step_occ.get(('bb', 3), [])),
        np.array(step_occ.get(('bb', 1), [])),
    )
    if row: permutation_rows.append(row)

    # 4. σ-ratio dichotomy: vol features higher σ-ratio than bb features (Exp 1)
    vol_sr = np.array([v for f in VOL_FEATURES for v in feature_sigma_ratios[f]])
    bb_sr  = np.array([v for f in BB_FEATURES  for v in feature_sigma_ratios[f]])
    row = _perm_row(
        'vol_sigma_ratio_vs_bb_sigma_ratio',
        'Vol-group features have higher σ-ratio than bb-group (variance vs direction dichotomy)',
        'vol_sigma_ratio', 'bb_sigma_ratio',
        vol_sr, bb_sr,
    )
    if row: permutation_rows.append(row)

    return bootstrap_rows, permutation_rows


# ─────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    exp1_rows         = []
    exp2_rows         = []
    order_rows        = []
    summary_rows      = []

    flat_tickers = [(tier, t) for tier, ts in STOCKS_BY_TIER.items()
                              for t in ts]
    total = len(flat_tickers)

    for idx, (cap_tier, ticker) in enumerate(flat_tickers, start=1):
        print(f"\n[{idx:>2}/{total}] {ticker}  ({cap_tier} cap)")
        do_plots = ticker in PLOT_TICKERS

        # --- Exp 1 -------------------------------------------------------
        print(f"  loading Exp 1 timeframe {EXP1_START}..{EXP1_END}")
        e1 = load_data(ticker, EXP1_START, EXP1_END)
        if e1 is not None:
            rs = run_exp1(ticker, cap_tier, e1, do_plots)
            exp1_rows.extend(rs)
            print(f"  Exp 1 done ({len(rs)//2} features fit)")
        else:
            print(f"  Exp 1 SKIPPED")

        # --- Exp 2 vol stacking + order ---------------------------------
        print(f"  loading Exp 2 vol timeframe {EXP2_VOL_START}..{EXP2_VOL_END}")
        e2v = load_data(ticker, EXP2_VOL_START, EXP2_VOL_END)
        agr_vol = {}
        max_disagree_vol = float('nan')
        if e2v is not None:
            stack_rows, agr_vol = run_stacking(
                ticker, cap_tier, 'vol', VOL_FEATURES, e2v, do_plots,
                os.path.join(OUTPUT_DIR, f'{ticker}_exp2_vol.png'),
            )
            exp2_rows.extend(stack_rows)
            print(f"  Exp 2 vol stacking done")

            ord_rows, max_disagree_vol = run_order_experiment(
                ticker, cap_tier, 'vol', VOL_FEATURES, e2v, False,
                os.path.join(OUTPUT_DIR, f'{ticker}_order_vol.png'),
            )
            order_rows.extend(ord_rows)
            print(f"  Exp 2 vol order experiment done  "
                  f"(max disagreement: {max_disagree_vol:.1f}%)")
        else:
            print(f"  Exp 2 vol SKIPPED")

        # --- Exp 2 bb stacking + order ----------------------------------
        print(f"  loading Exp 2 bb timeframe {EXP2_BB_START}..{EXP2_BB_END}")
        e2b = load_data(ticker, EXP2_BB_START, EXP2_BB_END)
        agr_bb = {}
        max_disagree_bb = float('nan')
        if e2b is not None:
            stack_rows, agr_bb = run_stacking(
                ticker, cap_tier, 'bb', BB_FEATURES, e2b, do_plots,
                os.path.join(OUTPUT_DIR, f'{ticker}_exp2_bb.png'),
            )
            exp2_rows.extend(stack_rows)
            print(f"  Exp 2 bb stacking done")

            ord_rows, max_disagree_bb = run_order_experiment(
                ticker, cap_tier, 'bb', BB_FEATURES, e2b, do_plots,
                os.path.join(OUTPUT_DIR, f'{ticker}_order_bb.png'),
            )
            order_rows.extend(ord_rows)
            print(f"  Exp 2 bb order experiment done   "
                  f"(max disagreement: {max_disagree_bb:.1f}%)")
        else:
            print(f"  Exp 2 bb SKIPPED")

        # --- Per-stock summary row --------------------------------------
        var_ratio_1d = float('nan')
        if e1 is not None:
            # variance ratio = sigma_0 / sigma_1 for daily-return univariate fit
            dr_rows = [r for r in exp1_rows
                       if r['ticker'] == ticker and r['feature'] == 'daily return']
            if len(dr_rows) == 2:
                s0 = dr_rows[0]['sigma']
                s1 = dr_rows[1]['sigma']
                if s0 > 0 and s1 > 0:
                    var_ratio_1d = max(s0, s1) / min(s0, s1)

        summary_rows.append({
            'ticker': ticker,
            'cap_tier': cap_tier,
            'daily_return_sigma_ratio': var_ratio_1d,
            'vol_1D_vs_2D_agreement': agr_vol.get('1D_vs_2D', float('nan')),
            'vol_1D_vs_3D_agreement': agr_vol.get('1D_vs_3D', float('nan')),
            'vol_2D_vs_3D_agreement': agr_vol.get('2D_vs_3D', float('nan')),
            'bb_1D_vs_2D_agreement':  agr_bb.get('1D_vs_2D', float('nan')),
            'bb_1D_vs_3D_agreement':  agr_bb.get('1D_vs_3D', float('nan')),
            'bb_2D_vs_3D_agreement':  agr_bb.get('2D_vs_3D', float('nan')),
            'vol_order_max_disagree_pct': max_disagree_vol,
            'bb_order_max_disagree_pct':  max_disagree_bb,
        })

    # ─────────────────────────────────────────────────────────────────────
    # WRITE ALL CSVs
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WRITING CSVs")
    print("=" * 70)

    write_csv(
        exp1_rows, os.path.join(OUTPUT_DIR, 'exp1_results.csv'),
        ['ticker', 'cap_tier', 'feature', 'state', 'mu', 'sigma',
         'occupancy_pct', 'sort_by'],
    )
    write_csv(
        exp2_rows, os.path.join(OUTPUT_DIR, 'exp2_stacking.csv'),
        ['ticker', 'cap_tier', 'group', 'step_n_features',
         'feature_combo', 'state', 'feature_index', 'feature_name',
         'mu', 'sigma', 'occupancy_pct'],
    )
    write_csv(
        order_rows, os.path.join(OUTPUT_DIR, 'order_experiment.csv'),
        ['ticker', 'cap_tier', 'group', 'perm_i_index', 'perm_j_index',
         'perm_i_features', 'perm_j_features', 'agreement_pct'],
    )
    write_csv(
        summary_rows, os.path.join(OUTPUT_DIR, 'summary.csv'),
        ['ticker', 'cap_tier', 'daily_return_sigma_ratio',
         'vol_1D_vs_2D_agreement', 'vol_1D_vs_3D_agreement',
         'vol_2D_vs_3D_agreement', 'bb_1D_vs_2D_agreement',
         'bb_1D_vs_3D_agreement', 'bb_2D_vs_3D_agreement',
         'vol_order_max_disagree_pct', 'bb_order_max_disagree_pct'],
    )

    # ─────────────────────────────────────────────────────────────────────
    # STATISTICAL TESTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RUNNING STATISTICAL TESTS  (bootstrap n=1000, permutation n=10000)")
    print("=" * 70)

    bootstrap_rows, permutation_rows = run_statistical_tests(
        exp1_rows, exp2_rows, n_boot=1000, n_perm=10000,
    )

    write_csv(
        bootstrap_rows,
        os.path.join(OUTPUT_DIR, 'bootstrap_ci.csv'),
        ['experiment', 'group', 'feature', 'statistic',
         'n_stocks', 'mean', 'ci_lower', 'ci_upper'],
    )
    write_csv(
        permutation_rows,
        os.path.join(OUTPUT_DIR, 'permutation_tests.csv'),
        ['comparison', 'description', 'group_a', 'group_b',
         'mean_a', 'mean_b', 'observed_diff', 'p_value',
         'n_a', 'n_b', 'significant_at_0.05'],
    )

    # Print permutation test summary to console
    print("\n  Permutation test summary:")
    for r in permutation_rows:
        sig = "*** SIGNIFICANT ***" if r['significant_at_0.05'] else "not significant"
        print(f"    {r['comparison']}")
        print(f"      diff={r['observed_diff']:+.3f}  p={r['p_value']:.4f}  {sig}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
