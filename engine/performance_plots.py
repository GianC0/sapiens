"""
Performance Plotting Module
===========================

Modular plotting functions for backtest performance analysis.
Each function is standalone and can be used in:
- Hyperparameter tuning workflows
- Jupyter notebooks
- Production backtests

All functions:
- Accept standardized inputs (time series, config params)
- Return matplotlib Figure objects
- Optionally log to MLflow
- Use seaborn styling for consistency
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set seaborn theme globally for module
sns.set_theme(style='darkgrid', palette='husl')


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def ensure_tz(series: pd.Series, target_tz: str = 'UTC') -> pd.Series:
    """Ensure series has consistent timezone."""
    if series.index.tz is None:
        return series.tz_localize(target_tz)
    return series.tz_convert(target_tz)


def get_frequency_params(freq: str) -> Dict:
    """
    Extract frequency parameters for annualization and resampling.
    
    Returns:
        Dict with periods_per_year, resample_freq, weekly_freq, monthly_freq
    """
    if 'D' in freq or 'B' in freq:
        return {
            'periods_per_year': 252,
            'resample_freq': '1D',
            'weekly_freq': 'W',
            'monthly_freq': 'ME',
            'annualization_factor': np.sqrt(252)
        }
    elif 'H' in freq:
        return {
            'periods_per_year': 252 * 6.5,
            'resample_freq': '1H',
            'weekly_freq': '1D',
            'monthly_freq': 'W',
            'annualization_factor': np.sqrt(252 * 6.5)
        }
    elif 'T' in freq or 'm' in freq:
        match = re.search(r'(\d+)', freq)
        minutes = int(match.group(1)) if match else 1
        periods_per_year = 252 * 6.5 * 60 / minutes
        return {
            'periods_per_year': periods_per_year,
            'resample_freq': freq,
            'weekly_freq': '1D',
            'monthly_freq': 'W',
            'annualization_factor': np.sqrt(periods_per_year)
        }
    else:
        return {
            'periods_per_year': 252,
            'resample_freq': '1D',
            'weekly_freq': 'W',
            'monthly_freq': 'ME',
            'annualization_factor': np.sqrt(252)
        }


def align_series(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    rf_ret: pd.Series,
    resample_freq: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Align strategy, benchmark, and risk-free return series with proper timezone handling.
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (original frequency)
        rf_ret: Risk-free returns (original frequency)
        resample_freq: Target frequency for resampling
    
    Returns:
        Tuple of aligned (strategy, benchmark, rf) series
    """
    # Resample benchmark/rf to match strategy frequency
    benchmark_resampled = benchmark_ret.resample(resample_freq).apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    )
    rf_resampled = rf_ret.resample(resample_freq).apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    )
    
    # Ensure timezone consistency
    strategy_ret = ensure_tz(strategy_ret)
    benchmark_resampled = ensure_tz(benchmark_resampled)
    rf_resampled = ensure_tz(rf_resampled)
    
    # Align to common date range
    common_dates = strategy_ret.index.intersection(benchmark_resampled.index)
    
    if len(common_dates) < 2:
        logger.warning(f"Insufficient overlap ({len(common_dates)} dates). Using strategy dates with ffill.")
        common_dates = strategy_ret.index
        benchmark_resampled = benchmark_resampled.reindex(common_dates, method='ffill').fillna(0)
        rf_resampled = rf_resampled.reindex(common_dates, method='ffill').fillna(0)
    
    return (
        strategy_ret.loc[common_dates],
        benchmark_resampled.loc[common_dates],
        rf_resampled.reindex(common_dates, method='ffill').fillna(0)
    )


# ═══════════════════════════════════════════════════════════════════════
# Plot Functions
# ═══════════════════════════════════════════════════════════════════════

def plot_balance_breakdown(
    account_df: pd.DataFrame,
    resample_freq: str = '1D',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot balance breakdown by currency and type (total/free/locked).
    
    Args:
        account_df: Account DataFrame with columns ['total', 'free', 'locked', 'currency']
        resample_freq: Resampling frequency for time series
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'currency' not in account_df.columns:
        logger.warning("No 'currency' column found. Skipping balance breakdown plot.")
        plt.close(fig)
        return fig
    
    currencies = account_df['currency'].unique()
    balance_types = ['total', 'free', 'locked']
    
    colors = sns.color_palette('husl', n_colors=len(currencies))
    line_styles = ['-', '--', ':']
    line_widths = [2.5, 2, 1.5]
    
    for i, currency in enumerate(currencies):
        currency_data = account_df[account_df['currency'] == currency].copy()
        for j, bal_type in enumerate(balance_types):
            if bal_type in currency_data.columns:
                series = currency_data[bal_type].resample(resample_freq).last().ffill()
                ax.plot(
                    series.index, series.values,
                    label=f'{currency} {bal_type}',
                    color=colors[i],
                    linestyle=line_styles[j],
                    linewidth=line_widths[j],
                    alpha=0.8
                )
    
    ax.set_title('Portfolio Balance over time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Balance')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_cumulative_returns(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot cumulative returns comparison between strategy and benchmark.
    
    Args:
        strategy_ret: Strategy returns series
        benchmark_ret: Benchmark returns series (aligned)
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    strategy_cum = (1 + strategy_ret).cumprod() * 100 - 100
    benchmark_cum = (1 + benchmark_ret).cumprod() * 100 - 100
    
    sns.lineplot(x=strategy_cum.index, y=strategy_cum.values, 
                 label='Strategy', linewidth=2, ax=ax)
    sns.lineplot(x=benchmark_cum.index, y=benchmark_cum.values, 
                 label='Benchmark', linewidth=2, alpha=0.7, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns (%)')
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_rolling_sharpe(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    rf_ret: pd.Series,
    window: int,
    annualization_factor: float,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio comparison.
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (aligned)
        rf_ret: Risk-free rate (aligned)
        window: Rolling window size
        annualization_factor: Factor for annualization
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    rolling_sharpe_s = (
        (strategy_ret.rolling(window).mean() - rf_ret.rolling(window).mean()) /
        strategy_ret.rolling(window).std() * annualization_factor
    )
    rolling_sharpe_b = (
        (benchmark_ret.rolling(window).mean() - rf_ret.rolling(window).mean()) /
        benchmark_ret.rolling(window).std() * annualization_factor
    )
    
    sns.lineplot(x=rolling_sharpe_s.index, y=rolling_sharpe_s.values,
                 label='Strategy Sharpe', linewidth=2, ax=ax)
    sns.lineplot(x=rolling_sharpe_b.index, y=rolling_sharpe_b.values,
                 label='Benchmark Sharpe', linewidth=2, alpha=0.7, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_title(f'Rolling Sharpe Ratio ({window}-period)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio (Annualized)')
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_risk_free_rate(
    rf_ret: pd.Series,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot cumulative risk-free rate.
    
    Args:
        rf_ret: Risk-free returns (aligned)
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    rf_cum = (1 + rf_ret).cumprod() * 100 - 100
    sns.lineplot(x=rf_cum.index, y=rf_cum.values, 
                 label='Risk-Free Rate', linewidth=2, ax=ax)
    
    ax.set_title('Risk-Free Rate (Cumulative)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_period_returns(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    agg_freq: str,
    period_label: str,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot aggregated period returns (weekly/monthly) comparison.
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (aligned)
        agg_freq: Aggregation frequency ('W', 'ME', etc.)
        period_label: Label for period type ('Weekly', 'Monthly')
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    # Aggregate returns
    strategy_agg = strategy_ret.resample(agg_freq).apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ) * 100
    benchmark_agg = benchmark_ret.resample(agg_freq).apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    ) * 100
    
    # Remove zero periods
    #strategy_agg = strategy_agg[strategy_agg != 0]
    #benchmark_agg = benchmark_agg[benchmark_agg != 0]
    
    if len(strategy_agg) == 0:
        logger.warning("No non-zero period returns to plot")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(strategy_agg))
    width = 0.35
    
    ax.bar(x - width/2, strategy_agg.values, width, 
           label='Strategy', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, benchmark_agg.values, width, 
           label='Benchmark', alpha=0.8, color='orange')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_title(f'{period_label} Returns Comparison (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Period Ending')
    ax.set_ylabel('Return (%)')
    
    # Smart x-axis labeling
    n_bars = len(strategy_agg)
    tick_spacing = max(1, n_bars // 10)
    tick_indices = x[::tick_spacing]
    tick_labels = [strategy_agg.index[i].strftime('%Y-%m-%d') for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_returns_distribution(
    strategy_ret: pd.Series,
    agg_freq: str,
    period_label: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot distribution of period returns.
    
    Args:
        strategy_ret: Strategy returns
        agg_freq: Aggregation frequency
        period_label: Label for period type
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    strategy_agg = strategy_ret.resample(agg_freq).apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan
    ) * 100
    strategy_agg = strategy_agg.dropna()
    
    if len(strategy_agg) < 2:
        logger.warning("Insufficient data for returns distribution")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    n_bins = min(20, max(5, len(strategy_agg) // 2))
    
    ax.hist(strategy_agg.values, bins=n_bins, alpha=0.7, 
            edgecolor='black', color='steelblue')
    ax.axvline(x=strategy_agg.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {strategy_agg.mean():.2f}%')
    ax.axvline(x=strategy_agg.median(), color='blue', linestyle='--',
               linewidth=2, label=f'Median: {strategy_agg.median():.2f}%')
    
    ax.set_title(f'Distribution of {period_label} Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{period_label} Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_active_returns(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    freq: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot active returns (Strategy - Benchmark).
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (aligned)
        freq: Frequency label
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    active_returns = (strategy_ret - benchmark_ret) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(active_returns.index, active_returns.values, 
            linewidth=1.5, alpha=0.8, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(active_returns.index, active_returns.values, 0,
                    where=(active_returns.values > 0), alpha=0.3, 
                    color='green', label='Outperformance')
    ax.fill_between(active_returns.index, active_returns.values, 0,
                    where=(active_returns.values <= 0), alpha=0.3, 
                    color='red', label='Underperformance')
    
    ax.set_title(f'Active Returns (%) - {freq} frequency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Active Return (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_active_returns_heatmap(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot active returns heatmap (Month x Year).
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (aligned)
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    active_returns = (strategy_ret - benchmark_ret) * 100
    active_monthly = active_returns.to_frame('active_ret')
    active_monthly['year'] = active_monthly.index.year
    active_monthly['month'] = active_monthly.index.month
    heatmap_data = active_monthly.groupby(['year', 'month'])['active_ret'].mean().unstack()
    
    if heatmap_data.empty or heatmap_data.shape[0] == 0:
        logger.warning("Insufficient data for heatmap")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = np.nanmax(np.abs(heatmap_data.values))
    vmin = -vmax if vmax > 0 else -1
    vmax = vmax if vmax > 0 else 1
    
    im = ax.imshow(heatmap_data.T.values, aspect='auto', cmap='RdYlGn', 
                   vmin=vmin, vmax=vmax)
    
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.T.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.T.index)
    
    # Annotate cells
    threshold = vmax * 0.5
    for i in range(len(heatmap_data.T.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.T.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > threshold else "black"
                ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Active Return (%)')
    ax.set_title('Active Returns Heatmap (Month x Year)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_rolling_ratios(
    strategy_ret: pd.Series,
    benchmark_ret: pd.Series,
    rf_ret: pd.Series,
    window: int,
    periods_per_year: float,
    annualization_factor: float,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot rolling risk metrics (Sharpe, Sortino, M²).
    
    Args:
        strategy_ret: Strategy returns
        benchmark_ret: Benchmark returns (aligned)
        rf_ret: Risk-free rate (aligned)
        window: Rolling window size
        periods_per_year: Periods per year for annualization
        annualization_factor: Factor for annualization
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Sharpe
    rolling_sharpe = (
        (strategy_ret.rolling(window).mean() - rf_ret.rolling(window).mean()) /
        strategy_ret.rolling(window).std() * annualization_factor
    )
    axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='steelblue')
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title(f'Rolling Sharpe Ratio ({window}-period)', fontweight='bold')
    axes[0].set_ylabel('Sharpe (Annualized)')
    axes[0].grid(True, alpha=0.3)
    
    # Sortino
    def downside_std(returns):
        downside = returns[returns < 0]
        return np.sqrt(np.mean(downside**2)) if len(downside) > 0 else np.nan
    
    rolling_sortino = (
        (strategy_ret.rolling(window).mean() - rf_ret.rolling(window).mean()) /
        strategy_ret.rolling(window).apply(downside_std) * annualization_factor
    )
    axes[1].plot(rolling_sortino.index, rolling_sortino.values, linewidth=2, color='orange')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title(f'Rolling Sortino Ratio ({window}-period)', fontweight='bold')
    axes[1].set_ylabel('Sortino (Annualized)')
    axes[1].grid(True, alpha=0.3)
    
    # M²
    rolling_port_vol = strategy_ret.rolling(window).std() * annualization_factor
    rolling_bench_vol = benchmark_ret.rolling(window).std() * annualization_factor
    rolling_port_ret = strategy_ret.rolling(window).mean() * periods_per_year
    rolling_rf_annual = rf_ret.rolling(window).mean() * periods_per_year
    
    rolling_m2 = np.where(
        rolling_port_vol > 0,
        (rolling_port_ret - rolling_rf_annual) * (rolling_bench_vol / rolling_port_vol) + rolling_rf_annual,
        np.nan
    )
    rolling_m2 = pd.Series(rolling_m2, index=strategy_ret.index)
    
    axes[2].plot(rolling_m2.index, rolling_m2.values * 100, linewidth=2, color='green')
    axes[2].set_title(f'Rolling Modigliani M² ({window}-period)', fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('M² (% Annualized)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_underwater(
    strategy_ret: pd.Series,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot underwater chart (drawdown from peak).
    
    Args:
        strategy_ret: Strategy returns
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    cumulative = (1 + strategy_ret).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax.plot(drawdown.index, drawdown.values, linewidth=2, color='darkred')
    ax.set_title('Underwater Plot (Drawdown from Peak)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(drawdown.min() * 1.1, -1))
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_portfolio_allocation(
    positions_df: pd.DataFrame,
    resample_freq: str,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot portfolio allocation changes over time.
    
    Args:
        positions_df: Positions DataFrame with 'instrument_id', 'quantity', 'ts_event'
        resample_freq: Resampling frequency
        figsize: Figure size
        log_mlflow: Whether to log to MLflow
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure object
    """
    required_cols = ['instrument_id', 'quantity', 'ts_init']
    if not all(col in positions_df.columns for col in required_cols):
        logger.warning("Missing required columns for portfolio allocation plot")
        return plt.figure()
    
    pos_df = positions_df.copy()
    pos_df['date'] = pd.to_datetime(pos_df['ts_init'], unit='ns', utc=True)
    pos_df = pos_df.set_index('date').sort_index()
    
    allocation = pos_df.groupby([pd.Grouper(freq=resample_freq), 'instrument_id'])['quantity'].sum().unstack(fill_value=0)
    
    if allocation.empty or len(allocation) < 2:
        logger.warning("Insufficient position data for allocation chart")
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=figsize)
    allocation.plot(kind='area', stacked=True, ax=ax, alpha=0.7, linewidth=0)
    ax.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig