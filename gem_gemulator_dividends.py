
import sys
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Ustaw interaktywny backend dla matplotlib (możesz zmienić na nieinteraktywny, jeśli wolisz)
plt.ion()

class GEMStrategy:
    """
    Strategia Global Equity Momentum (GEM) z pełnym uwzględnieniem dywidend:
    - Preferuje 'Adj Close' (total return: dywidendy + splity).
    - Gdy brak 'Adj Close', rekonstruuje indeks total return z 'Close' + 'Dividends'.
    """

    def __init__(self, 
                 risky_assets=['QQQ'],
                 safe_assets=['SHY'],
                 start_date='2000-01-01',
                 end_date=None,
                 lookback_period=12,
                 gap_period=1,
                 rebalance_frequency=1,  # w miesiącach (1=M, 3=Q, 12=A)
                 initial_capital=10_000,
                 monthly_contribution=0,
                 investment_strategy='lump_sum',
                 transaction_cost=0.001,
                 risk_free_asset='SHY',
                 top_k=1,
                 volatility_weighted=False):
        self.risky_assets = risky_assets
        self.safe_assets = safe_assets
        self.all_assets = risky_assets + safe_assets
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.lookback_period = lookback_period
        self.gap_period = gap_period
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.investment_strategy = investment_strategy
        self.transaction_cost = transaction_cost
        self.risk_free_asset = risk_free_asset
        self.top_k = min(top_k, len(risky_assets))
        self.volatility_weighted = volatility_weighted

        self.prices = None
        self.returns = None
        self.portfolio_value = None
        self.holdings = None
        self.shares_history = None
        self.rf_prices = None
        self.weights = None

    # ---------------------- DATA HELPERS ----------------------
    @staticmethod
    def _build_total_return_from_close_and_div(close: pd.Series, div: pd.Series) -> pd.Series:
        close = close.sort_index().dropna()
        div = div.sort_index().reindex(close.index).fillna(0.0)
        prev_close = close.shift(1)
        factor = (close + div) / prev_close
        factor.iloc[0] = 1.0
        tr = close.iloc[0] * factor.cumprod()
        return tr

    @staticmethod
    def _select_adj_close_or_reconstruct(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            tr_frames = []
            for t in tickers:
                tr_series = None
                # 1) Adj Close (układ 1)
                if ('Adj Close', t) in raw.columns:
                    tr_series = raw[('Adj Close', t)].dropna().rename(t)
                # 2) Adj Close (układ 2)
                elif (t, 'Adj Close') in raw.columns:
                    tr_series = raw[(t, 'Adj Close')].dropna().rename(t)
                else:
                    # Rekonstrukcja z Close + Dividends
                    close = None; div = None
                    if ('Close', t) in raw.columns:
                        close = raw[('Close', t)]
                    elif (t, 'Close') in raw.columns:
                        close = raw[(t, 'Close')]
                    if ('Dividends', t) in raw.columns:
                        div = raw[('Dividends', t)]
                    elif (t, 'Dividends') in raw.columns:
                        div = raw[(t, 'Dividends')]
                    if close is None:
                        raise KeyError(f"Brak 'Adj Close' i 'Close' dla {t}.")
                    if div is None:
                        div = pd.Series(0.0, index=close.index)
                    tr_series = GEMStrategy._build_total_return_from_close_and_div(close, div).rename(t)
                tr_frames.append(tr_series)
            return pd.concat(tr_frames, axis=1).sort_index()
        else:
            cols = raw.columns
            if 'Adj Close' in cols:
                s = raw['Adj Close'].dropna()
                return s.to_frame(name=tickers[0])
            else:
                close = raw['Close'].dropna() if 'Close' in cols else None
                div = raw['Dividends'] if 'Dividends' in cols else None
                if close is None:
                    raise KeyError("Brak 'Adj Close' i 'Close' w danych jednego tickera.")
                if div is None:
                    div = pd.Series(0.0, index=close.index)
                tr = GEMStrategy._build_total_return_from_close_and_div(close, div)
                return tr.to_frame(name=tickers[0])

    def _download_total_return_prices(self, tickers: list) -> pd.DataFrame:
        data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False, group_by=False, auto_adjust=False)
        if data is None or data.empty:
            raise ValueError("Puste dane z Yahoo Finance.")
        tr_df = self._select_adj_close_or_reconstruct(data, tickers)
        return tr_df.dropna(how='all')

    def download_data(self):
        print("Pobieranie danych (Total Return: Adj Close lub Close+Dividends)...")
        all_tickers = list(dict.fromkeys(self.all_assets + [self.risk_free_asset]))
        tr_all = self._download_total_return_prices(all_tickers)
        cols = [c for c in tr_all.columns if c in self.all_assets]
        if not cols:
            raise ValueError("Brak danych TR dla aktywów strategii.")
        self.prices = tr_all[cols].copy()
        self.rf_prices = tr_all[self.risk_free_asset].copy() if self.risk_free_asset in tr_all.columns else self.prices[self.safe_assets[0]].copy()
        self.returns = self.prices.pct_change()
        print(f"Zakres danych: {self.prices.index.min().date()} → {self.prices.index.max().date()} ({len(self.prices)} sesji)")

    # ---------------------- ANALYTICS ----------------------
    def calculate_volatility(self, date, asset, window=12):
        start_date = date - pd.DateOffset(months=window)
        series = self.prices.loc[start_date:date, asset].dropna()
        monthly = series.resample('M').last().dropna()
        if len(monthly) < max(2, int(0.75 * window)):
            return np.nan
        log_returns = np.log(monthly / monthly.shift(1)).dropna()
        if len(log_returns) < 2:
            return np.nan
        return float(log_returns.std() * np.sqrt(12))

    def calculate_momentum(self, date, asset):
        asset_data = self.prices[asset]
        if asset_data.first_valid_index() is None or asset_data.first_valid_index() > date:
            return np.nan
        L = self.lookback_period; K = self.gap_period
        end_date = date - pd.DateOffset(months=K)
        start_date = end_date - pd.DateOffset(months=L)
        if asset_data.first_valid_index() > start_date:
            return np.nan
        monthly = asset_data.resample('M').last()
        window = monthly.loc[start_date:end_date].dropna()
        if len(window) < max(2, int(0.75 * L)):
            return np.nan
        return float(window.iloc[-1] / window.iloc[0] - 1.0)

    def _select_safe_asset(self, date, current_asset=None):
        return self.safe_assets[0]

    def select_asset(self, date, current_asset=None):
        risky_momentum = {a: self.calculate_momentum(date, a) for a in self.risky_assets}
        valid = {k: v for k, v in risky_momentum.items() if pd.notna(v)}
        if not valid:
            return self._select_safe_asset(date, current_asset)
        rf_mom = self.calculate_momentum(date, self.risk_free_asset)
        sorted_assets = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for a, m in sorted_assets:
            if pd.isna(rf_mom) or m > rf_mom:
                selected.append(a)
            if len(selected) >= self.top_k:
                break
        if not selected:
            return self._select_safe_asset(date, current_asset)
        if len(selected) == 1 or not self.volatility_weighted:
            self.weights = {a: 1.0 / len(selected) for a in selected}
            return selected[0]
        inv_vol = {}
        for a in selected:
            vol = self.calculate_volatility(date, a)
            inv_vol[a] = 1.0 / vol if pd.notna(vol) and vol > 0 else 0.0
        s = sum(inv_vol.values())
        self.weights = {a: (inv_vol[a] / s if s > 0 else 1.0 / len(selected)) for a in selected}
        return max(self.weights.items(), key=lambda x: x[1])[0]

    # ---------------------- BACKTEST ----------------------
    def run_backtest(self):
        if self.prices is None:
            self.download_data()
        print("\nBacktest GEM (Total Return ceny)...")
        monthly_prices = self.prices.resample('M').last().dropna(how='all')
        monthly_dates = monthly_prices.index
        current_asset = self.safe_assets[0]
        shares = {a: 0.0 for a in self.all_assets}
        first_price = monthly_prices.loc[monthly_dates[0], current_asset]
        shares[current_asset] = (self.initial_capital / first_price) * (1 - self.transaction_cost)
        portfolio_value = []; holdings_history = []; dates = []; shares_history = []
        for i, date in enumerate(monthly_dates):
            if self.investment_strategy == 'dca' and i > 0 and self.monthly_contribution > 0:
                px = monthly_prices.loc[date, current_asset]
                shares[current_asset] += (self.monthly_contribution / px) * (1 - self.transaction_cost)
            if i % self.rebalance_frequency == 0 and i >= self.lookback_period:
                new_asset = self.select_asset(date, current_asset)
                if new_asset != current_asset:
                    cur_px = monthly_prices.loc[date, current_asset]
                    total_val = shares[current_asset] * cur_px * (1 - self.transaction_cost)
                    new_px = monthly_prices.loc[date, new_asset]
                    shares[current_asset] = 0.0
                    shares[new_asset] = (total_val / new_px) * (1 - self.transaction_cost)
                    current_asset = new_asset
            px = monthly_prices.loc[date, current_asset]
            val = shares[current_asset] * px
            portfolio_value.append(val); holdings_history.append(current_asset); dates.append(date); shares_history.append(shares[current_asset])
        self.portfolio_value = pd.Series(portfolio_value, index=pd.DatetimeIndex(dates))
        self.holdings = pd.Series(holdings_history, index=pd.DatetimeIndex(dates))
        self.shares_history = pd.Series(shares_history, index=pd.DatetimeIndex(dates))
        print("Backtest zakończony.")

    # ---------------------- METRICS ----------------------
    def calculate_metrics(self):
        if self.portfolio_value is None:
            raise ValueError("Najpierw uruchom backtest!")
        if self.investment_strategy == 'dca':
            total_months = len(self.portfolio_value)
            total_contributions = self.initial_capital + (total_months - 1) * self.monthly_contribution
        else:
            total_contributions = self.initial_capital
        rets = self.portfolio_value.pct_change().dropna()
        total_return = (self.portfolio_value.iloc[-1] / total_contributions - 1) * 100
        years = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days / 365.25
        cagr = (np.power(self.portfolio_value.iloc[-1] / total_contributions, 1 / years) - 1) * 100 if years > 0 else np.nan
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        rf_monthly = self.rf_prices.resample('M').last()
        rf_returns = rf_monthly.pct_change().reindex(rets.index)
        excess = rets - rf_returns
        sharpe = np.sqrt(12) * excess.mean() / excess.std() if len(excess) > 1 and excess.std() > 0 else 0.0
        vol = rets.std() * np.sqrt(12) * 100 if len(rets) > 1 else 0.0
        return {
            'Total Return (%)': float(total_return),
            'CAGR (%)': float(cagr),
            'Max Drawdown (%)': float(max_drawdown),
            'Sharpe Ratio': float(sharpe),
            'Volatility (%)': float(vol),
            'Final Value': float(self.portfolio_value.iloc[-1]),
            'Total Contributions': float(total_contributions),
        }

    def _download_benchmark_tr(self, ticker: str) -> pd.Series:
        raw = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, group_by=False, auto_adjust=False)
        if raw is None or raw.empty:
            raise ValueError(f"Brak danych benchmarku {ticker}.")
        tr_df = self._select_adj_close_or_reconstruct(raw, [ticker])
        return tr_df.iloc[:, 0].rename(ticker)

    def calculate_buyhold_benchmark(self, benchmark_ticker='QQQ'):
        bench_tr = self._download_benchmark_tr(benchmark_ticker)
        if self.investment_strategy == 'dca':
            total_months = len(self.portfolio_value)
            total_contributions = self.initial_capital + (total_months - 1) * self.monthly_contribution
        else:
            total_contributions = self.initial_capital
        if self.investment_strategy == 'dca':
            monthly_dates = self.portfolio_value.index
            shares = 0.0; values = []; idx = []
            for i, dt_i in enumerate(monthly_dates):
                contrib = self.initial_capital if i == 0 else self.monthly_contribution
                price = bench_tr.loc[:dt_i].iloc[-1]
                shares += (contrib / price) * (1 - self.transaction_cost)
                values.append(shares * price); idx.append(dt_i)
            buyhold_value = pd.Series(values, index=pd.DatetimeIndex(idx))
        else:
            start_date = self.portfolio_value.index[0]
            start_price = bench_tr.loc[:start_date].iloc[-1]
            shares = (self.initial_capital / start_price) * (1 - self.transaction_cost)
            prices_on_months = bench_tr.reindex(self.portfolio_value.index, method='ffill')
            buyhold_value = shares * prices_on_months
        rets = buyhold_value.pct_change().dropna()
        total_return = (buyhold_value.iloc[-1] / total_contributions - 1) * 100
        years = (buyhold_value.index[-1] - buyhold_value.index[0]).days / 365.25
        cagr = (np.power(buyhold_value.iloc[-1] / total_contributions, 1 / years) - 1) * 100 if years > 0 else np.nan
        cummax = buyhold_value.cummax()
        drawdown = (buyhold_value - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        rf_monthly = self.rf_prices.resample('M').last()
        rf_returns = rf_monthly.pct_change().reindex(rets.index)
        excess = rets - rf_returns
        sharpe = np.sqrt(12) * excess.mean() / excess.std() if len(excess) > 1 and excess.std() > 0 else 0.0
        vol = rets.std() * np.sqrt(12) * 100 if len(rets) > 1 else 0.0
        metrics = {
            'Total Return (%)': float(total_return),
            'CAGR (%)': float(cagr),
            'Max Drawdown (%)': float(max_drawdown),
            'Sharpe Ratio': float(sharpe),
            'Volatility (%)': float(vol),
            'Final Value': float(buyhold_value.iloc[-1]),
            'Total Contributions': float(total_contributions),
        }
        return buyhold_value, metrics

    def plot_results(self, benchmark_ticker='SPY'):
        if self.portfolio_value is None:
            raise ValueError("Najpierw uruchom backtest!")
        gem_metrics = self.calculate_metrics()
        buyhold_value, buyhold_metrics = self.calculate_buyhold_benchmark(benchmark_ticker)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('GEM (Total Return) vs Buy & Hold', fontsize=16, fontweight='bold')
        ax1 = axes[0, 0]
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, label='GEM Strategy', linewidth=2)
        ax1.plot(buyhold_value.index, buyhold_value.values, label=f'Buy & Hold ({benchmark_ticker})', linewidth=2, linestyle='--')
        ax1.set_title('Skumulowana Wartość Portfela'); ax1.set_xlabel('Data'); ax1.set_ylabel('Wartość ($)')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        metrics_to_plot = ['CAGR (%)', 'Max Drawdown (%)']
        gem_vals = [gem_metrics[m] for m in metrics_to_plot]
        bh_vals = [buyhold_metrics[m] for m in metrics_to_plot]
        x = np.arange(len(metrics_to_plot)); w = 0.35
        ax2.bar(x - w/2, gem_vals, w, label='GEM')
        ax2.bar(x + w/2, bh_vals, w, label=f'Buy&Hold {benchmark_ticker}')
        ax2.set_title('CAGR i Max Drawdown'); ax2.set_xticks(x); ax2.set_xticklabels(metrics_to_plot)
        ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)
        ax3 = axes[1, 0]
        asset_mapping = {asset: i for i, asset in enumerate(self.all_assets)}
        holdings_numeric = self.holdings.map(asset_mapping)
        for i, asset in enumerate(self.all_assets):
            mask = holdings_numeric == i
            if mask.any():
                dts = holdings_numeric[mask].index
                vals = [i] * len(dts)
                ax3.scatter(dts, vals, s=50, alpha=0.8, label=asset)
        ax3.set_title('Timeline Aktywów'); ax3.set_xlabel('Data'); ax3.set_ylabel('Aktywo')
        ax3.set_yticks(range(len(self.all_assets))); ax3.set_yticklabels(self.all_assets)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]; ax4.axis('off')
        table_data = [
            ['Metryka', 'GEM', f'Buy & Hold ({benchmark_ticker})'],
            ['Total Return (%)', f'{gem_metrics["Total Return (%)"]:.2f}', f'{buyhold_metrics["Total Return (%)"]:.2f}'],
            ['CAGR (%)', f'{gem_metrics["CAGR (%)"]:.2f}', f'{buyhold_metrics["CAGR (%)"]:.2f}'],
            ['Max Drawdown (%)', f'{gem_metrics["Max Drawdown (%)"]:.2f}', f'{buyhold_metrics["Max Drawdown (%)"]:.2f}'],
            ['Sharpe Ratio', f'{gem_metrics["Sharpe Ratio"]:.2f}', f'{buyhold_metrics["Sharpe Ratio"]:.2f}'],
            ['Volatility (%)', f'{gem_metrics["Volatility (%)"]:.2f}', f'{buyhold_metrics["Volatility (%)"]:.2f}'],
            ['Final Value ($)', f'{gem_metrics["Final Value"]:,.2f}', f'{buyhold_metrics["Final Value"]:,.2f}'],
        ]
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.4)
        plt.tight_layout(); plt.show()

def run_gem_strategy(strategy_type='dca', plot_type=1, top_k=1, volatility_weighted=False, benchmark='QQQ'):
    print("="*60)
    print("STRATEGIA GEM - GLOBAL EQUITY MOMENTUM (Total Return)")
    print("="*60)
    if strategy_type == 'dca':
        investment_strategy = 'dca'; initial_capital = 10_000; monthly_contribution = 1_000
        print(f"\nDCA: start ${initial_capital:,}, dopłaty ${monthly_contribution:,}/mies.")
    else:
        investment_strategy = 'lump_sum'; initial_capital = 10_000; monthly_contribution = 0
        print(f"\nLump Sum: start ${initial_capital:,}")
    gem = GEMStrategy(
        start_date='2003-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d'),
        lookback_period=12,
        gap_period=1,
        rebalance_frequency=1,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        investment_strategy=investment_strategy,
        transaction_cost=0.001,
        top_k=top_k,
        volatility_weighted=volatility_weighted
    )
    gem.run_backtest()
    gem.plot_results(benchmark_ticker=benchmark)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Strategia GEM - Global Equity Momentum (Total Return)')
    parser.add_argument('--strategy', choices=['dca', 'lump_sum'], default='dca')
    parser.add_argument('--plot', choices=['1', '2'], default='1')
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--volatility-weighted', action='store_true')
    parser.add_argument('--benchmark', choices=['SPY', 'QQQ'], default='QQQ')
    args = parser.parse_args()
    run_gem_strategy(args.strategy, int(args.plot), args.top_k, args.volatility_weighted, args.benchmark)
