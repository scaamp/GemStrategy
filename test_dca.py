#!/usr/bin/env python3

# Test DCA dla GEM strategy
from gem import GEMStrategy

# Test DCA
print("Test DCA Strategy:")
gem_dca = GEMStrategy(
    risky_assets=['SPY', 'QQQ', 'VWO'],
    safe_assets=['BND', 'IEF'],
    start_date='2007-01-01',
    end_date='2024-12-31',
    lookback_period=12,
    gap_period=1,
    rebalance_frequency=1,
    initial_capital=1000,  # Pierwsza wpłata
    monthly_contribution=1000,  # Miesięczne wpłaty
    investment_strategy='dca',
    transaction_cost=0.001,
    management_fee=0.005,  # 0.5% rocznie
    risk_free_rate=0.00
)

gem_dca.run_backtest()

# Test Lump Sum dla porównania
print("\n" + "="*50)
print("Test Lump Sum Strategy:")
gem_lump = GEMStrategy(
    risky_assets=['SPY', 'QQQ', 'VWO'],
    safe_assets=['BND', 'IEF'],
    start_date='2007-01-01',
    end_date='2024-12-31',
    lookback_period=12,
    gap_period=1,
    rebalance_frequency=1,
    initial_capital=10000,
    monthly_contribution=0,
    investment_strategy='lump_sum',
    transaction_cost=0.001,
    management_fee=0.005,  # 0.5% rocznie
    risk_free_rate=0.00
)

gem_lump.run_backtest()

# Porównaj wyniki
print("\n" + "="*50)
print("PORÓWNANIE WYNIKÓW:")
print("="*50)

dca_metrics = gem_dca.calculate_metrics()
lump_metrics = gem_lump.calculate_metrics()

print(f"DCA - Wartość końcowa: ${dca_metrics['Final Value']:,.2f}")
print(f"Lump Sum - Wartość końcowa: ${lump_metrics['Final Value']:,.2f}")
print(f"DCA - CAGR: {dca_metrics['CAGR (%)']:.2f}%")
print(f"Lump Sum - CAGR: {lump_metrics['CAGR (%)']:.2f}%")
