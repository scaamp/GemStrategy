#!/usr/bin/env python3

# Test z rzeczywistymi opłatami zarządczymi
from gem import GEMStrategy

print("Test DCA z rzeczywistymi opłatami zarządczymi:")
print("="*60)

# Test DCA z automatycznymi opłatami
gem_dca_real = GEMStrategy(
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
    management_fee='auto',  # Automatyczne pobieranie z Yahoo Finance
    risk_free_rate=0.00
)

gem_dca_real.run_backtest()

print("\n" + "="*60)
print("Test DCA bez opłat zarządczych (dla porównania):")
print("="*60)

# Test DCA bez opłat
gem_dca_no_fees = GEMStrategy(
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
    management_fee=0.0,  # Bez opłat
    risk_free_rate=0.00
)

gem_dca_no_fees.run_backtest()

# Porównaj wyniki
print("\n" + "="*60)
print("PORÓWNANIE WYNIKÓW:")
print("="*60)

real_metrics = gem_dca_real.calculate_metrics()
no_fees_metrics = gem_dca_no_fees.calculate_metrics()

print(f"Z rzeczywistymi opłatami:")
print(f"  Wartość końcowa: ${real_metrics['Final Value']:,.2f}")
print(f"  CAGR: {real_metrics['CAGR (%)']:.2f}%")
print(f"  Total Return: {real_metrics['Total Return (%)']:.2f}%")

print(f"\nBez opłat zarządczych:")
print(f"  Wartość końcowa: ${no_fees_metrics['Final Value']:,.2f}")
print(f"  CAGR: {no_fees_metrics['CAGR (%)']:.2f}%")
print(f"  Total Return: {no_fees_metrics['Total Return (%)']:.2f}%")

print(f"\nRóżnica:")
print(f"  Wartość: ${real_metrics['Final Value'] - no_fees_metrics['Final Value']:,.2f}")
print(f"  CAGR: {real_metrics['CAGR (%)'] - no_fees_metrics['CAGR (%)']:.2f}%")
