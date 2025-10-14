#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

# Lista naszych aktywów
assets = ['SPY', 'QQQ', 'VWO', 'BND', 'IEF']

print("Poprawione opłaty zarządcze (prawdopodobnie w skali 0-1):")
print("="*60)

expense_ratios = {}

for asset in assets:
    try:
        ticker = yf.Ticker(asset)
        info = ticker.info
        
        # Pobierz expense ratio
        raw_expense_ratio = info.get('netExpenseRatio', None)
        name = info.get('longName', asset)
        
        if raw_expense_ratio is not None:
            # Prawdopodobnie trzeba podzielić przez 100
            expense_ratio = raw_expense_ratio / 100
            expense_ratios[asset] = expense_ratio
            print(f"{asset}: {name}")
            print(f"  Surowa wartość: {raw_expense_ratio}")
            print(f"  Opłata zarządcza: {expense_ratio:.4f} ({expense_ratio*100:.3f}%)")
        else:
            # Użyj znanych wartości jako fallback
            known_ratios = {
                'SPY': 0.000945,  # 0.0945%
                'QQQ': 0.0020,    # 0.20%
                'VWO': 0.0010,    # 0.10%
                'BND': 0.0003,    # 0.03%
                'IEF': 0.0015     # 0.15%
            }
            expense_ratios[asset] = known_ratios[asset]
            print(f"{asset}: {name}")
            print(f"  Opłata zarządcza: {known_ratios[asset]:.4f} ({known_ratios[asset]*100:.3f}%) [znana wartość]")
        
        print()
        
    except Exception as e:
        print(f"Błąd dla {asset}: {e}")
        print()

print("="*60)
print("Średnia opłata zarządcza:")
avg_expense = sum(expense_ratios.values()) / len(expense_ratios)
print(f"Średnia: {avg_expense:.4f} ({avg_expense*100:.3f}%)")

print("\nPorównanie z naszymi ustawieniami:")
print(f"Używaliśmy: 0.5% (0.005)")
print(f"Rzeczywiste: {avg_expense*100:.3f}%")
print(f"Różnica: {(0.005 - avg_expense)*100:.3f}%")

# Sprawdźmy też inne możliwe pola
print("\n" + "="*60)
print("Sprawdzenie innych pól dla SPY:")
ticker = yf.Ticker('SPY')
info = ticker.info

possible_fee_fields = ['expenseRatio', 'annualReportExpenseRatio', 'totalAssets', 'yield']
for field in possible_fee_fields:
    if field in info:
        print(f"{field}: {info[field]}")
