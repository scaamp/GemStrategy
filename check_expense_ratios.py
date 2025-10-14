#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

# Lista naszych aktywów
assets = ['SPY', 'QQQ', 'VWO', 'BND', 'IEF']

print("Sprawdzanie rzeczywistych opłat zarządczych (Expense Ratios):")
print("="*60)

for asset in assets:
    try:
        ticker = yf.Ticker(asset)
        info = ticker.info
        
        # Pobierz expense ratio
        expense_ratio = info.get('expenseRatio', 'N/A')
        name = info.get('longName', asset)
        category = info.get('category', 'N/A')
        
        print(f"{asset}: {name}")
        print(f"  Kategoria: {category}")
        print(f"  Opłata zarządcza: {expense_ratio}")
        print()
        
    except Exception as e:
        print(f"Błąd dla {asset}: {e}")
        print()

print("="*60)
print("Uwagi:")
print("- Expense ratio jest podawany jako ułamek (np. 0.0009 = 0.09%)")
print("- Niektóre dane mogą być niedostępne dla starszych ETF-ów")
print("- Opłaty mogą się zmieniać w czasie")
