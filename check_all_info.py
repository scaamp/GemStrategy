#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

# Sprawdźmy wszystkie dostępne informacje dla SPY
print("Sprawdzanie wszystkich dostępnych informacji dla SPY:")
print("="*60)

ticker = yf.Ticker('SPY')
info = ticker.info

# Znajdź pola związane z opłatami
fee_related_keys = [key for key in info.keys() if any(word in key.lower() for word in ['fee', 'expense', 'ratio', 'cost', 'management'])]

print("Pola związane z opłatami:")
for key in fee_related_keys:
    print(f"  {key}: {info[key]}")

print("\n" + "="*60)
print("Wszystkie dostępne klucze (pierwsze 20):")
for i, key in enumerate(list(info.keys())[:20]):
    print(f"  {key}: {info[key]}")

print("\n" + "="*60)
print("Znane opłaty zarządcze (z innych źródeł):")
print("SPY (SPDR S&P 500): ~0.0945% (0.000945)")
print("QQQ (Invesco QQQ): ~0.20% (0.0020)")
print("VWO (Vanguard EM): ~0.10% (0.0010)")
print("BND (Vanguard Total Bond): ~0.03% (0.0003)")
print("IEF (iShares 7-10Y Treasury): ~0.15% (0.0015)")
