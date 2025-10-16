import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Ustaw interaktywny backend dla matplotlib
plt.ion()

class GEMStrategy:
    """
    Implementacja rozszerzonej strategii Global Equity Momentum (GEM).
    
    Strategia działa dwuetapowo:
    1. Relative Momentum: Wybiera aktywo o najwyższym momentum spośród:
       - Główne indeksy USA (SPY, RSP, QQQ, IWM)
       - Rynki międzynarodowe (EFA, VWO, EEM)
       - Alternatywne klasy aktywów (GLD, VNQ, BTC)
    
    2. Absolute Momentum: Porównuje momentum lidera ze stopą wolną od ryzyka (T-Bills)
       - Jeśli excess momentum > 0: inwestuje 100% w lidera
       - Jeśli excess momentum ≤ 0: przechodzi w 100% do obligacji (TLT/IEF/SHY)
       
    Uwagi:
    - Niektóre aktywa (np. BTC) mają krótszą historię - są uwzględniane tylko gdy 
      dostępne jest minimum 75% wymaganych danych w okresie lookback
    - Obligacje o różnej duracji (TLT, IEF, SHY) pozwalają dostosować się do 
      różnych środowisk stóp procentowych
    """

    """
    safe assets:
    IE00BYXPSP02 (IBTA - iShares $ Treasury Bond 1-3yr UCITS ETF USD (Acc))
    IE00B3VWN518 (iShares VII PLC - iShares $ Treasury Bd 7-10y ETF USD Acc)
    IE00BMFKG444 (XNAS - Xtrackers NASDAQ 100 UCITS ETF 1C)

    risky_assets:
    IE00BMFKG444 (XNAS - Xtrackers NASDAQ 100 UCITS ETF 1C)

    IE0006WW1TQ4 (EXUS - Xtrackers MSCI World ex USA UCITS ETF 1C USD).
    """
    
    def __init__(self, 
                 risky_assets=[
                     # Główne indeksy USA
                    #  'SPY',   # S&P 500
                     'QQQ',   # NASDAQ 100
                    #  'IWM',   # Russell 2000 (małe spółki)
                     
                     # Rynki międzynarodowe
                    #  'EFA',   # MSCI EAFE (rynki rozwinięte poza USA)
                     'EEM',   # MSCI Emerging Markets
                     
                     # Alternatywne klasy aktywów
                     'GLD',   # SPDR Gold Shares
                    #  'VNQ',   # Vanguard Real Estate ETF
                    #  'BTC-USD', # Bitcoin (dostępny od 2014)
                 ],
                 safe_assets=[
                    #  'TLT',   # 20+ Year Treasury Bond
                    #  'IEF',   # 7-10 Year Treasury Bond
                     'SHY',   # 1-3 Year Treasury Bond
                    #  'TBIL',   # 1-3 Year Treasury Bond
                 ],
                 start_date='2008-01-01',
                 end_date=None,
                 lookback_period=12,
                 gap_period=1,
                 rebalance_frequency=1,
                 initial_capital=10000,
                 monthly_contribution=1000,
                 investment_strategy='lump_sum',  # 'lump_sum' lub 'dca'
                 transaction_cost=0.001,  # 0.1% na pokrycie kosztów transakcyjnych i slippage
                 risk_free_asset='SHY',
                 top_k=1,  # Liczba najlepszych aktywów do portfela (1 = klasyczny GEM)
                 volatility_weighted=False,  # Czy ważyć aktywa odwrotnie do zmienności
                 rebalance_threshold=0.01):  # Margines w % - nowe aktywo musi być lepsze o co najmniej tyle
        """
        Parametry:
        ----------
        risky_assets : list
            Lista tickerów ryzykownych aktywów (akcje, ETF-y akcyjne)
        safe_assets : list
            Lista tickerów aktywów bezpiecznych (obligacje)
        start_date : str
            Data rozpoczęcia symulacji (format: 'YYYY-MM-DD')
        end_date : str
            Data zakończenia symulacji (domyślnie: dzisiejsza data)
        lookback_period : int
            Okres w miesiącach do obliczania momentum
        gap_period : int
            Liczba ostatnich miesięcy wykluczonych z obliczeń momentum
        rebalance_frequency : int
            Częstotliwość rebalansowania w miesiącach
        initial_capital : float
            Kapitał początkowy (dla lump_sum) lub pierwsza wpłata (dla DCA)
        monthly_contribution : float
            Miesięczna wpłata dla strategii DCA (np. 1000)
        investment_strategy : str
            'lump_sum' - jedna wpłata na początku, 'dca' - miesięczne wpłaty
        transaction_cost : float
            Koszt transakcyjny jako % wartości transakcji (np. 0.001 = 0.1%)
        risk_free_asset : str
            Ticker aktywa wolnego od ryzyka (np. 'SHY', 'TLT')
        top_k : int
            Liczba najlepszych aktywów do portfela (1 = klasyczny GEM)
        volatility_weighted : bool
            Czy ważyć aktywa odwrotnie do zmienności
        rebalance_threshold : float
            Margines rebalansowania jako % (np. 0.01 = 1%). Nowe aktywo musi być 
            lepsze o co najmniej tyle, aby nastąpiła zmiana. Zmniejsza liczbę transakcji.
        """
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
        self.top_k = min(top_k, len(risky_assets))  # Nie więcej niż liczba dostępnych aktywów
        self.volatility_weighted = volatility_weighted
        self.rebalance_threshold = rebalance_threshold
        
        self.prices = None
        self.returns = None
        self.portfolio_value = None
        self.holdings = None
        self.rf_prices = None  # Ceny aktywa wolnego od ryzyka
        self.weights = None  # Wagi aktywów w portfelu
    
        
    def download_data(self):
        """
        Pobiera skorygowane dane cenowe (total return, uwzględniające dywidendy) 
        dla wszystkich aktywów z Yahoo Finance.
        """
        print("Pobieranie danych historycznych...")
        
        # Pobierz wszystkie aktywa + risk_free_asset (może być różne od safe_assets)
        all_tickers = list(set(self.all_assets + [self.risk_free_asset]))
        
        try:
            # 💡 KLUCZOWA POPRAWKA: auto_adjust=True
            # To gwarantuje, że kolumna 'Close' będzie zawierać skorygowane ceny (Total Return).
            data = yf.download(
                all_tickers, 
                start=self.start_date, 
                end=self.end_date,
                auto_adjust=True,  
                progress=False
            )['Close'] # Po auto_adjust=True, skorygowane ceny są w kolumnie 'Close'

            if data.empty:
                raise ValueError("Pobrany DataFrame jest pusty. Sprawdź tickery i zakres dat.")

            # Sprawdzenie, czy wszystkie tickery zostały pobrane i usunięcie wierszy z NaN
            if isinstance(data, pd.Series):
                 # Przypadek dla jednego tickera
                 self.prices = data.to_frame(name=self.all_assets[0]).dropna()
                 self.rf_prices = data.to_frame(name=self.risk_free_asset).dropna()
            else:
                 # Pobierz aktywa ryzykowne i bezpieczne
                 self.prices = data[self.all_assets].dropna()
                 # Pobierz risk_free_asset (może być różne od safe_assets)
                 if self.risk_free_asset in data.columns:
                     self.rf_prices = data[self.risk_free_asset].dropna()
                 else:
                     # Jeśli risk_free_asset nie jest w danych, użyj pierwszego safe_asset
                     self.rf_prices = data[self.safe_assets[0]].dropna()
            
            # Wyrównaj indeksy czasowe
            common_index = self.prices.index.intersection(self.rf_prices.index)
            self.prices = self.prices.loc[common_index]
            self.rf_prices = self.rf_prices.loc[common_index]

            self.returns = self.prices.pct_change()
            
            print(f"✅ Pobrano skorygowane dane (Total Return) od {self.prices.index[0].date()} do {self.prices.index[-1].date()}")
            print(f"Liczba dni handlowych: {len(self.prices)}")

        except Exception as e:
            print(f"Wystąpił błąd podczas pobierania danych: {e}")
            raise
        
    def calculate_volatility(self, date, asset, window=12):
        """
        Oblicza roczną zmienność aktywa na podstawie miesięcznych zwrotów.
        
        Parametry:
        ----------
        date : datetime
            Data, na którą obliczamy zmienność
        asset : str
            Ticker aktywa
        window : int
            Liczba miesięcy do obliczenia zmienności
            
        Zwraca:
        -------
        float lub np.nan
            Roczna zmienność lub np.nan jeśli brak wystarczających danych
        """
        # Znajdź datę początkową dla okna zmienności
        start_date = date - pd.DateOffset(months=window)
        
        # Pobierz ceny w tym okresie
        prices = self.prices.loc[start_date:date, asset].dropna()
        
        # Wymagamy co najmniej 75% kompletnych danych
        min_required_points = int(0.75 * (window * 21))  # ~21 dni handlowych w miesiącu
        if len(prices) < min_required_points:
            return np.nan
        
        # Oblicz miesięczne logarytmiczne zwroty
        monthly_prices = prices.resample('M').last()
        log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
        
        # Oblicz roczną zmienność
        if len(log_returns) > 1:
            return log_returns.std() * np.sqrt(12)  # Annualizacja
        else:
            return np.nan
    
    def calculate_momentum(self, date, asset):
        """
        Oblicza momentum dla danego aktywa na określoną datę.
        
        Momentum = skumulowany zwrot za okres (lookback_period - gap_period) miesięcy
        
        Parametry:
        ----------
        date : datetime
            Data, na którą obliczamy momentum
        asset : str
            Ticker aktywa
            
        Zwraca:
        -------
        float lub np.nan
            Momentum aktywa lub np.nan jeśli brak wystarczających danych
        """
        # Sprawdź czy aktywo ma wystarczająco długą historię
        # Sprawdź czy to risk_free_asset (może nie być w self.prices)
        if asset == self.risk_free_asset:
            asset_data = self.rf_prices
        else:
            asset_data = self.prices[asset]
        if asset_data.first_valid_index() is None or asset_data.first_valid_index() > date:
            return np.nan
            
        # Znajdź datę początkową dla momentum
        lookback_months = self.lookback_period
        gap_months = self.gap_period
        
        # Data końcowa: gap_period miesięcy wstecz od bieżącej daty
        end_date = date - pd.DateOffset(months=gap_months)
        # Data początkowa: lookback_period miesięcy wstecz od daty końcowej
        start_date = end_date - pd.DateOffset(months=lookback_months)
        
        # Sprawdź czy mamy wystarczająco danych historycznych
        if asset_data.first_valid_index() > start_date:
            return np.nan
        
        # Pobierz ceny w tym okresie
        prices_period = asset_data.loc[start_date:end_date].dropna()
        
        # Wymagamy co najmniej 75% kompletnych danych w okresie
        min_required_points = int(0.75 * (self.lookback_period * 21))  # ~21 dni handlowych w miesiącu
        if len(prices_period) < min_required_points:
            return np.nan
        
        # Oblicz skumulowany zwrot
        momentum = (prices_period.iloc[-1] / prices_period.iloc[0]) - 1
        
        return momentum
    
    def select_asset(self, date, current_asset=None):
        """
        Wybiera aktywo do inwestycji według logiki GEM:
        1. Wybierz aktywo ryzykowne z najwyższym momentum (Relative Momentum)
        2. Jeśli jego momentum > 0, inwestuj w nie (Absolute Momentum)
        3. Jeśli momentum <= 0 lub brak ważnych danych, inwestuj w aktywo bezpieczne
           z najwyższym momentum lub zachowaj obecne aktywo
        
        Parametry:
        ----------
        date : datetime
            Data, na którą obliczamy momentum
        current_asset : str, optional
            Obecnie trzymane aktywo (używane jako fallback)
        """
        # Krok 1: Oblicz momentum dla wszystkich ryzykownych aktywów
        risky_momentum = {}
        for asset in self.risky_assets:
            mom = self.calculate_momentum(date, asset)
            risky_momentum[asset] = mom
        
        # Odfiltruj NaN i znajdź lidera
        valid_risky = {k: v for k, v in risky_momentum.items() if pd.notna(v)}
        
        if not valid_risky:
            # Brak ważnych danych dla aktywów ryzykownych
            if current_asset in self.risky_assets:
                # Zachowaj obecne aktywo ryzykowne
                return current_asset
            else:
                # Przejdź do aktywów bezpiecznych
                return self._select_safe_asset(date, current_asset)
        
        # Krok 2: Absolute Momentum - porównaj z T-Bills
        rf_momentum = self.calculate_momentum(date, self.risk_free_asset)
        
        # Znajdź top-k aktywów z dodatnim excess momentum
        selected_assets = []
        for asset, mom in sorted(valid_risky.items(), key=lambda x: x[1], reverse=True):
            if pd.notna(rf_momentum) and mom > rf_momentum:
                selected_assets.append(asset)
                if len(selected_assets) >= self.top_k:
                    break
        
        if not selected_assets:
            return self._select_safe_asset(date, current_asset)
        
        # Krok 3: Sprawdź margines rebalansowania
        best_asset = selected_assets[0]  # Najlepszy aktyw ryzykowny
        
        # Jeśli mamy obecne aktywo i to nie jest aktyw bezpieczny, sprawdź margines
        if (current_asset is not None and 
            current_asset in self.risky_assets and 
            current_asset in valid_risky):
            
            current_momentum = valid_risky[current_asset]
            best_momentum = valid_risky[best_asset]
            
            # Sprawdź czy nowe aktywo jest lepsze o co najmniej threshold
            momentum_improvement = best_momentum - current_momentum
            
            if momentum_improvement < self.rebalance_threshold:
                # Zachowaj obecne aktywo - różnica za mała
                return current_asset
        
        # Jeśli tylko jeden aktyw lub nie używamy ważenia zmiennością
        if len(selected_assets) == 1 or not self.volatility_weighted:
            self.weights = {asset: 1.0 / len(selected_assets) for asset in selected_assets}
            return best_asset  # Zwróć najlepszy aktyw
        
        # Oblicz wagi odwrotnie proporcjonalne do zmienności
        volatilities = {}
        for asset in selected_assets:
            vol = self.calculate_volatility(date, asset)
            if pd.notna(vol) and vol > 0:
                volatilities[asset] = 1.0 / vol
            else:
                volatilities[asset] = 0.0
        
        # Normalizuj wagi
        total_inv_vol = sum(volatilities.values())
        if total_inv_vol > 0:
            self.weights = {asset: vol/total_inv_vol for asset, vol in volatilities.items()}
        else:
            # Jeśli nie można obliczyć wag, użyj równych wag
            self.weights = {asset: 1.0/len(selected_assets) for asset in selected_assets}
        
        # Zwróć aktyw z największą wagą
        return max(self.weights.items(), key=lambda x: x[1])[0]
    
    def _select_safe_asset(self, date, current_asset=None):
        """
        Wybiera najlepsze aktywo bezpieczne na podstawie momentum.
        Uwzględnia margines rebalansowania - nie zmienia aktywa jeśli różnica za mała.
        Jeśli brak ważnych danych, zwraca pierwsze dostępne aktywo bezpieczne.
        """
        # Oblicz momentum dla wszystkich aktywów bezpiecznych
        safe_momentum = {}
        for asset in self.safe_assets:
            mom = self.calculate_momentum(date, asset)
            safe_momentum[asset] = mom
        
        # Odfiltruj NaN i znajdź najlepsze aktywo bezpieczne
        valid_safe = {k: v for k, v in safe_momentum.items() if pd.notna(v)}
        
        if not valid_safe:
            # Brak ważnych danych - zwróć pierwsze dostępne aktywo bezpieczne
            return self.safe_assets[0]
        
        # Znajdź najlepsze aktywo bezpieczne
        best_safe_asset = max(valid_safe.items(), key=lambda x: x[1])[0]
        
        # Sprawdź margines rebalansowania dla aktywów bezpiecznych
        if (current_asset is not None and 
            current_asset in self.safe_assets and 
            current_asset in valid_safe):
            
            current_momentum = valid_safe[current_asset]
            best_momentum = valid_safe[best_safe_asset]
            
            # Sprawdź czy nowe aktywo jest lepsze o co najmniej threshold
            momentum_improvement = best_momentum - current_momentum
            
            if momentum_improvement < self.rebalance_threshold:
                # Zachowaj obecne aktywo bezpieczne - różnica za mała
                return current_asset
        
        return best_safe_asset
    
    def run_backtest(self):
        """Przeprowadza backtesting strategii GEM."""
        if self.prices is None:
            self.download_data()
        
        print("\nUruchamianie backtestingu strategii GEM...")
        
        # Inicjalizacja
        portfolio_value = []
        holdings_history = []
        dates = []
        shares_history = []
        
        # Utwórz miesięczne daty rebalansowania używając rzeczywistych dat z danych
        monthly_prices = self.prices.resample('M').last()
        monthly_dates = monthly_prices.index
        
        # Inicjalizacja portfela - wybierz najlepsze aktywo bezpieczne na pierwszą datę
        first_date = monthly_dates[0]
        current_asset = self._select_safe_asset(first_date)
        shares = {}  # Słownik przechowujący liczbę jednostek dla każdego aktywa
        for asset in self.all_assets:
            shares[asset] = 0
            
        # Pierwsza wpłata
        if self.investment_strategy == 'dca':
            initial_investment = self.initial_capital
        else:
            initial_investment = self.initial_capital
        
        # Zainwestuj pierwszą wpłatę w bezpieczne aktywo
        initial_price = monthly_prices.loc[monthly_dates[0], current_asset]
        # Koszt transakcyjny przy pierwszej wpłacie
        shares[current_asset] = (initial_investment / initial_price) * (1 - self.transaction_cost)
        
        for i, date in enumerate(monthly_dates):
            # Dodaj miesięczną wpłatę DCA (jeśli nie jest to pierwszy miesiąc)
            if self.investment_strategy == 'dca' and i > 0:
                # Dodaj nowe jednostki do obecnie trzymanego aktywa
                current_price = monthly_prices.loc[date, current_asset]
                # Koszt transakcyjny przy każdym zakupie DCA
                new_shares = (self.monthly_contribution / current_price) * (1 - self.transaction_cost)
                shares[current_asset] += new_shares
            
            # Sprawdź, czy to okres rebalansowania i mamy wystarczająco danych
            if i % self.rebalance_frequency == 0 and i >= self.lookback_period:
                # Wybierz nowe aktywo
                new_asset = self.select_asset(date, current_asset)
                
                # Jeśli zmiana aktywa, wykonaj rebalansowanie
                if new_asset != current_asset:
                    # Sprzedaj obecne aktywo
                    current_price = monthly_prices.loc[date, current_asset]
                    total_value = shares[current_asset] * current_price
                    
                    # Uwzględnij koszt transakcyjny tylko przy zmianie aktywa
                    total_value *= (1 - self.transaction_cost)  # Koszt sprzedaży
                    
                    # Kup nowe aktywo
                    new_price = monthly_prices.loc[date, new_asset]
                    shares[current_asset] = 0  # Wyzeruj stare aktywo
                    shares[new_asset] = (total_value / new_price) * (1 - self.transaction_cost)  # Koszt kupna
                    current_asset = new_asset
            
            # Oblicz wartość portfela
            current_price = monthly_prices.loc[date, current_asset]
            value = shares[current_asset] * current_price
            
            # Zapisz historię
            portfolio_value.append(value)
            holdings_history.append(current_asset)
            dates.append(date)
            shares_history.append(shares[current_asset])
        
        # Zapisz wyniki
        self.portfolio_value = pd.Series(portfolio_value, index=dates)
        self.holdings = pd.Series(holdings_history, index=dates)
        self.shares_history = pd.Series(shares_history, index=dates)
        
        print("Backtest zakończony pomyślnie!")
        
        # Debug: pokaż szczegółowe statystyki
        if self.investment_strategy == 'dca':
            total_contributions = self.initial_capital + (len(monthly_dates) - 1) * self.monthly_contribution
            print(f"\nDEBUG - DCA Strategy:")
            print(f"Całkowite wpłaty: ${total_contributions:,.2f}")
            print(f"Wartość końcowa: ${portfolio_value[-1]:,.2f}")
            print(f"Zysk/Strata: ${portfolio_value[-1] - total_contributions:,.2f}")
            print(f"\nStatystyki końcowe:")
            print(f"  Aktywo końcowe: {current_asset}")
            print(f"  Liczba jednostek: {shares_history[-1]:.6f}")
            print(f"  Cena końcowa: ${monthly_prices.loc[dates[-1], current_asset]:,.2f}")
            
            # Pokaż rozkład aktywów
            holdings_stats = pd.Series(holdings_history).value_counts()
            print(f"\nRozkład aktywów:")
            for asset, count in holdings_stats.items():
                percentage = (count / len(holdings_history)) * 100
                print(f"  {asset}: {count} okresów ({percentage:.1f}%)")
            
            # Pokaż statystyki transakcji
            trades = sum(1 for i in range(1, len(holdings_history)) if holdings_history[i] != holdings_history[i-1])
            print(f"\nStatystyki transakcji:")
            print(f"  Liczba zmian aktywów: {trades}")
            if trades > 0:
                print(f"  Średni czas trzymania: {len(holdings_history)/trades:.1f} miesięcy")
            else:
                print("  Średni czas trzymania: N/A (brak zmian aktywów)")
        
    def calculate_metrics(self):
        """Oblicza wskaźniki wydajności strategii."""
        if self.portfolio_value is None:
            raise ValueError("Najpierw uruchom backtest!")
        
        # Oblicz całkowite wpłaty
        if self.investment_strategy == 'dca':
            total_months = len(self.portfolio_value)
            total_contributions = self.initial_capital + (total_months - 1) * self.monthly_contribution
        else:
            total_contributions = self.initial_capital
        
        # Zwroty miesięczne (bo używamy miesięcznych dat)
        returns = self.portfolio_value.pct_change().dropna()
        
        # Total Return - względem całkowitych wpłat
        total_return = (self.portfolio_value.iloc[-1] / total_contributions - 1) * 100
        
        # CAGR (Compound Annual Growth Rate) - względem całkowitych wpłat
        years = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days / 365.25
        cagr = (np.power(self.portfolio_value.iloc[-1] / total_contributions, 
                         1 / years) - 1) * 100
        
        # Maximum Drawdown
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (roczny) - używamy miesięcznych zwrotów i rzeczywistej stopy wolnej od ryzyka
        rf_monthly = self.rf_prices.resample('M').last()
        rf_returns = rf_monthly.pct_change().reindex(returns.index)
        
        excess_returns = returns - rf_returns
        if len(excess_returns) > 1 and excess_returns.std() > 0:
            sharpe_ratio = np.sqrt(12) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = 0
        
        # Volatility (roczna) - używamy miesięcznych zwrotów
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(12) * 100
        else:
            volatility = 0
        
        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Volatility (%)': volatility,
            'Final Value': self.portfolio_value.iloc[-1],
            'Total Contributions': total_contributions
        }
        
        return metrics
    
    def calculate_buyhold_benchmark(self, benchmark_ticker='SPY'):
        """Oblicza wyniki strategii Buy & Hold dla benchmarku."""
        # Pobierz dane benchmarku niezależnie od parametrów strategii GEM
        benchmark_data = yf.download(benchmark_ticker, 
                                   start=self.start_date, 
                                   end=self.end_date,
                                   auto_adjust=True,  # Użyj tego samego co w GEM
                                   progress=False)
        
        # Po auto_adjust=True, skorygowane ceny są w kolumnie 'Close'
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            benchmark_prices = benchmark_data['Close'].squeeze()
        else:
            benchmark_prices = benchmark_data['Close'].squeeze()
                
        # Upewnij się, że mamy Series a nie DataFrame
        if isinstance(benchmark_prices, pd.DataFrame):
            benchmark_prices = benchmark_prices.iloc[:, 0]
        
        # Oblicz całkowite wpłaty
        if self.investment_strategy == 'dca':
            total_months = len(self.portfolio_value)
            total_contributions = self.initial_capital + (total_months - 1) * self.monthly_contribution
        else:
            total_contributions = self.initial_capital
        
        # Oblicz wartość portfela Buy & Hold - NIEZALEŻNIE od strategii GEM
        if self.investment_strategy == 'dca':
            # Dla DCA: symuluj miesięczne zakupy na dziennych danych
            daily_buyhold = pd.Series(index=benchmark_prices.index, dtype=float)
            shares_owned = 0
            
            # Utwórz własne daty miesięczne dla benchmarku (niezależne od GEM)
            monthly_dates = benchmark_prices.resample('M').last().index
            
            for i, monthly_date in enumerate(monthly_dates):
                if i == 0:
                    # Pierwsza wpłata - użyj ceny z końca pierwszego miesiąca
                    contribution = self.initial_capital
                else:
                    # Miesięczne wpłaty
                    contribution = self.monthly_contribution
                
                # Znajdź cenę na koniec miesiąca (lub najbliższą dostępną)
                price = benchmark_prices.loc[:monthly_date].iloc[-1]
                # Koszt transakcyjny przy każdym zakupie DCA
                new_shares = (contribution / price) * (1 - self.transaction_cost)
                shares_owned += new_shares
                
                # Aktualizuj wartość dla wszystkich dni od tej daty
                daily_buyhold.loc[monthly_date:] = shares_owned * benchmark_prices.loc[monthly_date:]
            
            # Użyj własnych dat miesięcznych dla benchmarku
            buyhold_value = daily_buyhold.resample('M').last()
        else:
            # Dla lump sum: tradycyjny Buy & Hold
            # Użyj pierwszej dostępnej daty benchmarku
            start_date = benchmark_prices.index[0]
            start_price = benchmark_prices.loc[start_date]
            initial_shares = self.initial_capital / start_price
            buyhold_value = (benchmark_prices * initial_shares).resample('M').last()
        
        # Oblicz metryki - używamy miesięcznych zwrotów
        returns = buyhold_value.pct_change().dropna()
        
        total_return = (buyhold_value.iloc[-1] / total_contributions - 1) * 100
        years = (buyhold_value.index[-1] - buyhold_value.index[0]).days / 365.25
        cagr = (np.power(buyhold_value.iloc[-1] / total_contributions, 
                        1 / years) - 1) * 100
        
        cummax = buyhold_value.cummax()
        drawdown = (buyhold_value - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (roczny) - pobierz dane dla stopy wolnej od ryzyka niezależnie
        rf_data = yf.download(self.risk_free_asset, 
                             start=self.start_date, 
                             end=self.end_date,
                             auto_adjust=True,
                             progress=False)
        
        if isinstance(rf_data.columns, pd.MultiIndex):
            rf_prices = rf_data['Close'].squeeze()
        else:
            rf_prices = rf_data['Close'].squeeze()
            
        rf_monthly = rf_prices.resample('M').last()
        rf_returns = rf_monthly.pct_change().reindex(returns.index)
        
        excess_returns = returns - rf_returns
        if len(excess_returns) > 1 and excess_returns.std() > 0:
            sharpe_ratio = np.sqrt(12) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = 0
        
        # Volatility (roczna) - używamy miesięcznych zwrotów
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(12) * 100
        else:
            volatility = 0
        
        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Volatility (%)': volatility,
            'Final Value': buyhold_value.iloc[-1],
            'Total Contributions': total_contributions
        }
        
        return buyhold_value, metrics
    
    def plot_results(self, benchmark_ticker='SPY'):
        """Tworzy wizualizacje wyników."""
        if self.portfolio_value is None:
            raise ValueError("Najpierw uruchom backtest!")
        
        # Oblicz metryki
        gem_metrics = self.calculate_metrics()
        buyhold_value, buyhold_metrics = self.calculate_buyhold_benchmark(benchmark_ticker)
        
        # Utwórz figurę z 4 wykresami
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Analiza Strategii GEM vs Buy & Hold', fontsize=16, fontweight='bold')
        
        # Wykres 1: Skumulowana wartość portfela
        ax1 = axes[0, 0]
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, 
                label='GEM Strategy', linewidth=2, color='blue')
        ax1.plot(buyhold_value.index, buyhold_value.values, 
                label=f'Buy & Hold ({benchmark_ticker})', linewidth=2, 
                color='orange', linestyle='--')
        
        ax1.set_title('Skumulowana Wartość Portfela', fontweight='bold')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Wartość Portfela ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Wykres 2: Porównanie CAGR i Max Drawdown
        ax2 = axes[0, 1]
        metrics_to_plot = ['CAGR (%)', 'Max Drawdown (%)']
        gem_values = [gem_metrics[m] for m in metrics_to_plot]
        buyhold_values = [buyhold_metrics[m] for m in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        ax2.bar(x - width/2, gem_values, width, label='GEM Strategy', color='blue', alpha=0.7)
        ax2.bar(x + width/2, buyhold_values, width, label=f'Buy & Hold ({benchmark_ticker})', 
               color='orange', alpha=0.7)
        ax2.set_title('CAGR i Maximum Drawdown', fontweight='bold')
        ax2.set_ylabel('Wartość (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_to_plot)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Wykres 3: Timeline aktywów
        ax3 = axes[1, 0]
        
        # Utwórz mapę kolorów dla aktywów
        asset_colors = {
            # Główne indeksy USA
            'SPY': '#1f77b4',  # Niebieski
            'RSP': '#2ca02c',  # Zielony
            'QQQ': '#ff7f0e',  # Pomarańczowy
            'IWM': '#9467bd',  # Fioletowy
            
            # Rynki międzynarodowe
            'EFA': '#e377c2',  # Różowy
            'EEM': '#8c564b',  # Brązowy
            
            # Alternatywne klasy aktywów
            'GLD': '#d62728',  # Czerwony
            'VNQ': '#17becf',  # Turkusowy
            'BTC-USD': '#ffd700',  # Złoty/Żółty
            
            # Obligacje
            'TLT': '#bcbd22',  # Żółto-zielony
            'IEF': '#ff9896',  # Jasnoróżowy
            'SHY': '#98df8a',  # Jasnozielony
        }
        
        # Konwertuj holdings na numeryczne wartości dla wykresu
        asset_mapping = {asset: i for i, asset in enumerate(self.all_assets)}
        holdings_numeric = self.holdings.map(asset_mapping)
        
        # Rysuj punkty tylko przy zmianie aktywa
        for i, asset in enumerate(self.all_assets):
            # Znajdź daty, gdzie zaczyna się trzymanie danego aktywa
            asset_dates = []
            holdings_list = self.holdings.tolist()
            for j in range(len(holdings_list)):
                if j == 0 and holdings_list[j] == asset:  # Pierwszy wpis
                    asset_dates.append(self.holdings.index[j])
                elif j > 0 and holdings_list[j] == asset and holdings_list[j-1] != asset:  # Zmiana aktywa
                    asset_dates.append(self.holdings.index[j])
            
            if asset_dates:
                values = [i] * len(asset_dates)
                ax3.scatter(asset_dates, values, color=asset_colors.get(asset, 'gray'), 
                          s=50, alpha=0.8, label=asset)
        
        ax3.set_title('Timeline Aktywów w Strategii GEM', fontweight='bold')
        ax3.set_xlabel('Data')
        ax3.set_ylabel('Aktywo')
        ax3.set_yticks(range(len(self.all_assets)))
        ax3.set_yticklabels(self.all_assets)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Wykres 4: Tabela metryk
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Użyj metryk z calculate_metrics
        total_contributions = gem_metrics['Total Contributions']
        final_value_gem = gem_metrics['Final Value']
        final_value_bh = buyhold_metrics['Final Value']
        profit_gem = final_value_gem - total_contributions
        profit_bh = final_value_bh - total_contributions
        
        table_data = []
        if self.investment_strategy == 'dca':
            table_data.append(['Całkowite wpłaty', f'${total_contributions:,.2f}', f'${total_contributions:,.2f}'])
        else:
            table_data.append(['Kapitał początkowy', f'${total_contributions:,.2f}', f'${total_contributions:,.2f}'])
        table_data.append(['Wartość końcowa', f'${final_value_gem:,.2f}', f'${final_value_bh:,.2f}'])
        table_data.append(['Zysk/Strata', f'${profit_gem:,.2f}', f'${profit_bh:,.2f}'])
        table_data.append(['Total Return (%)', f'{gem_metrics["Total Return (%)"]:.2f}%', f'{buyhold_metrics["Total Return (%)"]:.2f}%'])
        table_data.append(['CAGR (%)', f'{gem_metrics["CAGR (%)"]:.2f}%', f'{buyhold_metrics["CAGR (%)"]:.2f}%'])
        table_data.append(['Max Drawdown (%)', f'{gem_metrics["Max Drawdown (%)"]:.2f}%', f'{buyhold_metrics["Max Drawdown (%)"]:.2f}%'])
        table_data.append(['Sharpe Ratio', f'{gem_metrics["Sharpe Ratio"]:.2f}', f'{buyhold_metrics["Sharpe Ratio"]:.2f}'])
        table_data.append(['Volatility (%)', f'{gem_metrics["Volatility (%)"]:.2f}%', f'{buyhold_metrics["Volatility (%)"]:.2f}%'])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Metryka', 'GEM Strategy', f'Buy & Hold ({benchmark_ticker})'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.35, 0.325, 0.325])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Kolorowanie nagłówków
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Kolorowanie wierszy z zyskiem/stratą
        if profit_gem > 0:
            table[(3, 1)].set_facecolor('#E8F5E8')  # Jasnozielone tło dla zysku
        else:
            table[(3, 1)].set_facecolor('#FFE8E8')  # Jasnoczerwone tło dla straty
            
        if profit_bh > 0:
            table[(3, 2)].set_facecolor('#E8F5E8')
        else:
            table[(3, 2)].set_facecolor('#FFE8E8')
        
        
        plt.tight_layout()
        
        # Dodaj instrukcje dla użytkownika
        print("\n" + "="*60)
        print("INSTRUKCJE OBSŁUGI WYKRESÓW")
        print("="*60)
        print("• Kliknij i przeciągnij aby przybliżyć obszar")
        print("• Kliknij prawym przyciskiem i przeciągnij aby przesunąć wykres")
        print("• Kliknij dwukrotnie aby zresetować widok")
        print("• Użyj kółka myszy aby przybliżyć/oddalić")
        print("• Zamknij okno wykresu aby kontynuować")
        print("="*60)
        
        plt.show()
        
        # Czekaj na zamknięcie okna tylko w trybie interaktywnym
        if len(sys.argv) == 1:  # Tryb interaktywny
            input("Naciśnij Enter po zamknięciu okna wykresu...")
        
        # Wyświetl metryki w konsoli
        print("\n" + "="*60)
        print("PODSUMOWANIE WYNIKÓW")
        print("="*60)
        print(f"\n{'Metryka':<25} {'GEM':<15} {'Buy & Hold':<15}")
        print("-"*60)
        for metric in ['Total Return (%)', 'CAGR (%)', 'Max Drawdown (%)', 
                      'Sharpe Ratio', 'Volatility (%)']:
            print(f"{metric:<25} {gem_metrics[metric]:>14.2f} {buyhold_metrics[metric]:>14.2f}")
        print(f"{'Final Value':<25} ${gem_metrics['Final Value']:>13,.2f} ${buyhold_metrics['Final Value']:>13,.2f}")
        print("="*60)
        
        # Statystyki trzymania aktywów
        print("\n" + "="*60)
        print("STATYSTYKI TRZYMANIA AKTYWÓW")
        print("="*60)
        holdings_stats = self.holdings.value_counts()
        total_periods = len(self.holdings)
        for asset, count in holdings_stats.items():
            percentage = (count / total_periods) * 100
            print(f"{asset:<10} {count:>5} okresów ({percentage:>5.1f}%)")
        print("="*60)
    
    def plot_detailed_results(self, benchmark_ticker='SPY'):
        """Tworzy szczegółowy, interaktywny wykres wyników."""
        if self.portfolio_value is None:
            raise ValueError("Najpierw uruchom backtest!")
        
        # Oblicz metryki
        gem_metrics = self.calculate_metrics()
        buyhold_value, buyhold_metrics = self.calculate_buyhold_benchmark(benchmark_ticker)
        
        # Utwórz duży, interaktywny wykres
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
        fig.suptitle('Szczegółowa Analiza Strategii GEM vs Buy & Hold', fontsize=16, fontweight='bold')
        
        # Wykres 1: Skumulowana wartość portfela z oznaczeniem aktywów
        ax1.plot(self.portfolio_value.index, self.portfolio_value.values, 
                label='GEM Strategy', linewidth=2, color='blue')
        ax1.plot(buyhold_value.index, buyhold_value.values, 
                label=f'Buy & Hold ({benchmark_ticker})', linewidth=2, 
                color='orange', linestyle='--')
        
        # Dodaj kolorowe punkty pokazujące trzymane aktywa
        asset_colors = {
            # Główne indeksy USA
            'SPY': '#1f77b4',  # Niebieski
            'RSP': '#2ca02c',  # Zielony
            'QQQ': '#ff7f0e',  # Pomarańczowy
            'IWM': '#9467bd',  # Fioletowy
            
            # Rynki międzynarodowe
            'EFA': '#e377c2',  # Różowy
            'EEM': '#8c564b',  # Brązowy
            
            # Alternatywne klasy aktywów
            'GLD': '#d62728',  # Czerwony
            'VNQ': '#17becf',  # Turkusowy
            'BTC-USD': '#ffd700',  # Złoty/Żółty
            
            # Obligacje
            'TLT': '#bcbd22',  # Żółto-zielony
            'IEF': '#ff9896',  # Jasnoróżowy
            'SHY': '#98df8a',  # Jasnozielony
        }
        
        # Rysuj punkty tylko przy zmianie aktywa
        holdings_list = self.holdings.tolist()
        for asset in self.all_assets:
            # Znajdź daty, gdzie zaczyna się trzymanie danego aktywa
            asset_dates = []
            for i in range(len(holdings_list)):
                if i == 0 and holdings_list[i] == asset:  # Pierwszy wpis
                    asset_dates.append(self.holdings.index[i])
                elif i > 0 and holdings_list[i] == asset and holdings_list[i-1] != asset:  # Zmiana aktywa
                    asset_dates.append(self.holdings.index[i])
            
            if asset_dates:
                asset_values = self.portfolio_value.loc[asset_dates]
                ax1.scatter(asset_dates, asset_values, 
                          color=asset_colors.get(asset, 'gray'), 
                          s=40, alpha=0.8, label=f'{asset}')
        
        ax1.set_title('Skumulowana Wartość Portfela z Oznaczeniem Aktywów', fontweight='bold')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Wartość Portfela ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Wykres 2: Timeline aktywów
        asset_colors = {
            # Główne indeksy USA
            'SPY': '#1f77b4',  # Niebieski
            'RSP': '#2ca02c',  # Zielony
            'QQQ': '#ff7f0e',  # Pomarańczowy
            'IWM': '#9467bd',  # Fioletowy
            
            # Rynki międzynarodowe
            'EFA': '#e377c2',  # Różowy
            'EEM': '#8c564b',  # Brązowy
            
            # Alternatywne klasy aktywów
            'GLD': '#d62728',  # Czerwony
            'VNQ': '#17becf',  # Turkusowy
            'BTC-USD': '#ffd700',  # Złoty/Żółty
            
            # Obligacje
            'TLT': '#bcbd22',  # Żółto-zielony
            'IEF': '#ff9896',  # Jasnoróżowy
            'SHY': '#98df8a',  # Jasnozielony
        }
        
        # Konwertuj holdings na numeryczne wartości dla wykresu
        asset_mapping = {asset: i for i, asset in enumerate(self.all_assets)}
        holdings_numeric = self.holdings.map(asset_mapping)
        
        # Rysuj timeline
        for i, asset in enumerate(self.all_assets):
            mask = holdings_numeric == i
            if mask.any():
                dates = holdings_numeric[mask].index
                values = [i] * len(dates)
                ax2.scatter(dates, values, color=asset_colors.get(asset, 'gray'), 
                           s=60, alpha=0.8, label=asset)
        
        ax2.set_title('Timeline Aktywów w Strategii GEM', fontweight='bold')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Aktywo')
        ax2.set_yticks(range(len(self.all_assets)))
        ax2.set_yticklabels(self.all_assets)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Wykres 3: Drawdown w czasie
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax * 100
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax3.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        ax3.set_title('Drawdown Strategii GEM w Czasie', fontweight='bold')
        ax3.set_xlabel('Data')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Dodaj instrukcje dla użytkownika
        print("\n" + "="*60)
        print("INSTRUKCJE OBSŁUGI INTERAKTYWNEGO WYKRESU")
        print("="*60)
        print("• Kliknij i przeciągnij aby przybliżyć obszar")
        print("• Kliknij prawym przyciskiem i przeciągnij aby przesunąć wykres")
        print("• Kliknij dwukrotnie aby zresetować widok")
        print("• Użyj kółka myszy aby przybliżyć/oddalić")
        print("• Zamknij okno wykresu aby kontynuować")
        print("="*60)
        
        plt.show()
        
        # Czekaj na zamknięcie okna tylko w trybie interaktywnym
        if len(sys.argv) == 1:  # Tryb interaktywny
            input("Naciśnij Enter po zamknięciu okna wykresu...")


# PRZYKŁAD UŻYCIA
def run_gem_strategy(strategy_type='dca', fee_type='auto', plot_type=1, top_k=1, volatility_weighted=False, benchmark='QQQ', rebalance_threshold=0.01):
    """
    Uruchamia strategię GEM z określonymi parametrami.
    
    Parametry:
    ----------
    strategy_type : str
        'dca' lub 'lump_sum'
    fee_type : str lub float
        'auto', 0.0, 0.005, 0.01
    plot_type : int
        1 (standardowe) lub 2 (szczegółowe)
    top_k : int
        Liczba najlepszych aktywów do portfela (1 = klasyczny GEM)
    volatility_weighted : bool
        Czy ważyć aktywa odwrotnie do zmienności
    benchmark : str
        Benchmark do porównania ('QQQ' lub 'SPY')
    rebalance_threshold : float
        Margines rebalansowania jako % (np. 0.01 = 1%). Nowe aktywo musi być 
        lepsze o co najmniej tyle, aby nastąpiła zmiana. Zmniejsza liczbę transakcji.
    """
    print("="*60)
    print("STRATEGIA GEM - GLOBAL EQUITY MOMENTUM")
    print("="*60)
    
    # Konfiguracja strategii inwestycyjnej
    if strategy_type == 'dca':
        investment_strategy = 'dca'
        initial_capital = 10000  # Pierwsza wpłata
        monthly_contribution = 1000
        print(f"\nWybrano strategię DCA:")
        print(f"- Pierwsza wpłata: ${initial_capital:,}")
        print(f"- Miesięczne wpłaty: ${monthly_contribution:,}")
    else:
        investment_strategy = 'lump_sum'
        initial_capital = 10000
        monthly_contribution = 0
        print(f"\nWybrano strategię Lump Sum:")
        print(f"- Wpłata początkowa: ${initial_capital:,}")
    
    # Konfiguracja opłat zarządczych
    print("\nUwaga: Koszty zarządzania ETF-ów (expense ratio) są już uwzględnione w cenach.")
    print(f"Margines rebalansowania: {rebalance_threshold*100:.1f}% - nowe aktywo musi być lepsze o co najmniej tyle, aby nastąpiła zmiana.")
    
    # Konfiguracja strategii
    gem = GEMStrategy(
         risky_assets=[
                     # Główne indeksy USA
                    #  'SPY',   # S&P 500
                     'QQQ',   # NASDAQ 100
                    # 'XNAS', # NASDAQ 100 BOSSA
                    #  'IWM',   # Russell 2000 (małe spółki)
                     
                     # Rynki międzynarodowe
                     'EFA',   # MSCI EAFE (rynki rozwinięte poza USA)
                    #  'EEM',   # MSCI Emerging Markets
                    #  'EXUS' # MSCI (rynki rozwinięte poza USA) BOSSA
                     
                     # Alternatywne klasy aktywów
                    #  'GLD',   # SPDR Gold Shares
                    #  'VNQ',   # Vanguard Real Estate ETF
                    #  'BTC-USD', # Bitcoin (dostępny od 2014)
                 ],
                 safe_assets=[
                    #  'TLT',   # 20+ Year Treasury Bond
                     'IEF',   # 7-10 Year Treasury Bond
                    #  'CBU0',   # 7-10 Year Treasury Bond BOSSA
                     'SHY',   # 1-3 Year Treasury Bond
                    #  'IBTA',   # 1-3 Year Treasury Bond BOSSA
                    #  'TLT',
                    #  'BIL'
                 ],
        start_date='2002-01-01',              # Data początkowa (start SHY ETF)
        end_date='2025-10-15',                        # Data końcowa (automatycznie)
        lookback_period=12,                   # 12 miesięcy momentum
        gap_period=1,                         # Wykluczamy ostatni miesiąc
        rebalance_frequency=1,                # Rebalansowanie co miesiąc
        initial_capital=initial_capital,      # Kapitał początkowy
        monthly_contribution=monthly_contribution,  # Miesięczne wpłaty
        investment_strategy=investment_strategy,    # Strategia inwestycyjna
        transaction_cost=0.001,               # Koszt transakcji 0.1%
        top_k=top_k,                         # Liczba najlepszych aktywów
        volatility_weighted=volatility_weighted,  # Ważenie odwrotnie do zmienności
        rebalance_threshold=rebalance_threshold,  # Margines rebalansowania (1% domyślnie)
        risk_free_asset='SHY'
    )
    
    # Uruchom backtest
    gem.run_backtest()
    
    # Wyświetl wyniki
    if plot_type == 2:
        gem.plot_detailed_results(benchmark_ticker=benchmark)
    else:
        gem.plot_results(benchmark_ticker=benchmark)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategia GEM - Global Equity Momentum')
    parser.add_argument('--strategy', choices=['dca', 'lump_sum'], default='dca',
                      help='Strategia inwestycyjna (dca lub lump_sum)')
    parser.add_argument('--fee', choices=['auto', '0', '0.5', '1.0'], default='auto',
                      help='Opłata zarządcza (auto, 0, 0.5 lub 1.0)')
    parser.add_argument('--plot', choices=['1', '2'], default='1',
                      help='Typ wykresu (1=standardowy, 2=szczegółowy)')
    parser.add_argument('--top-k', type=int, default=1,
                      help='Liczba najlepszych aktywów do portfela (1 = klasyczny GEM)')
    parser.add_argument('--volatility-weighted', action='store_true',
                            help='Ważyć aktywa odwrotnie do zmienności')
    parser.add_argument('--benchmark', choices=['SPY', 'QQQ'], default='QQQ',
                            help='Benchmark do porównania (SPY lub QQQ)')
    
    # Jeśli nie ma argumentów, użyj trybu interaktywnego
    if len(sys.argv) == 1:
        try:
            # print("Wybierz strategię inwestycyjną:")
            # print("1. Lump Sum - jedna wpłata $10,000 na początku")
            # print("2. DCA - $1,000 miesięcznie przez cały okres")
            # strategy_choice = input("Wprowadź wybór (1 lub 2): ").strip()
            # strategy_type = 'dca' if strategy_choice == "2" else 'lump_sum'
            strategy_type = 'dca'
            
            # print("\nCzy chcesz uwzględnić opłaty zarządcze?")
            # print("1. Bez opłat zarządczych")
            # print("2. Rzeczywiste opłaty z Yahoo Finance (automatyczne)")
            # print("3. Opłata 0.5% rocznie (szacunkowa)")
            # print("4. Opłata 1.0% rocznie (szacunkowa)")
            # fee_choice = input("Wprowadź wybór (1, 2, 3 lub 4): ").strip()
            fee_choice = 2
            
            if fee_choice == "2":
                fee_type = 'auto'
            elif fee_choice == "3":
                fee_type = 0.005
            elif fee_choice == "4":
                fee_type = 0.01
            else:
                fee_type = 0.0
            
            # print("\nWybierz liczbę najlepszych aktywów (Top-K):")
            # print("1. Klasyczny GEM (1 aktywo)")
            # print("2. Top-2 aktywa")
            # print("3. Top-3 aktywa")
            # top_k = int(input("Wprowadź wybór (1-3): ").strip())
            top_k = 1
            
            # print("\nCzy chcesz ważyć aktywa odwrotnie do zmienności?")
            # print("1. Nie - równe wagi")
            # print("2. Tak - wagi odwrotnie proporcjonalne do zmienności")
            # vol_choice = input("Wprowadź wybór (1 lub 2): ").strip()

            vol_choice = 1
            volatility_weighted = (vol_choice == "2")
           
            # print("\nWybierz typ wykresu:")
            # print("1. Standardowe wykresy (4 wykresy)")
            # print("2. Szczegółowy interaktywny wykres (2 duże wykresy)")
            # plot_choice = int(input("Wprowadź wybór (1 lub 2): ").strip())
            plot_choice = 1
            
            # print("\nWybierz benchmark do porównania:")
            # print("1. QQQ (NASDAQ 100)")
            # print("2. SPY (S&P 500)")
            # benchmark_choice = input("Wprowadź wybór (1 lub 2): ").strip()
            # benchmark = 'QQQ' if benchmark_choice == '1' else 'SPY'
            benchmark = 'SPY'

            run_gem_strategy(strategy_type, fee_type, plot_choice, top_k, volatility_weighted, benchmark)
            
        except (EOFError, KeyboardInterrupt):
            print("\nWykryto tryb nieinteraktywny. Użyj argumentów wiersza poleceń:")
            print("python gem.py --help")
            sys.exit(1)
    else:
        args = parser.parse_args()
        
        # Konwersja argumentów
        strategy_type = args.strategy
        if args.fee == 'auto':
            fee_type = 'auto'
        elif args.fee == '0':
            fee_type = 0.0
        elif args.fee == '0.5':
            fee_type = 0.005
        else:
            fee_type = 0.01
        plot_type = int(args.plot)
        
        run_gem_strategy(strategy_type, fee_type, plot_type, args.top_k, args.volatility_weighted, args.benchmark)