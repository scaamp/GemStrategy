import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import telebot
import io
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()

# Konfiguracja
API_KEY = os.getenv('EODHD_API_KEY')  # Pobierz klucz API z pliku .env
if not API_KEY:
    raise ValueError("Brak klucza API w pliku .env!")
BASE_URL = "https://eodhd.com/api/"

# Lista ISIN-ów do śledzenia
ISINS = {
    'IE00BYXPSP02': 'iShares $ Treasury Bond 1-3yr UCITS ETF USD (Acc)',
    'IE00B3VWN518': 'iShares $ Treasury Bd 7-10y ETF USD Acc',
    'IE00BMFKG444': 'Xtrackers NASDAQ 100 UCITS ETF 1C',
    'IE0006WW1TQ4': 'Xtrackers MSCI World ex USA UCITS ETF 1C USD',
}

def get_ticker_for_isin(isin: str) -> Tuple[str, str, str]:
    """
    Wyszukuje symbol giełdowy dla danego ISIN.
    """
    url = f"{BASE_URL}search/{isin}"
    params = {
        'api_token': API_KEY,
        'fmt': 'json',
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError(f"Nie znaleziono tickera dla {isin}")
            
        # Znajdź pierwszy wynik z giełdy XETRA lub weź pierwszy dostępny
        preferred_result = None
        for result in data:
            if result.get('Exchange') == 'XETRA':
                preferred_result = result
                break
        
        if not preferred_result:
            preferred_result = data[0]
            
        ticker = f"{preferred_result['Code']}.{preferred_result['Exchange']}"
        exchange = preferred_result.get('Exchange', 'Unknown')
        currency = preferred_result.get('Currency', 'Unknown')
        
        return ticker, exchange, currency
        
    except Exception as e:
        print(f"Błąd podczas wyszukiwania tickera dla {isin}: {str(e)}")
        return None, None, None

def get_forex_rate(from_currency: str, to_currency: str, date: str) -> float:
    """
    Pobiera kurs wymiany walut z EODHD.
    """
    if from_currency == to_currency:
        return 1.0
        
    url = f"{BASE_URL}eod/{from_currency}{to_currency}.FOREX"
    params = {
        'api_token': API_KEY,
        'fmt': 'json',
        'from': date,
        'to': date,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return float(data[0]['close'])
        else:
            print(f"Brak kursu {from_currency}/{to_currency} dla daty {date}")
            return None
            
    except Exception as e:
        print(f"Błąd podczas pobierania kursu {from_currency}/{to_currency}: {str(e)}")
        return None

def get_eod_data(isin: str, start_date: str, end_date: str) -> Tuple[pd.Series, str, str, str]:
    """
    Pobiera dane historyczne dla danego ISIN z EODHD.
    """
    ticker, exchange, currency = get_ticker_for_isin(isin)
    if not ticker:
        return pd.Series(), "unknown", "unknown", "unknown"
        
    url = f"{BASE_URL}eod/{ticker}"
    
    params = {
        'api_token': API_KEY,
        'fmt': 'json',
        'from': start_date,
        'to': end_date,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            raise ValueError(f"Brak danych dla {isin}")
        
        # Konwertuj dane do DataFrame i ustaw indeks
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Sprawdź typ danych cenowych
        price_type = "adjusted_close" if 'adjusted_close' in df.columns else "close"
        return df[price_type], price_type, exchange, currency
        
    except Exception as e:
        print(f"Błąd podczas pobierania danych dla {isin}: {str(e)}")
        return pd.Series(), "error", "unknown", "unknown"

def calculate_momentum(prices: pd.Series, lookback_months: int = 12, gap_months: int = 1) -> float:
    """
    Oblicza momentum dla szeregu cenowego z wyłączeniem ostatniego miesiąca.
    """
    if len(prices) < 2:
        return np.nan
        
    # Znajdź datę końcową (miesiąc temu)
    end_date = prices.index[-1] - pd.DateOffset(months=gap_months)
    start_date = end_date - pd.DateOffset(months=lookback_months)
    
    # Wybierz ceny w odpowiednim okresie
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    period_prices = prices[mask]
    
    if len(period_prices) < 2:
        return np.nan
        
    start_price = period_prices.iloc[0]
    end_price = period_prices.iloc[-1]
    
    return (end_price / start_price - 1) * 100

def plot_performance_comparison(data: Dict[str, pd.Series], symbols: Dict[str, str], results: Dict[str, float]) -> io.BytesIO:
    """
    Generuje wykres porównawczy dla wszystkich aktywów.
    """
    current_date = datetime.now()
    end_date = current_date - timedelta(days=31)  # Data analizy (miesiąc temu)
    plt.figure(figsize=(15, 8))
    
    colors = {
        'IE00BMFKG444': '#0000FF',  # NASDAQ 100 - niebieski
        'IE0006WW1TQ4': '#000000',  # World ex-USA - czarny
        'IE00B3VWN518': '#008000',  # 7-10Y Treasury - zielony
        'IE00BYXPSP02': '#FF0000',  # 1-3Y Treasury - czerwony
    }
    
    # Normalizuj dane do procentowych zmian i przytnij do daty końcowej
    normalized_data = {}
    for isin, prices in data.items():
        if not prices.empty:
            mask = prices.index <= end_date  # Trim data to analysis date
            period_prices = prices[mask]
            if not period_prices.empty:
                normalized_data[isin] = (period_prices / period_prices.iloc[0] - 1) * 100
    
    for isin, values in normalized_data.items():
        symbol = symbols[isin]
        momentum = results[isin]  # Use calculated momentum for legend
        plt.plot(values.index, values.values, 
                label=f"{symbol} ({momentum:.2f}%)", 
                color=colors.get(isin, 'gray'),
                linewidth=1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title('Porównanie Wyników ETF-ów (znormalizowane)', pad=20)
    plt.xlabel('Data')
    plt.ylabel('Zmiana (%)')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %b'))
    
    # Dostosuj marginesy
    plt.tight_layout()
    
    # Zapisz wykres do bufora
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def get_current_leader() -> Tuple[str, Dict[str, float], Dict[str, pd.Series], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Pobiera dane dla wszystkich ISIN-ów i wyznacza obecnego lidera.
    """
    # Ustaw zakres dat (ostatni rok + 1 miesiąc na gap_period)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365+31)).strftime('%Y-%m-%d')
    
    # Pobierz dane i oblicz momentum dla każdego ISIN
    results = {}
    price_data = {}
    price_types = {}
    exchanges = {}
    currencies = {}
    for isin in ISINS.keys():
        prices, price_type, exchange, currency = get_eod_data(isin, start_date, end_date)
        price_data[isin] = prices
        price_types[isin] = price_type
        exchanges[isin] = exchange
        currencies[isin] = currency
        if not prices.empty:
            momentum = calculate_momentum(prices)
            results[isin] = momentum
    
    if not results:
        raise ValueError("Nie udało się pobrać danych dla żadnego ISIN")
    
    # Znajdź lidera (najwyższe momentum)
    leader = max(results.items(), key=lambda x: x[1])[0]
    
    return leader, results, price_data, price_types, exchanges, currencies

def format_telegram_message(leader: str, results: Dict[str, float], price_data: Dict[str, pd.Series], 
                     exchanges: Dict[str, str], currencies: Dict[str, str],
                     original_results: Dict[str, float] = None, target_currency: str = 'original') -> str:
    """
    Wyświetla wyniki analizy.
    """
    current_date = datetime.now()
    analysis_date = current_date - timedelta(days=31)  # Data analizy (miesiąc temu)
    start_date = analysis_date - timedelta(days=365)   # Data początkowa (rok przed datą analizy)
    
    # Nagłówek
    output = []
    output.append("*🤖 GEM Strategy - Global Equity Momentum*")
    output.append("*🔄 Analiza Momentum (12M)*")
    output.append("💰 Sygnał do wykonania DCA (Dollar-Cost Averaging)")
    output.append(f"📅 Okres: {start_date.strftime('%Y-%m-%d')} → {analysis_date.strftime('%Y-%m-%d')}")
    
    # Informacja o walucie
    if target_currency != 'original':
        output.append(f"\n💱 *Wyniki w {target_currency}*")
        
        # Wpływ zmian kursowych (tylko dla aktywów ze zmianą)
        changes = []
        for isin in results:
            if original_results:
                change = results[isin] - original_results[isin]
                if abs(change) > 0.01:  # Pokazuj tylko istotne zmiany
                    orig_currency = currencies[isin] if target_currency == 'original' else currencies.get(isin + '_orig', 'Unknown')
                    name = ISINS[isin].split(' UCITS')[0]  # Skróć nazwę
                    changes.append(f"• {name}: {orig_currency} → {target_currency}")
                    changes.append(f"  {original_results[isin]:6.2f}% → {results[isin]:6.2f}% ({change:+.2f}%)")
        
        if changes:
            output.append("\n*Wpływ kursu walutowego:*")
            output.extend(changes)
    
    # Ranking aktywów
    output.append("\n📊 *Ranking aktywów:*")
    for isin, momentum in sorted(results.items(), key=lambda x: x[1], reverse=True):
        name = ISINS[isin].split(' UCITS')[0]  # Skróć nazwę
        prices = price_data[isin]
        
        # Znajdź ceny
        start_price = prices[prices.index >= start_date].iloc[0]
        end_price = prices[prices.index <= analysis_date].iloc[-1]
        
        # Emoji dla trendu
        trend = "📈" if momentum > 0 else "📉"
        leader_mark = "👑 " if isin == leader else "   "
        
        output.append(f"\n{leader_mark}{trend} *{name}*")
        output.append(f"   {start_price:.2f} → {end_price:.2f} {currencies[isin]}")
        output.append(f"   Momentum: *{momentum:+.2f}%*")
        output.append(f"   {exchanges[isin]}")
    
    # Rekomendacja
    output.append("\n🎯 *Rekomendacja*")
    output.append(f"Lider: *{ISINS[leader].split(' UCITS')[0]}*")
    output.append(f"Momentum: *{results[leader]:+.2f}%*")
    
    # Data generowania
    output.append(f"\n🕒 Wygenerowano: {current_date.strftime('%Y-%m-%d %H:%M')}")
    
    message = '\n'.join(output)
    try:
        print(message)  # Dla debugowania
    except UnicodeEncodeError:
        print("Nie można wyświetlić wiadomości w konsoli (problemy z kodowaniem)")
    return message

def convert_prices(prices: pd.Series, from_currency: str, to_currency: str, start_date: str, end_date: str) -> pd.Series:
    """
    Przelicza ceny z jednej waluty na drugą.
    """
    if from_currency == to_currency:
        return prices
        
    # Pobierz kursy dla dat początku i końca
    start_rate = get_forex_rate(from_currency, to_currency, start_date)
    end_rate = get_forex_rate(from_currency, to_currency, end_date)
    
    if start_rate is None or end_rate is None:
        print(f"Nie można przeliczyć {from_currency} na {to_currency}")
        return prices
        
    # Interpoluj kurs liniowo między datami
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    rates = pd.Series(np.linspace(start_rate, end_rate, len(dates)), index=dates)
    
    # Przelicz ceny
    return prices * rates.reindex(prices.index, method='ffill')

def send_to_telegram(message: str, image: io.BytesIO) -> None:
    """
    Wysyła wiadomość i obraz na Telegram.
    """
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        raise ValueError("Brak tokenu bota lub ID czatu w pliku .env")
    
    bot = telebot.TeleBot(bot_token)
    
    try:
        # Wyślij wiadomość z formatowaniem Markdown
        bot.send_message(chat_id, message, parse_mode='Markdown')
        
        # Wyślij wykres
        image.seek(0)
        bot.send_photo(chat_id, image, caption="📈 Wykres porównawczy aktywów")
        
    except Exception as e:
        print(f"Błąd podczas wysyłania na Telegram: {str(e)}")

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Analiza momentum ETF-ów')
        parser.add_argument('--currency', choices=['original', 'PLN', 'USD', 'EUR'], default='original',
                          help='Waluta do wyświetlenia wyników (original, PLN, USD lub EUR)')
        args = parser.parse_args()
        
        leader, results, price_data, price_types, exchanges, currencies = get_current_leader()
        
        # Zachowaj oryginalne wyniki
        original_results = results.copy()
        original_currencies = currencies.copy()
        
        # Jeśli wybrano inną walutę niż oryginalna, przelicz ceny
        if args.currency != 'original':
            start_date = (datetime.now() - timedelta(days=365+31)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Zapisz oryginalne waluty
            original_currency_map = {isin + '_orig': currencies[isin] for isin in currencies}
            currencies.update(original_currency_map)
            
            for isin in price_data:
                if currencies[isin] != args.currency:
                    price_data[isin] = convert_prices(price_data[isin], 
                                                    currencies[isin], args.currency,
                                                    start_date, end_date)
                    currencies[isin] = args.currency  # Zaktualizuj walutę
                    
            # Przelicz momentum ponownie dla przeliczonych cen
            for isin in results:
                if not price_data[isin].empty:
                    results[isin] = calculate_momentum(price_data[isin])
            
            # Znajdź nowego lidera po przeliczeniu
            leader = max(results.items(), key=lambda x: x[1])[0]
        
        # Pobierz symbole dla wszystkich ISIN-ów
        symbols = {}
        for isin in results.keys():
            ticker, _, _ = get_ticker_for_isin(isin)
            symbols[isin] = ticker.split('.')[0] if ticker else 'N/A'
            
        # Generuj wyniki i wykres
        message = format_telegram_message(leader, results, price_data, exchanges, currencies,
                                        original_results=original_results if args.currency != 'original' else None,
                                        target_currency=args.currency)
        plot_image = plot_performance_comparison(price_data, symbols, results)
        
        # Wyślij na Telegram
        send_to_telegram(message, plot_image)
        
    except Exception as e:
        print(f"Wystąpił błąd: {str(e)}")