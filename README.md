# GEM Strategy 🤖

## Global Equity Momentum Strategy z automatycznym powiadamianiem

Implementacja strategii GEM (Global Equity Momentum) z automatycznym systemem powiadomień o konieczności wykonania DCA (Dollar-Cost Averaging) poprzez Telegram.

### 📊 O Strategii

GEM (Global Equity Momentum) to strategia inwestycyjna oparta na dwóch kluczowych elementach:
1. **Relative Momentum**: Wybór najsilniejszego aktywa z dostępnego uniwersum
2. **Absolute Momentum**: Porównanie lidera z bezpiecznym aktywem (obligacje skarbowe)

#### Monitorowane ETFy

##### Aktywa Ryzykowne
- `IE00BMFKG444` - Xtrackers NASDAQ 100 UCITS ETF 1C
- `IE0006WW1TQ4` - Xtrackers MSCI World ex USA UCITS ETF 1C USD

##### Aktywa Bezpieczne
- `IE00BYXPSP02` - iShares $ Treasury Bond 1-3yr UCITS ETF USD (Acc)
- `IE00B3VWN518` - iShares $ Treasury Bd 7-10y ETF USD Acc

### 🛠️ Komponenty Systemu

1. **gem.py**
   - Główna implementacja strategii GEM
   - Obliczanie momentum dla aktywów
   - Wybór lidera i analiza bezwzględnego momentum

2. **gem_notification.py**
   - System powiadomień przez Telegram
   - Generowanie raportów i wykresów
   - Obsługa różnych walut (PLN, USD, EUR)

3. **gem_scheduler.py**
   - Automatyczne uruchamianie analizy
   - Wysyłanie powiadomień pierwszego dnia każdego miesiąca
   - Monitorowanie i raportowanie statusu

### 📱 Powiadomienia

System automatycznie wysyła na Telegram:
- Analizę momentum dla wszystkich aktywów
- Rekomendację zakupu (lider)
- Wykres porównawczy wyników
- Wpływ zmian kursowych (jeśli wybrano inną walutę)

### 🚀 Instalacja

1. Klonowanie repozytorium:
```bash
git clone [repository_url]
cd GemStrategy
```

2. Instalacja zależności:
```bash
pip install -r requirements.txt
```

3. Konfiguracja pliku `.env`:
```env
EODHD_API_KEY=your_api_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 💻 Użycie

1. **Jednorazowa analiza**:
```bash
python gem_notification.py --currency PLN
```

2. **Uruchomienie schedulera**:
```bash
python gem_scheduler.py
```

### ⚙️ Konfiguracja jako Usługa Windows

1. Utwórz plik `start_gem_scheduler.bat`:
```batch
@echo off
cd /d [ścieżka_do_projektu]
python gem_scheduler.py
```

2. Dodaj skrót do folderu autostartu:
```
C:\Users\[username]\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

### 📈 Parametry Strategii

- Okres momentum: 12 miesięcy
- Gap period: 1 miesiąc
- Częstotliwość rebalansowania: miesięczna
- Metoda inwestowania: DCA (Dollar-Cost Averaging)

### 🔑 Wymagane API

1. **EOD Historical Data (EODHD)**
   - Dostęp do danych historycznych
   - Wyszukiwanie po ISIN
   - Kursy walut

2. **Telegram Bot API**
   - Wysyłanie powiadomień
   - Wykresy i raporty

### 📝 Uwagi

- System używa adjusted close prices (uwzględnia dywidendy)
- Wspiera przeliczanie wyników na różne waluty
- Automatycznie obsługuje dni wolne od handlu
- Wysyła powiadomienia o 9:00 rano pierwszego dnia każdego miesiąca

### 🤝 Wkład i Rozwój

Propozycje zmian i usprawnień są mile widziane! Możesz:
1. Zgłaszać problemy przez Issues
2. Proponować zmiany przez Pull Requests
3. Sugerować nowe funkcjonalności

### 📄 Licencja

MIT License