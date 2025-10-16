import schedule
import time
from datetime import datetime
import subprocess
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Ustaw ścieżki bazowe
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / 'logs'
LOG_FILE = LOG_DIR / 'gem_strategy.log'

# Utwórz katalog na logi jeśli nie istnieje
LOG_DIR.mkdir(exist_ok=True)

# Konfiguracja logowania
logging.basicConfig(
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=100000, backupCount=5)],
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Załaduj zmienne środowiskowe
load_dotenv(BASE_DIR / '.env')
logger.info(f"Uruchamiam z katalogu: {BASE_DIR}")

def get_next_first_day():
    """Znajduje następny pierwszy dzień miesiąca"""
    current = datetime.now()
    if current.day == 1 and current.hour < 9:
        return current
    # Jeśli dziś nie jest 1. lub jest po 9:00, przejdź do następnego miesiąca
    if current.month == 12:
        next_date = current.replace(year=current.year + 1, month=1, day=1)
    else:
        next_date = current.replace(month=current.month + 1, day=1)
    return next_date

def should_run_today():
    """
    Sprawdza, czy dzisiaj powinniśmy uruchomić skrypt
    (1. dzień miesiąca)
    """
    return datetime.now().day == 1

def run_gem_notification():
    """
    Uruchamia skrypt gem_notification.py z odpowiednimi parametrami
    """
    # Sprawdź czy to pierwszy dzień miesiąca
    if not should_run_today():
        logger.debug("Nie jest pierwszy dzień miesiąca - pomijam wykonanie")
        return
        
    try:
        logger.info(f"=== Uruchamiam GEM Strategy ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        
        # Użyj pełnej ścieżki do Pythona i skryptu
        python_path = Path(os.environ.get('VIRTUAL_ENV', '')) / 'bin' / 'python'
        if not python_path.exists():
            python_path = 'python'  # Fallback na systemowego Pythona
            
        script_path = BASE_DIR / 'gem_notification.py'
        
        result = subprocess.run(
            [str(python_path), str(script_path), '--currency', 'PLN'],
            capture_output=True,
            text=True
        )
        
        # Loguj output
        if result.stdout:
            logger.info("Output:")
            logger.info(result.stdout)
            
        # Loguj błędy
        if result.stderr:
            logger.error("Błędy:")
            logger.error(result.stderr)
            
        # Sprawdź kod wyjścia
        if result.returncode != 0:
            logger.error(f"Skrypt zakończył się z błędem (kod: {result.returncode})")
        else:
            logger.info("Skrypt zakończył się pomyślnie")
            
    except Exception as e:
        logger.exception(f"Wystąpił błąd podczas uruchamiania skryptu: {str(e)}")

def main():
    """
    Główna funkcja schedulera
    """
    logger.info("=== GEM Strategy Scheduler ===")
    logger.info("Bot będzie uruchamiany: pierwszy dzień każdego miesiąca o 9:00")
    
    # Zaplanuj wykonanie na pierwszy dzień każdego miesiąca o 9:00
    next_run = get_next_first_day()
    schedule.every().day.at("09:00").do(run_gem_notification)
    logger.info(f"Status: Aktywny | Następne wykonanie: {next_run.strftime('%Y-%m-%d')} 09:00")
    
    # Uruchom od razu przy starcie jeśli:
    # - jest pierwszy dzień miesiąca
    # - jest przed 9:00 rano
    if should_run_today() and datetime.now().hour < 9:
        logger.info("Uruchamiam natychmiast (pierwszy dzień miesiąca, przed 9:00)")
        run_gem_notification()
    
    # Główna pętla
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Sprawdzaj co minutę
            
            # Wyświetl status co godzinę
            current_minute = datetime.now().minute
            if current_minute == 0:
                next_run = get_next_first_day()
                logger.info(f"Status: Aktywny | Następne wykonanie: {next_run.strftime('%Y-%m-%d')} 09:00")
                
        except KeyboardInterrupt:
            logger.info("Zatrzymywanie schedulera...")
            break
        except Exception as e:
            logger.exception(f"Wystąpił błąd: {str(e)}")
            logger.info("Scheduler kontynuuje działanie...")
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Krytyczny błąd w main():")
        raise