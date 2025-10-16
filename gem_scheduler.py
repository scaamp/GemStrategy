import schedule
import time
from datetime import datetime
import subprocess
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe
load_dotenv()

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
        return
        
    try:
        print(f"\n=== Uruchamiam GEM Strategy ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        
        # Uruchom skrypt gem_notification.py
        result = subprocess.run(
            ['python', 'gem_notification.py', '--currency', 'PLN'],
            capture_output=True,
            text=True
        )
        
        # Wyświetl output (nawet jeśli wystąpił błąd)
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        # Wyświetl błędy (jeśli wystąpiły)
        if result.stderr:
            print("Błędy:")
            print(result.stderr)
            
        # Sprawdź kod wyjścia
        if result.returncode != 0:
            print(f"Skrypt zakończył się z błędem (kod: {result.returncode})")
        else:
            print("Skrypt zakończył się pomyślnie")
            
    except Exception as e:
        print(f"Wystąpił błąd podczas uruchamiania skryptu: {str(e)}")

def main():
    """
    Główna funkcja schedulera
    """
    print("=== GEM Strategy Scheduler ===")
    print("Bot będzie uruchamiany:")
    print("- Na początku każdego miesiąca")
    print("- O godzinie 9:00 rano")
    print("\nCzekam na następne wykonanie...")
    
    # Zaplanuj wykonanie codziennie o 9:00
    # (funkcja should_run_today sprawdzi czy to pierwszy dzień miesiąca)
    schedule.every().day.at("09:00").do(run_gem_notification)
    
    # Uruchom od razu przy starcie jeśli:
    # - jest pierwszy dzień miesiąca
    # - jest przed 9:00 rano
    if should_run_today() and datetime.now().hour < 9:
        run_gem_notification()
    
    # Główna pętla
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Sprawdzaj co minutę
            
            # Wyświetl następne zaplanowane wykonanie
            next_run = None
            for job in schedule.get_jobs():
                if job.next_run:
                    next_run = job.next_run
                    break
                    
            if next_run:
                # Wyczyść linię i wyświetl czas do następnego wykonania
                print(f"\rNastępne wykonanie: {next_run.strftime('%Y-%m-%d %H:%M:%S')}   ", end='')
                
        except KeyboardInterrupt:
            print("\nZatrzymywanie schedulera...")
            break
        except Exception as e:
            print(f"\nWystąpił błąd: {str(e)}")
            print("Scheduler kontynuuje działanie...")
            time.sleep(60)

if __name__ == "__main__":
    main()