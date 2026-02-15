from datetime import datetime
from colorama import init, Fore

# راه‌اندازی colorama
init(autoreset=True)

def log(msg, level='info', color=None, save_to_file=True):
    color_prefix = getattr(Fore, color.upper(), '') if color else ''
    print(f"{color_prefix}{msg}")

    if save_to_file:
        log_filename = f"swing_logs_{datetime.now().strftime('%Y-%m-%d')}.txt"
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(f"{msg}\n")
        except Exception as e:
            print(f"خطا در ذخیره لاگ: {e}")