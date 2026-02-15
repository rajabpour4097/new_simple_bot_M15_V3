# ساعات مختلف بازار فارکس بر اساس ساعت ایران

# جلسه سیدنی (05:30 - 14:30 ایران)
SYDNEY_HOURS_IRAN = {
    'start': '05:30',
    'end': '14:30'
}

# جلسه توکیو (07:30 - 16:30 ایران)  
TOKYO_HOURS_IRAN = {
    'start': '07:30',
    'end': '16:30'
}

# جلسه لندن (12:30 - 21:30 ایران)
LONDON_HOURS_IRAN = {
    'start': '12:30',
    'end': '21:30'
}

# جلسه نیویورک (17:30 - 02:30 ایران)
NEWYORK_HOURS_IRAN = {
    'start': '17:30',
    'end': '02:30'  # روز بعد
}

# همپوشانی لندن-نیویورک (17:30 - 21:30 ایران) - بهترین زمان
OVERLAP_LONDON_NY_IRAN = {
    'start': '17:30',
    'end': '21:30'
}

# ساعات فعال ایرانی (09:00 - 21:00)
IRAN_ACTIVE_HOURS = {
    'start': '09:00',
    'end': '21:00'
}

# 24 ساعته
FULL_TIME_IRAN = {
    'start': '00:00',
    'end': '23:59'
}

MY_CUSTOM_TIME_IRAN = {
    'start': '01:00',
    'end': '23:59'
}

BEST_TIME_IRAN = {
    'start': '09:00',
    'end': '18:30'
}

# تنظیمات MT5
MT5_CONFIG = {
    'symbol': 'EURUSD',
    'lot_size': 0.01,
    'win_ratio': 2,
    'magic_number': 234000,
    'deviation': 20,
    'max_spread': 3.0,
    'min_balance': 1,
    'max_daily_trades': 10,
    'trading_hours': FULL_TIME_IRAN,
    'risk_percent': 0.02,  # 2% ریسک در هر معامله
}

# تنظیمات استراتژی
TRADING_CONFIG = {
    'threshold': 6,  # Changed from 6 to 60 to detect major legs (6 pips minimum)
    'fib_705': 0.705,
    'fib_90': 0.9,
    'window_size': 100,
    'min_swing_size': 4,
    'entry_tolerance': 2.0,
    'lookback_period': 20,
    # Optional: epsilon tolerance for 0.705 touch detection (in pips)
    # 'touch_epsilon_pips': 0.15,
    'prevent_multiple_positions': True,  # جلوگیری از باز کردن پوزیشن‌های متعدد همزمان
    'position_check_mode': 'all',  # 'all': همه پوزیشن‌ها، 'conflicting': فقط پوزیشن‌های مخالف
}

# مدیریت پویا چند مرحله‌ای جدید - 19 مرحله (2R تا 20R)
# مراحل بر اساس درخواست:
# 1) 2.0R: SL روی +2.0R، TP به 3.0R
# 2) 3.0R: SL روی +3.0R، TP به 4.0R
# 3) 4.0R: SL روی +4.0R، TP به 5.0R
# ... و همینطور تا 20R
DYNAMIC_RISK_CONFIG = {
    'enable': True,
    'commission_per_lot': 4.5,          # کمیسیون کل (رفت و برگشت یا فقط رفت؟ طبق بروکر - قابل تنظیم)
    'commission_mode': 'per_lot',       # per_lot (کل)، per_side (نیمی از رفت و برگشت) در صورت نیاز توسعه
    'round_trip': False,                # اگر True و per_side باشد دو برابر می‌کند
    'base_tp_R': 2.0,                   # TP اولیه تنظیم‌شده هنگام ورود (برای مرجع)
    'stages': [
        {  # 2.0R stage
            'id': 'stage_2_0R',
            'trigger_R': 2.0,
            'sl_lock_R': 2.0,
            'tp_R': 3.0
        },
        {  # 3.0R stage
            'id': 'stage_3_0R',
            'trigger_R': 3.0,
            'sl_lock_R': 3.0,
            'tp_R': 4.0
        },
        {  # 4.0R stage
            'id': 'stage_4_0R',
            'trigger_R': 4.0,
            'sl_lock_R': 4.0,
            'tp_R': 5.0
        },
        {  # 5.0R stage
            'id': 'stage_5_0R',
            'trigger_R': 5.0,
            'sl_lock_R': 5.0,
            'tp_R': 6.0
        },
        {  # 6.0R stage
            'id': 'stage_6_0R',
            'trigger_R': 6.0,
            'sl_lock_R': 6.0,
            'tp_R': 7.0
        },
        {  # 7.0R stage
            'id': 'stage_7_0R',
            'trigger_R': 7.0,
            'sl_lock_R': 7.0,
            'tp_R': 8.0
        },
        {  # 8.0R stage
            'id': 'stage_8_0R',
            'trigger_R': 8.0,
            'sl_lock_R': 8.0,
            'tp_R': 9.0
        },
        {  # 9.0R stage
            'id': 'stage_9_0R',
            'trigger_R': 9.0,
            'sl_lock_R': 9.0,
            'tp_R': 10.0
        },
        {  # 10.0R stage
            'id': 'stage_10_0R',
            'trigger_R': 10.0,
            'sl_lock_R': 10.0,
            'tp_R': 11.0
        },
        {  # 11.0R stage
            'id': 'stage_11_0R',
            'trigger_R': 11.0,
            'sl_lock_R': 11.0,
            'tp_R': 12.0
        },
        {  # 12.0R stage
            'id': 'stage_12_0R',
            'trigger_R': 12.0,
            'sl_lock_R': 12.0,
            'tp_R': 13.0
        },
        {  # 13.0R stage
            'id': 'stage_13_0R',
            'trigger_R': 13.0,
            'sl_lock_R': 13.0,
            'tp_R': 14.0
        },
        {  # 14.0R stage
            'id': 'stage_14_0R',
            'trigger_R': 14.0,
            'sl_lock_R': 14.0,
            'tp_R': 15.0
        },
        {  # 15.0R stage
            'id': 'stage_15_0R',
            'trigger_R': 15.0,
            'sl_lock_R': 15.0,
            'tp_R': 16.0
        },
        {  # 16.0R stage
            'id': 'stage_16_0R',
            'trigger_R': 16.0,
            'sl_lock_R': 16.0,
            'tp_R': 17.0
        },
        {  # 17.0R stage
            'id': 'stage_17_0R',
            'trigger_R': 17.0,
            'sl_lock_R': 17.0,
            'tp_R': 18.0
        },
        {  # 18.0R stage
            'id': 'stage_18_0R',
            'trigger_R': 18.0,
            'sl_lock_R': 18.0,
            'tp_R': 19.0
        },
        {  # 19.0R stage
            'id': 'stage_19_0R',
            'trigger_R': 19.0,
            'sl_lock_R': 19.0,
            'tp_R': 20.0
        },
        {  # 20.0R stage (final)
            'id': 'stage_20_0R',
            'trigger_R': 20.0,
            'sl_lock_R': 20.0,
            'tp_R': 20.0
        }
    ]
}

# تنظیمات لاگ
LOG_CONFIG = {
    'log_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR
    'save_to_file': True,       # ذخیره در فایل
    'max_log_size': 10,         # حداکثر حجم فایل لاگ (MB)
}
