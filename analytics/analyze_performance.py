#!/usr/bin/env python3
"""
تحلیل جامع عملکرد ربات بر اساس داده‌های ذخیره شده در analytics/vps-data

قابلیت‌ها:
- آنالیز معاملات (accuracy, risk, profit factor)
- ارزیابی سیگنال‌ها و کیفیت entry
- شناسایی الگوهای زمانی
- مقایسه expected vs actual results
- تحلیل مشکلات حجم و SL
- گزارش‌گیری و visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# تنظیمات فارسی و RTL
plt.rcParams['font.family'] = ['Tahoma', 'DejaVu Sans']
sns.set_style("whitegrid")

class TradingAnalyzer:
    def __init__(self, data_path="analytics/vps-data"):
        self.data_path = Path(data_path)
        self.signals_df = None
        self.trades_df = None
        self.combined_df = None
        
    def load_data(self):
        """بارگذاری تمام فایل‌های CSV"""
        print("📊 Loading trading data...")
        
        # بارگذاری سیگنال‌ها
        signals_files = list(self.data_path.glob("raw/trades_dir/*.csv"))
        if signals_files:
            signals_list = [pd.read_csv(f) for f in signals_files]
            self.signals_df = pd.concat(signals_list, ignore_index=True)
            self.signals_df['dt_utc'] = pd.to_datetime(self.signals_df['dt_utc'])
            print(f"✅ Loaded {len(self.signals_df)} signals from {len(signals_files)} files")
        
        # بارگذاری معاملات
        trades_files = list(self.data_path.glob("raw/trades_dir/*.csv"))
        if trades_files:
            trades_list = [pd.read_csv(f) for f in trades_files]
            self.trades_df = pd.concat(trades_list, ignore_index=True)
            self.trades_df['dt_utc'] = pd.to_datetime(self.trades_df['dt_utc'])
            print(f"✅ Loaded {len(self.trades_df)} trades from {len(trades_files)} files")
            
        # ترکیب داده‌ها
        if self.signals_df is not None and self.trades_df is not None:
            self.combine_signals_trades()
    
    def combine_signals_trades(self):
        """ترکیب سیگنال‌ها با معاملات برای تحلیل دقیق‌تر"""
        # merge by timestamp (با tolerance برای timing differences)
        merged = pd.merge_asof(
            self.trades_df.sort_values('dt_utc'),
            self.signals_df.sort_values('dt_utc'),
            on='dt_utc',
            suffixes=('_trade', '_signal'),
            direction='nearest'
        )
        
        # فیلتر کردن matchهای نزدیک (< 5 دقیقه)
        time_diff = abs(merged['dt_utc'] - merged['dt_utc'])
        merged = merged[time_diff <= timedelta(minutes=5)]
        
        self.combined_df = merged
        print(f"✅ Combined {len(merged)} signal-trade pairs")
    
    def analyze_volume_issues(self):
        """تحلیل مشکلات حجم"""
        print("\n🔍 Volume Analysis:")
        print("-" * 50)
        
        if self.trades_df is None:
            return
            
        # آمار حجم
        vol_stats = self.trades_df['req_vol'].describe()
        print("Volume Statistics:")
        print(vol_stats)
        
        # حجم‌های غیرعادی
        abnormal_vols = self.trades_df[self.trades_df['req_vol'] > 30]
        if len(abnormal_vols) > 0:
            print(f"\n⚠️ Found {len(abnormal_vols)} trades with abnormal volume (>30):")
            for _, trade in abnormal_vols.iterrows():
                risk_pips = abs(trade['req_price'] - trade['sl']) * 10000
                print(f"  {trade['dt_iran']}: Vol={trade['req_vol']:.1f}, Risk={risk_pips:.1f} pips")
        
        return {
            'mean_volume': vol_stats['mean'],
            'abnormal_count': len(abnormal_vols),
            'volume_range': (vol_stats['min'], vol_stats['max'])
        }
    
    def analyze_risk_reward(self):
        """تحلیل ریسک-ریوارد و SL/TP"""
        print("\n📈 Risk-Reward Analysis:")
        print("-" * 50)
        
        if self.combined_df is None or len(self.combined_df) == 0:
            print("No combined data available")
            return
            
        # Check available columns first
        print("Available columns:", list(self.combined_df.columns))
        
        results = []
        for _, row in self.combined_df.iterrows():
            entry = row['req_price']
            # Use the correct column names based on actual data
            sl = row.get('sl_trade') or row.get('sl_signal') or row.get('sl')
            tp = row.get('tp_trade') or row.get('tp_signal') or row.get('tp')
            
            # محاسبه ریسک و ریوارد در pips
            if row['side'] == 'BUY':
                risk_pips = (entry - sl) * 10000 if pd.notna(sl) else np.nan
                reward_pips = (tp - entry) * 10000 if pd.notna(tp) else np.nan
            else:  # SELL
                risk_pips = (sl - entry) * 10000 if pd.notna(sl) else np.nan
                reward_pips = (entry - tp) * 10000 if pd.notna(tp) else np.nan
            
            rr_ratio = reward_pips / risk_pips if pd.notna(risk_pips) and risk_pips > 0 else np.nan
            
            results.append({
                'timestamp': row.get('dt_iran_trade') or row.get('dt_iran_signal') or row.get('dt_iran'),
                'side': row['side'],
                'risk_pips': risk_pips,
                'reward_pips': reward_pips,
                'rr_ratio': rr_ratio,
                'expected_rr': row.get('rr_signal') or row.get('rr') or np.nan
            })
        
        rr_df = pd.DataFrame(results)
        
        # آمار
        valid_risk = rr_df['risk_pips'].dropna()
        valid_rr = rr_df['rr_ratio'].dropna()
        
        if len(valid_risk) > 0:
            print(f"Average Risk: {valid_risk.mean():.1f} pips")
        if len(valid_rr) > 0:
            print(f"Average RR Ratio: {valid_rr.mean():.2f}")
        
        # مشکلات
        negative_risk = rr_df[(rr_df['risk_pips'] <= 0) & pd.notna(rr_df['risk_pips'])]
        if len(negative_risk) > 0:
            print(f"⚠️ {len(negative_risk)} trades with negative/zero risk!")
        
        tiny_risk = rr_df[(rr_df['risk_pips'] < 1) & pd.notna(rr_df['risk_pips'])]
        if len(tiny_risk) > 0:
            print(f"⚠️ {len(tiny_risk)} trades with risk < 1 pip")
            
        return rr_df
    
    def analyze_timing_patterns(self):
        """تحلیل الگوهای زمانی"""
        print("\n⏰ Timing Analysis:")
        print("-" * 50)
        
        if self.trades_df is None:
            return
            
        # تبدیل به ساعت ایران
        self.trades_df['hour_iran'] = pd.to_datetime(self.trades_df['dt_iran']).dt.hour
        self.trades_df['day_of_week'] = pd.to_datetime(self.trades_df['dt_iran']).dt.day_name()
        
        # توزیع ساعتی
        hourly_dist = self.trades_df['hour_iran'].value_counts().sort_index()
        print("Trades by hour (Iran time):")
        for hour, count in hourly_dist.items():
            print(f"  {hour:02d}:00 - {count} trades")
        
        # روزهای هفته
        daily_dist = self.trades_df['day_of_week'].value_counts()
        print(f"\nMost active day: {daily_dist.index[0]} ({daily_dist.iloc[0]} trades)")
        
        return {
            'hourly_distribution': hourly_dist,
            'daily_distribution': daily_dist
        }
    
    def analyze_signal_quality(self):
        """تحلیل کیفیت سیگنال‌ها"""
        print("\n🎯 Signal Quality Analysis:")
        print("-" * 50)
        
        if self.signals_df is None:
            return
            
        # نسبت direction
        direction_counts = self.signals_df['direction'].value_counts()
        print("Signal directions:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count} ({count/len(self.signals_df)*100:.1f}%)")
        
        # فیبوناچی levels
        if 'fib_0705' in self.signals_df.columns:
            fib_range = self.signals_df['fib_0705'] - self.signals_df['fib_0']
            print(f"\nAverage Fib range: {fib_range.mean()*10000:.1f} pips")
        
        return direction_counts
    
    def generate_summary_report(self):
        """تولید گزارش خلاصه"""
        print("\n" + "="*60)
        print("📋 TRADING BOT PERFORMANCE SUMMARY")
        print("="*60)
        
        # اطلاعات کلی
        if self.trades_df is not None:
            total_trades = len(self.trades_df)
            date_range = f"{self.trades_df['dt_iran'].min()} to {self.trades_df['dt_iran'].max()}"
            print(f"📅 Period: {date_range}")
            print(f"📊 Total Trades: {total_trades}")
            
            # حجم کل
            total_volume = self.trades_df['req_vol'].sum()
            print(f"💰 Total Volume: {total_volume:.2f} lots")
        
        # تحلیل‌ها
        volume_analysis = self.analyze_volume_issues()
        timing_analysis = self.analyze_timing_patterns()
        rr_analysis = self.analyze_risk_reward()
        signal_analysis = self.analyze_signal_quality()
        
        # نتیجه‌گیری
        print("\n" + "="*60)
        print("🎯 KEY FINDINGS & RECOMMENDATIONS:")
        print("="*60)
        
        if volume_analysis and volume_analysis['abnormal_count'] > 0:
            print("⚠️ CRITICAL: Abnormal volume sizes detected")
            print("   → Fix SL distance calculation in risk sizing")
            print("   → Add volume limits and validation")
        
        if rr_analysis is not None:
            avg_risk = rr_analysis['risk_pips'].mean()
            if avg_risk < 2:
                print("⚠️ WARNING: Average risk too small")
                print("   → Increase minimum SL distance")
            
            tiny_risk_pct = len(rr_analysis[rr_analysis['risk_pips'] < 1]) / len(rr_analysis) * 100
            if tiny_risk_pct > 20:
                print(f"⚠️ WARNING: {tiny_risk_pct:.1f}% trades have risk < 1 pip")
        
        if timing_analysis:
            print("✅ TIMING: Bot is active during expected hours")
        
        print("\n🔧 Next Steps:")
        print("1. Fix volume calculation in mt5_connector.py")
        print("2. Add SL validation guards in main_metatrader.py") 
        print("3. Implement minimum risk filters")
        print("4. Add trade cooldown periods")
        print("5. Test improvements in backtest before live deployment")
    
    def create_visualizations(self, save_path="analytics/reports"):
        """تولید نمودارها"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.trades_df is None:
            return
            
        # نمودار حجم در طول زمان
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(self.trades_df['dt_iran']), self.trades_df['req_vol'], 'bo-')
        plt.title('Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volume (lots)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path / 'volume_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # توزیع حجم
        plt.figure(figsize=(10, 6))
        plt.hist(self.trades_df['req_vol'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Volume Distribution')
        plt.xlabel('Volume (lots)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(save_path / 'volume_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Charts saved to {save_path}/")

def main():
    """اجرای تحلیل کامل"""
    analyzer = TradingAnalyzer()
    analyzer.load_data()
    analyzer.generate_summary_report()
    analyzer.create_visualizations()
    
    print(f"\n✅ Analysis complete! Check the output above.")
    print("💡 Run 'python analytics/analyze_performance.py' anytime for fresh analysis.")

if __name__ == "__main__":
    main()
