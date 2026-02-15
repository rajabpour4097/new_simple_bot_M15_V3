import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime, time
from metatrader5_config import MT5_CONFIG
from analytics.hooks import log_market, log_trade, log_position_event

RET_OK = 10009  # mt5.TRADE_RETCODE_DONE

class MT5Connector:
    def __init__(self):
        cfg = MT5_CONFIG
        self.symbol = cfg['symbol']
        self.lot = cfg['lot_size']
        self.deviation = cfg['deviation']
        self.magic = cfg['magic_number']
        self.max_spread = cfg['max_spread']
        self.min_balance = cfg['min_balance']
        self.trading_hours = cfg['trading_hours']
        # کمیسیون هر سمت (per-side) به‌ازای هر 1 لات، واحد: ارز حساب. اگر ندادی 0.
        # self.commission_per_lot_side = cfg.get('commission_per_lot_side', 0.0)  # removed
        self.iran_tz = pytz.timezone('Asia/Tehran')
        self.utc_tz = pytz.UTC

    # ---------- Time / Session ----------
    def get_iran_time(self):
        return datetime.now(self.utc_tz).astimezone(self.iran_tz)

    def is_trading_time(self):
        start = time.fromisoformat(self.trading_hours['start'])
        end = time.fromisoformat(self.trading_hours['end'])
        now_t = self.get_iran_time().time()
        if start <= end:
            return start <= now_t <= end
        # window passes midnight
        return now_t >= start or now_t <= end

    def check_weekend(self):
        # Forex shuts late Fri (server time). Simplified: block Saturday/Sunday
        wd = self.get_iran_time().weekday()  # Monday=0
        return wd not in (5, 6)  # 5=Saturday,6=Sunday (adjust if broker different)

    def can_trade(self):
        if not self.check_weekend():
            return False, "Weekend - trading disabled"
        if not self.is_trading_time():
            return False, "Outside configured trading hours"
        ti = mt5.terminal_info()
        if not ti:
            return False, "Terminal info unavailable"
        if not ti.trade_allowed:
            return False, "Terminal AutoTrading disabled"
        acc = mt5.account_info()
        if not acc:
            return False, "Account info unavailable"
        if acc.balance < self.min_balance:
            return False, f"Insufficient balance < {self.min_balance}"
        return True, "Trading is allowed"

    # ---------- Initialization ----------
    def initialize(self):
        if not mt5.initialize():
            print("❌ MT5 initialize failed:", mt5.last_error())
            return False
        acc = mt5.account_info()
        if acc and acc.balance < self.min_balance:
            print(f"❌ Balance {acc.balance} < min {self.min_balance}")
            return False
        print("✅ MT5 connection established")
        return True

    def shutdown(self):
        mt5.shutdown()

    # ---------- Data ----------
    def get_live_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return None
        # try logging market tick
        try:
            info = mt5.symbol_info(self.symbol)
            if info:
                log_market(self.symbol, getattr(tick, "bid", None), getattr(tick, "ask", None),
                           getattr(tick, "last", None), info.point, info.digits, source="mt5", session="bot")
        except Exception:
            pass
        spread = (tick.ask - tick.bid) * 10000
        if spread > self.max_spread:
            print(f"⚠️ Spread {spread:.1f} > max {self.max_spread}")
        utc_time = datetime.fromtimestamp(tick.time, tz=self.utc_tz)
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': spread,
            'time': utc_time.astimezone(self.iran_tz),
            'utc_time': utc_time
        }

    def get_historical_data(self, timeframe=mt5.TIMEFRAME_M1, count=500):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(self.iran_tz)
        df.set_index('time', inplace=True)
        df = df.rename(columns={'tick_volume': 'volume'})
        df['timestamp'] = df.index
        return df

    # ---------- Broker capability helpers ----------
    def test_filling_modes(self):
        info = mt5.symbol_info(self.symbol)
        if not info:
            print("Symbol info not available")
            return None
        print(f"Filling mode raw: {info.filling_mode}")
        return info.filling_mode

    def get_supported_filling_modes(self):
        info = mt5.symbol_info(self.symbol)
        if not info:
            return []
        fm = getattr(info, 'filling_mode', 0)
        modes = []
        for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            try:
                # برخی بروکرها bitmask می‌دهند
                if (fm & m) == m:
                    modes.append(m)
            except Exception:
                # برخی فقط یک مقدار منفرد می‌دهند
                if fm == m:
                    modes.append(m)
        return modes

    def try_all_filling_modes(self, request):
        tried = []
        modes = self.get_supported_filling_modes()

        # 1) اول مدهای اعلام‌شده‌ی بروکر
        for m in modes:
            req = dict(request)
            req["type_filling"] = m
            res = mt5.order_send(req)
            tried.append((m, getattr(res, 'retcode', None)))
            if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
                return res

        # 2) یک بار بدون type_filling (auto)
        req = dict(request)
        req.pop("type_filling", None)
        res = mt5.order_send(req)
        tried.append(("auto", getattr(res, 'retcode', None)))
        if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
            return res

        # 3) در نهایت brute-force برای حالتی که flags نادرست گزارش شده
        for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            if m in modes:
                continue
            req = dict(request)
            req["type_filling"] = m
            res = mt5.order_send(req)
            tried.append((m, getattr(res, 'retcode', None)))
            if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
                return res

        print(f"[order_send] filling mode attempts: {tried}")
        return res  # آخرین نتیجه

    # ---------- Stop validation ----------
    def calculate_valid_stops(self, entry_price, sl_price, tp_price, order_type):
        """
        Validate stops:
        - حداقل فاصله استاپ از نقطه ورود: دقیقا >= 1 pip (اگر کمتر باشد سفارش رد می‌شود)
        - 1 pip = 10 * point برای نمادهای 5 یا 3 رقمی، در غیر این صورت = point
        - هیچ تغییری روی SL/TP اعمال نمی‌شود؛ فقط در صورت نامعتبر بودن None برمی‌گرداند.
        """
        info = mt5.symbol_info(self.symbol)
        if not info:
            print("Symbol info unavailable")
            return None, None
        point = info.point
        pip_size = 10 * point if info.digits in (3, 5) else point

        # اعتبار جهت SL
        if order_type == mt5.ORDER_TYPE_BUY and sl_price >= entry_price:
            print("❌ SL برای BUY باید پایین‌تر از ورود باشد")
            return None, None
        if order_type == mt5.ORDER_TYPE_SELL and sl_price <= entry_price:
            print("❌ SL برای SELL باید بالاتر از ورود باشد")
            return None, None

        distance = abs(entry_price - sl_price)
        if distance + 1e-12 <= pip_size:
            print(f"❌ فاصله SL ({distance:.{info.digits}f}) < 1 pip ({pip_size}) — سفارش ارسال نمی‌شود")
            return None, None

        # اعتبار ساده جهت TP (اختیاری: فقط اگر خلاف جهت باشد رد می‌کنیم)
        if tp_price is not None:
            if order_type == mt5.ORDER_TYPE_BUY and tp_price <= entry_price:
                print("❌ TP برای BUY باید بالاتر از ورود باشد")
                return None, None
            if order_type == mt5.ORDER_TYPE_SELL and tp_price >= entry_price:
                print("❌ TP برای SELL باید پایین‌تر از ورود باشد")
                return None, None

        def norm(p):
            return float(f"{p:.{info.digits}f}")

        return norm(sl_price), norm(tp_price)

    # ---------- Order sending core ----------
    def try_all_filling_modes(self, request):
        tried = []
        modes = self.get_supported_filling_modes()

        # 1) اول مدهای اعلام‌شده‌ی بروکر
        for m in modes:
            req = dict(request)
            req["type_filling"] = m
            res = mt5.order_send(req)
            tried.append((m, getattr(res, 'retcode', None)))
            if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
                return res

        # 2) یک بار بدون type_filling (auto)
        req = dict(request)
        req.pop("type_filling", None)
        res = mt5.order_send(req)
        tried.append(("auto", getattr(res, 'retcode', None)))
        if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
            return res

        # 3) در نهایت brute-force برای حالتی که flags نادرست گزارش شده
        for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            if m in modes:
                continue
            req = dict(request)
            req["type_filling"] = m
            res = mt5.order_send(req)
            tried.append((m, getattr(res, 'retcode', None)))
            if res and res.retcode in (RET_OK, mt5.TRADE_RETCODE_PLACED):
                return res

        print(f"[order_send] filling mode attempts: {tried}")
        return res  # آخرین نتیجه

    # ---------- Trading ----------
    def open_buy_position(self, tick, sl, tp, comment="", volume=None, risk_pct=None):
        if not tick:
            print("No tick data")
            return None
        entry = tick.ask
        sl_adj, tp_adj = self.calculate_valid_stops(entry, sl, tp, mt5.ORDER_TYPE_BUY)
        if sl_adj is None:
            return None
        vol = self._resolve_volume(volume, entry, sl_adj, tick, risk_pct)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": vol,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry,
            "sl": sl_adj,
            "tp": tp_adj,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        print(f"📤 BUY {self.symbol} @ {entry} VOL={vol} SL={sl_adj} TP={tp_adj}")
        result = self.try_all_filling_modes(request)
        try:
            log_trade(self.symbol, "BUY", request, result, reason="strategy_signal")
            if result and getattr(result, 'retcode', None) == RET_OK:
                # ثبت رویداد باز شدن پوزیشن (خلاصه؛ مدیریت دقیق در main)
                log_position_event(
                    symbol=self.symbol,
                    ticket=getattr(result, 'order', 0),
                    event='open_order',
                    direction='buy',
                    entry=entry,
                    current_price=entry,
                    sl=sl_adj,
                    tp=tp_adj,
                    profit_R=0.0,
                    stage=0,
                    risk_abs=abs(entry - sl_adj),
                    locked_R=None,
                    volume=request.get('volume'),
                    note='initial order'
                )
        except Exception:
            pass
        return result

    def open_sell_position(self, tick, sl, tp, comment="", volume=None, risk_pct=None):
        if not tick:
            print("No tick data")
            return None
        entry = tick.bid
        sl_adj, tp_adj = self.calculate_valid_stops(entry, sl, tp, mt5.ORDER_TYPE_SELL)
        if sl_adj is None:
            return None
        vol = self._resolve_volume(volume, entry, sl_adj, tick, risk_pct)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": vol,
            "type": mt5.ORDER_TYPE_SELL,
            "price": entry,
            "sl": sl_adj,
            "tp": tp_adj,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        print(f"📤 SELL {self.symbol} @ {entry} VOL={vol} SL={sl_adj} TP={tp_adj}")
        result = self.try_all_filling_modes(request)
        try:
            log_trade(self.symbol, "SELL", request, result, reason="strategy_signal")
            if result and getattr(result, 'retcode', None) == RET_OK:
                log_position_event(
                    symbol=self.symbol,
                    ticket=getattr(result, 'order', 0),
                    event='open_order',
                    direction='sell',
                    entry=entry,
                    current_price=entry,
                    sl=sl_adj,
                    tp=tp_adj,
                    profit_R=0.0,
                    stage=0,
                    risk_abs=abs(entry - sl_adj),
                    locked_R=None,
                    volume=request.get('volume'),
                    note='initial order'
                )
        except Exception:
            pass
        return result

    def close_all_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
        for pos in positions:
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                continue
            if pos.type == mt5.POSITION_TYPE_BUY:
                price = tick.bid  # close BUY at bid with SELL
                order_type = mt5.ORDER_TYPE_SELL
            else:
                price = tick.ask  # close SELL at ask with BUY
                order_type = mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(request)

    def get_positions(self):
        return mt5.positions_get(symbol=self.symbol)

    # ---------- Diagnostic stubs (used by main/tests) ----------
    def check_trading_limits(self):
        return True

    def check_account_trading_permissions(self):
        return True

    def check_market_state(self):
        return True

    def check_symbol_properties(self):
        info = mt5.symbol_info(self.symbol)
        if not info:
            print("Symbol info not found")
            return
        if not info.visible:
            mt5.symbol_select(self.symbol, True)

    # ---------- Volume helpers ----------
    def _normalize_volume(self, vol: float) -> float:
        info = mt5.symbol_info(self.symbol)
        if not info:
            return vol
        step = info.volume_step or 0.01
        vmin = info.volume_min or step
        vmax = info.volume_max or 100.0
        steps = round(vol / step)
        vol_rounded = steps * step
        return max(vmin, min(vmax, vol_rounded))

    def _get_tick_specs(self, info):
        """
        Resolve tick_size and tick_value with safe fallbacks:
        - Prefer trade_tick_size/trade_tick_value
        - Fallback to tick_size/tick_value if broker exposes them
        - Finally fallback to point and contract-size approximation
        """
        tick_size = getattr(info, 'trade_tick_size', None) or getattr(info, 'tick_size', None) or getattr(info, 'point', None)
        tick_value = getattr(info, 'trade_tick_value', None) or getattr(info, 'tick_value', None)
        if tick_value is None:
            contract = getattr(info, 'trade_contract_size', None)
            if contract and tick_size:
                # Approximation: value of one tick_size move for 1 lot in account currency
                # Accurate for USD-quoted pairs on USD accounts (e.g., EURUSD/USD account).
                tick_value = contract * tick_size
        return tick_size, tick_value

    def calculate_volume_by_risk(self, entry: float, sl: float, tick, risk_pct: float = 0.02) -> float:
        """Position sizing with price risk + current spread (commission removed)."""
        acc = mt5.account_info()
        info = mt5.symbol_info(self.symbol)
        if not acc or not info:
            return self.lot

        tick_size, tick_value = self._get_tick_specs(info)
        if not tick_size or not tick_value:
            return self.lot

        risk_money = acc.balance * float(risk_pct)

        risk_points = abs(entry - sl) / float(tick_size)
        price_risk_per_lot = risk_points * float(tick_value)

        spread_points = abs(getattr(tick, 'ask', 0.0) - getattr(tick, 'bid', 0.0)) / float(tick_size)
        spread_cost_per_lot = spread_points * float(tick_value)

        # commission removed
        total_cost_per_lot = price_risk_per_lot + spread_cost_per_lot
        if total_cost_per_lot <= 0:
            return self.lot

        vol = risk_money / total_cost_per_lot

        MAX_LEVERAGE_FACTOR = 0.02  # حداکثر 2% اکوییتی در ریسک قیمت
        # اگر ریسک پولی هر 1 لات خیلی کوچک شده و vol بسیار بزرگ است، کلمپ کن
        theoretical_loss_per_lot = price_risk_per_lot
        if theoretical_loss_per_lot <= 0:
            return self.lot
        max_allowed_vol = (acc.balance * MAX_LEVERAGE_FACTOR) / theoretical_loss_per_lot
        if vol > max_allowed_vol:
            vol = max_allowed_vol

        return self._normalize_volume(vol)

    def _resolve_volume(self, volume, entry, sl, tick, risk_pct):
        if volume is not None:
            return self._normalize_volume(volume)
        if risk_pct is not None:
            return self.calculate_volume_by_risk(entry, sl, tick, risk_pct)
        return self.lot

    # ---------- Modify SL/TP ----------
    def modify_sl_tp(self, ticket: int, new_sl=None, new_tp=None):
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
        }
        if new_sl is not None:
            req["sl"] = new_sl
        if new_tp is not None:
            req["tp"] = new_tp
        res = mt5.order_send(req)
        return res