from colorama import Fore


def get_swing_points(data, legs):
    if len(legs) == 3:
        
        s_index = 0
        swing_type = ''
        is_swing = False
        ### Up swing ###
        if legs[1]['end_value'] > legs[0]['start_value'] and legs[0]['end_value'] > legs[1]['end_value']:
            
            ### Chek true swing ###
            s_index = data.index.tolist().index(legs[1]['start'])
            e_index = data.index.tolist().index(legs[1]['end'])
            true_candles = 0
            first_candle = False

            for k in range(s_index, e_index+1):  # Check the current poolback for have 3 bearish candles
                
                if data.iloc[k]['status'] == 'bearish':   # If current candle is bearish for check swing
                    
                    if first_candle:  # If first candle of poolback
                        if data.iloc[k]['close'] < last_candle_close:  # If last close is less than current candle close
                            true_candles += 1
                            last_candle_close = data.iloc[k]['close']
                            
                    else:  # If not first candle of poolback give value
                        last_candle_close = data.iloc[k]['close']
                    
                    first_candle = True
            
            if true_candles >= 3:
                swing_type = 'bullish'
                is_swing = True
            
        ### Down swing ###
        elif legs[1]['end_value'] < legs[0]['start_value'] and legs[0]['end_value'] < legs[1]['end_value']:

            ### Chek true swing ###
            s_index = data.index.tolist().index(legs[1]['start'])
            e_index = data.index.tolist().index(legs[1]['end'])
            true_candles = 0
            first_candle = False

            for k in range(s_index, e_index+1):  # Check the current poolback for have 3 bullish candles
                
                if data.iloc[k]['status'] == 'bullish':   # If current candle is bullish for check swing
                    
                    if first_candle:  # If first candle of poolback
                        if data.iloc[k]['close'] > last_candle_close:  # If last close is more than current candle close
                            true_candles += 1
                            last_candle_close = data.iloc[k]['close']
                            
                    else:  # If not first candle of poolback give value
                        last_candle_close = data.iloc[k]['close']
                    
                    first_candle = True
                
            if true_candles >= 3:
                swing_type = 'bearish'
                is_swing = True

        return swing_type, is_swing