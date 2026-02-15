from metatrader5_config import TRADING_CONFIG

def get_legs(data, custom_threshold=None, verbose: bool=False):
    threshold = custom_threshold if custom_threshold else TRADING_CONFIG['threshold']
    if verbose:
        print(f'Using threshold: {threshold}')
        print('len(data): ', len(data))
        print(f'Start time: {data.index[0]}, End time: {data.index[-1]}')
    legs = []
    start_index = data.index[0]
    j = 0
    last_start_price = None
    i = 1

    while i < len(data):
        

    ##################          Current Price      ###############################################################   
        
        current_is_bullish = data['close'].iloc[i] >= data['open'].iloc[i]
        if j>0 and legs[j-1]['direction'] == 'up' and data['high'].iloc[i] >= data['high'].iloc[i-1]:
            current_price = data['high'].iloc[i]
        elif j>0 and legs[j-1]['direction'] == 'down' and data['low'].iloc[i] <= data['low'].iloc[i-1]:
            current_price = data['low'].iloc[i]
            
        else:
            current_price = data['high'].iloc[i] if current_is_bullish else data['low'].iloc[i]

    ##################          Start Price      ############################################################### 
        
        start_is_bullish = data['close'].loc[start_index] >= data['open'].loc[start_index]

        start_price = data['high'].loc[start_index] if start_is_bullish else data['low'].loc[start_index]

    ##################                ###############################################################   

        price_diff = abs(current_price - start_price) * 10000
        
        if j>0:
            timestamp_value = legs[j-1]['start']
            row = data.loc[timestamp_value]

        # تشخیص جهت بر اساس مقایسه با نقطه شروع و روند اخیر
        if data['high'].iloc[i] > data['high'].loc[start_index]:
            mydirection = 'up'
        elif data['low'].iloc[i] < data['low'].loc[start_index]:
            mydirection = 'down'
        # اگر هنوز جهت مشخص نشده، از روند کوتاه مدت استفاده کن
        else:
            if data['high'].iloc[i] > data['high'].iloc[i-1]:
                mydirection = 'up'
            elif data['low'].iloc[i] < data['low'].iloc[i-1]:
                mydirection = 'down'
            else:
                # اگر هنوز نامشخص است، از close استفاده کن
                mydirection = 'up' if data['close'].iloc[i] >= data['close'].iloc[i-1] else 'down'
            
        
        if price_diff >= threshold and price_diff < threshold * 5:
            
            direction = 'up' if data['high'].iloc[i] > data['high'].loc[start_index] or (data['high'].iloc[i] > data['high'].iloc[i-1] and data['close'].iloc[i] > data['open'].iloc[i]) else 'down'
            if j > 0 and legs[j-1]['direction'] == direction:
                    price_diff += legs[j-1]['length']
                    legs[j-1]['end'] = data.index[i]
                    legs[j-1]['end_value'] = current_price
                    legs[j-1]['length'] = price_diff
                    legs[j-1]['direction'] = direction
                    start_index = data.index[i]
                    

            elif len(data.loc[start_index:data.index[i]]) >= 3:
                if legs and len(legs) > 0:
                    timestamp_value = legs[-1]['end']
                    row = data.loc[timestamp_value]
                    
                    # اگر جهت جدید با جهت قبلی متفاوت است، از نقطه پایان قبلی شروع کنیم
                    if (mydirection != legs[-1]['direction'] and 
                        ((mydirection == 'up' and current_price > row['high']) or 
                         (mydirection == 'down' and current_price < row['low']))):
                        if legs[-1]['direction'] == 'up':
                            start_price = row['high']
                        else:
                            start_price = row['low']
                    else:
                        if legs[-1]['direction'] == 'up':
                            start_price = row['high']
                        else:
                            start_price = row['low']
                price_diff = abs(current_price - start_price) * 10000
                legs.append({
                    'start': start_index,
                    'start_value': start_price,
                    'end': data.index[i],
                    'end_value': current_price,
                    'length': price_diff,
                    'direction': direction,
                            })
                
                j += 1
                
                start_index = data.index[i]
            
        elif j>0 and legs[j-1]['direction'] == 'up' and data['high'].iloc[i] >= data['high'].loc[start_index] and price_diff < threshold:
            
            if j > 1 :
                price_diff = custom_price_diff(data=data, j=j, current_price=current_price, legs=legs)
                
            else:
                price_diff += legs[j-1]['length']
            
            
            start_index = data.index[i]
            legs[j-1]['end'] = data.index[i]
            legs[j-1]['end_value'] = current_price
            legs[j-1]['length'] = price_diff
            legs[j-1]['direction'] = direction
            
        elif j>0 and legs[j-1]['direction'] == 'down' and data['low'].iloc[i] <= data['low'].loc[start_index] and price_diff < threshold:
            
            if j > 1 :
                price_diff = custom_price_diff(data=data, j=j, current_price=current_price, legs=legs)
                
            else:
                price_diff += legs[j-1]['length']
            
            
            start_index = data.index[i]
            legs[j-1]['end'] = data.index[i]
            legs[j-1]['end_value'] = current_price
            legs[j-1]['length'] = price_diff
            
        
        last_start_price = start_price
        i += 1
    
    return legs


def custom_price_diff(data, j, current_price=0, legs=[]):
    
    timestamp_value = legs[j-2]['end']
    row = data.loc[timestamp_value]
    
    if legs[j-2]['direction'] == 'up':
        price_diff = abs(current_price - row['high']) * 10000
        return price_diff
    else:
        price_diff = abs(current_price - row['low']) * 10000
        return price_diff
