def fibonacci_retracement(start_price, end_price):
    fib_levels = {
        '0.0': start_price,
        '0.705': start_price + 0.705 * (end_price - start_price),
        '0.9': start_price + 0.9 * (end_price - start_price),
        '1.0': end_price
    }
    return fib_levels