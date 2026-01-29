```python
import random

def simulate_market(days=200, start_price=10000):
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + random.uniform(-0.02, 0.02)))
    return prices

def moving_average(data, window):
    return sum(data[-window:]) / window if len(data) >= window else None

def run_strategy(prices, capital=100000):
    cash = capital
    shares = 0

    for day in range(20, len(prices)):
        short = moving_average(prices[:day], 5)
        long = moving_average(prices[:day], 20)
        price = prices[day]

        if short and long:
            if short > long and cash > 0:
                shares = cash / price
                cash = 0
            elif short < long and shares > 0:
                cash = shares * price
                shares = 0

    return cash + shares * prices[-1]

if __name__ == "__main__":
    prices = simulate_market()
    final_capital = run_strategy(prices)
    print("Final capital:", round(final_capital, 2))
