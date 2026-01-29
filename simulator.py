"""
Quant Investment Simulator

This script simulates the growth of an investment over time
using randomized monthly market returns.
"""
# Simulates investment growth over a given number of months
# initial_capital: starting investment amount
# months: simulation duration

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

# ---- DATA LOADING ----
def load_market_data():
    """
    Monthly returns based on real Nasdaq-100 historical data.
    Source: Investing.com
    Period: Apr 2025 – Jan 2026
    """
    return [
        0.0904,
        0.0627,
        0.0238,
        0.0085,
        0.0540,
        0.0477,
        -0.0164,
        -0.0073,
        0.0306
    ]


# ---- STRATEGY LOGIC ----
def apply_strategy(returns, initial_capital):
    capital = initial_capital

    for r in returns:
        capital *= (1 + r)

    return capital


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    initial_capital = 100000

    returns = load_market_data()
    final_capital = apply_strategy(returns, initial_capital)

    print("Final capital:", round(final_capital, 2))