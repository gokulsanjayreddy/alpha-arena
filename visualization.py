"""
visualization.py - YOU write this file. It is one of your deliverables.

Make a cumulative portfolio growth graph of your strategy:
  * X-axis: Day
  * Y-axis: Portfolio value in Rs, starting from Rs 1,00,000
  * At most 10 years (2520 days) on one graph, e.g. your training window or your validation days
  * Add the equal-weight benchmark (engine.equal_weight) as a second line so we can compare

Useful pieces:
  data = engine.load("data/train.csv")
  engine.train_agent(strategy, data)                    # trains your agent on TRAIN_WINDOW
  result = engine.backtest(strategy.decide, data, verbose=False)
  result["log"]["return"]                               # your daily returns, index = day
  engine.STARTING_CAPITAL                               # Rs 1,00,000

Run it:   python visualization.py
"""
import matplotlib.pyplot as plt

import engine
import strategy

# TODO: your code here
