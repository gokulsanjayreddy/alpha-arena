"""
===============================================================================
Alpha Arena Visualization: Cumulative Portfolio Growth vs Equal-Weight Benchmark
===============================================================================
Generates exactly one graph comparing:
1. Final Selected Strategy (AlphaQuant-Prime: Multi-Horizon Cross-Sectional Momentum)
2. Equal-Weight Market Benchmark (engine.equal_weight)

Over the selected 10-year TRAIN_WINDOW (days 1008 to 3527, exactly 2,520 trading days).
Both curves start at initial capital of ₹1,00,000 on Day 1008.
Saves the figure as 'portfolio_growth.png' and displays it.
===============================================================================
"""

import os
import tempfile

# Ensure matplotlib writable temp cache before importing matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import engine
import strategy


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "train.csv"
    data = engine.load(data_path)

    # 1. Train agent on the selected TRAIN_WINDOW
    start_day, end_day = engine.train_agent(strategy, data)
    print(f"Visualizing 10-year training window: Day {start_day} to Day {end_day} ({end_day - start_day + 1} days)")

    # 2. Run backtests from start_day
    strat_res = engine.backtest(strategy.decide, data, start_day=start_day, verbose=False)
    bench_res = engine.backtest(engine.equal_weight, data, start_day=start_day, verbose=False)

    # 3. Align daily returns inside the exact 10-year training window
    strat_log = strat_res["log"]
    bench_log = bench_res["log"]

    mask_strat = (strat_log.index > start_day) & (strat_log.index <= end_day)
    mask_bench = (bench_log.index > start_day) & (bench_log.index <= end_day)

    daily_strat = strat_log.loc[mask_strat, "return"]
    daily_bench = bench_log.loc[mask_bench, "return"]

    # 4. Calculate cumulative equity from starting capital of Rs 1,00,000
    equity_strat = engine.STARTING_CAPITAL * (1.0 + daily_strat).cumprod()
    equity_bench = engine.STARTING_CAPITAL * (1.0 + daily_bench).cumprod()

    # Prepend starting capital at Day start_day
    days_series = [start_day] + list(daily_strat.index)
    strat_curve = [float(engine.STARTING_CAPITAL)] + list(equity_strat.values)
    bench_curve = [float(engine.STARTING_CAPITAL)] + list(equity_bench.values)

    # 5. Build exactly one figure and one axes object
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

    # Plot exactly two lines
    strategy_label = f"Strategy: {strategy.TEAM_NAME} ({strategy.decide.__name__})"
    benchmark_label = "Equal-Weight Benchmark (engine.equal_weight)"

    ax.plot(days_series, strat_curve, label=strategy_label, color="#1f77b4", linewidth=2.0)
    ax.plot(days_series, bench_curve, label=benchmark_label, color="#ff7f0e", linewidth=1.8, linestyle="--")

    # Formatting axes and labels
    ax.set_title(
        f"Cumulative Portfolio Growth vs Equal-Weight Benchmark\n"
        f"10-Year Evaluation Interval: Days {start_day} to {end_day} ({len(days_series)} Trading Days)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Day", fontsize=11, labelpad=8)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11, labelpad=8)

    # Rupee currency formatting on Y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"₹{x:,.0f}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Grid and legend
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)

    # Set x-limits to match the exact window
    ax.set_xlim(start_day, end_day)

    plt.tight_layout()

    # Save PNG deliverable
    output_png = base_dir / "portfolio_growth.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved graph to {output_png}")

    # Display graph (non-blocking if headless, interactive if supported)
    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
    except Exception:
        plt.close(fig)


if __name__ == "__main__":
    main()
