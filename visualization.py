"""
===============================================================================
Alpha Arena Visualization: Cumulative Portfolio Growth vs Equal-Weight Benchmark
===============================================================================
Plots cumulative equity growth over the last 10 years (<= 2520 trading days).
Compares:
1. Final Strategy (Core Equal-Weight 65% + Momentum Tilt 35% with Stateless Hysteresis)
2. Equal-Weight Market Benchmark (engine.equal_weight)

Features:
- Both curves start at initial capital of ₹1,00,000 on Day 1188.
- X-axis: Day (trading days 1188 to 3707, exactly 2,520 days).
- Y-axis: Portfolio value in ₹ with formatted currency commas.
- Shaded region highlighting the 10-year training window.
- Out-of-sample boundary clearly labelled (Secret Test Period: Day 3708+).
- Saved as 'portfolio_growth.png'.
===============================================================================
"""

import os
import tempfile
from pathlib import Path

# Ensure matplotlib writable temp cache before importing matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import engine
import strategy


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "train.csv"
    data = engine.load(data_path)

    # 1. Train agent on its configured TRAIN_WINDOW
    train_start, train_end = engine.train_agent(strategy, data)
    print(f"Agent trained on TRAIN_WINDOW: Days {train_start} to {train_end} ({train_end - train_start + 1} days)")

    # 2. Determine last 10 years evaluation window (<= 2520 trading days)
    total_days = len(data["close"])
    plot_start = max(0, total_days - engine.MAX_TRAIN_DAYS)
    plot_end = total_days - 1
    eval_days_count = plot_end - plot_start + 1

    print(f"Evaluating last 10 years: Days {plot_start} to {plot_end} ({eval_days_count} days)")

    # 3. Run full backtest for strategy and benchmark
    strat_res = engine.backtest(strategy.decide, data, start_day=plot_start, verbose=False)
    bench_res = engine.backtest(engine.equal_weight, data, start_day=plot_start, verbose=False)

    strat_log = strat_res["log"]
    bench_log = bench_res["log"]

    # Filter for the last 10 years
    mask_strat = (strat_log.index > plot_start) & (strat_log.index <= plot_end)
    mask_bench = (bench_log.index > plot_start) & (bench_log.index <= plot_end)

    daily_strat = strat_log.loc[mask_strat, "return"]
    daily_bench = bench_log.loc[mask_bench, "return"]

    # 4. Calculate cumulative equity curves starting from Rs 1,00,000
    equity_strat = engine.STARTING_CAPITAL * (1.0 + daily_strat).cumprod()
    equity_bench = engine.STARTING_CAPITAL * (1.0 + daily_bench).cumprod()

    days_series = [plot_start] + list(daily_strat.index)
    strat_curve = [float(engine.STARTING_CAPITAL)] + list(equity_strat.values)
    bench_curve = [float(engine.STARTING_CAPITAL)] + list(equity_bench.values)

    # 5. Calculate window metrics for display
    strat_metrics = engine.metrics(strat_log.loc[mask_strat])
    bench_metrics = engine.metrics(bench_log.loc[mask_bench])

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=160, facecolor="#ffffff")
    ax.set_facecolor("#fafbfc")

    # Plot benchmark curve
    bench_final = bench_curve[-1]
    ax.plot(
        days_series,
        bench_curve,
        label=f"Equal-Weight Benchmark (Sharpe: {bench_metrics['sharpe']:.2f}, MaxDD: {bench_metrics['max_drawdown_%']:.1f}%)",
        color="#64748b",
        linewidth=1.75,
        linestyle="--",
        alpha=0.9,
    )

    # Plot strategy curve
    strat_final = strat_curve[-1]
    ax.plot(
        days_series,
        strat_curve,
        label=f"Submitted Strategy: {strategy.decide.__name__} (Sharpe: {strat_metrics['sharpe']:.2f}, MaxDD: {strat_metrics['max_drawdown_%']:.1f}%)",
        color="#1d4ed8",
        linewidth=2.2,
    )

    # Shaded training window
    shade_start = max(plot_start, train_start)
    shade_end = min(plot_end, train_end)
    ax.axvspan(
        shade_start,
        shade_end,
        color="#3b82f6",
        alpha=0.08,
        label=f"Training Window (In-Sample: Days {train_start}–{train_end})",
    )

    # If there is out-of-sample data within the last 10 years, shade and label it
    if train_end < plot_end:
        ax.axvspan(
            train_end,
            plot_end,
            color="#f59e0b",
            alpha=0.12,
            label=f"Out-of-Sample Historical Validation (Days {train_end + 1}–{plot_end})",
        )

    # Mark secret evaluation boundary post Day 3707
    ax.axvline(plot_end, color="#dc2626", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.annotate(
        "Secret Test Period\n(Day 3708+)",
        xy=(plot_end, strat_final * 0.92),
        xytext=(plot_end - 280, strat_final * 0.75),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        fontsize=9,
        fontweight="semibold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#fca5a5", alpha=0.9),
    )

    # Formatting axes and titles
    ax.set_title(
        f"Cumulative Portfolio Equity: Strategy vs Equal-Weight Benchmark\n"
        f"Last 10 Years ({eval_days_count} Trading Days: Days {plot_start} to {plot_end})",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.set_xlabel("Trading Day", fontsize=11, fontweight="medium", color="#334155", labelpad=8)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11, fontweight="medium", color="#334155", labelpad=8)

    # Currency formatting on Y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"₹{x:,.0f}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Grid and borders
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    # Metrics comparison summary box
    summary_text = (
        f"Strategy Performance Summary:\n"
        f"• Total Return: {strat_metrics['total_return_%']:+.1f}%\n"
        f"• Sharpe Ratio: {strat_metrics['sharpe']:.2f}\n"
        f"• Sortino Ratio: {strat_metrics['sortino']:.2f}\n"
        f"• Calmar Ratio: {strat_metrics['calmar']:.2f}\n"
        f"• Max Drawdown: {strat_metrics['max_drawdown_%']:.1f}%\n"
        f"• Transaction Costs: {strat_metrics['costs_paid_%']:.2f}%\n"
        f"• Avg Gross Exposure: {strat_metrics['avg_exposure_%']:.1f}%"
    )
    ax.text(
        0.025,
        0.58,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.95),
    )

    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95, fontsize=9.5)
    ax.set_xlim(plot_start, plot_end + 15)

    plt.tight_layout()

    # Save PNG deliverable
    output_png = base_dir / "portfolio_growth.png"
    plt.savefig(output_png, dpi=160)
    print(f"Saved visualization to {output_png}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
    except Exception:
        plt.close(fig)


if __name__ == "__main__":
    main()
