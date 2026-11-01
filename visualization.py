"""
===============================================================================
Alpha Arena Visualization: Top 3 Strategies + Benchmark Cumulative Equity
(Evaluated under COST_PER_TRADE = 0.003 / 0.3% per trade)
===============================================================================
Plots cumulative equity growth over the last 10 years (<= 2520 trading days).
Compares the Top 3 algorithmic strategies against the market benchmark:
1. decide_momentum (Submitted Champion: Core 80% + Momentum Tilt 20% with Hysteresis)
2. decide_ml (Supervised Machine Learning: Regularized Ridge L2 with Core Anchor)
3. decide_trend_following (Systematic Trend: Dual MA & ATR Breakout with Core Anchor)
4. Equal-Weight Benchmark (engine.equal_weight: 1/N Buy-and-Hold Baseline)

Features:
- All 4 curves start at initial capital of ₹1,00,000 on Day 1188.
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
    print(f"Engine COST_PER_TRADE: {engine.COST_PER_TRADE:.3f} ({engine.COST_PER_TRADE * 100:.1f}%)")

    # 2. Determine last 10 years evaluation window (<= 2520 trading days)
    total_days = len(data["close"])
    plot_start = max(0, total_days - engine.MAX_TRAIN_DAYS)
    plot_end = total_days - 1
    eval_days_count = plot_end - plot_start + 1

    print(f"Evaluating last 10 years: Days {plot_start} to {plot_end} ({eval_days_count} days)")

    # 3. Define candidate strategies to plot (Top 3 active models + Benchmark under COST_PER_TRADE = 0.003)
    strategies_to_plot = [
        {
            "name": "Benchmark: Equal-Weight",
            "fn": engine.equal_weight,
            "color": "#64748b",
            "linewidth": 1.75,
            "linestyle": "--",
            "alpha": 0.85,
            "zorder": 2,
        },
        {
            "name": "decide_trend_following (Rank 3: Dual MA Trend)",
            "fn": strategy.decide_trend_following,
            "color": "#9333ea",
            "linewidth": 1.75,
            "linestyle": "-",
            "alpha": 0.9,
            "zorder": 3,
        },
        {
            "name": "decide_ml (Rank 2: Ridge L2 Regression)",
            "fn": strategy.decide_ml,
            "color": "#059669",
            "linewidth": 1.85,
            "linestyle": "-",
            "alpha": 0.95,
            "zorder": 4,
        },
        {
            "name": "decide_momentum (Rank 1 / Submitted: Core + Tilt)",
            "fn": strategy.decide_momentum,
            "color": "#1d4ed8",
            "linewidth": 2.5,
            "linestyle": "-",
            "alpha": 1.0,
            "zorder": 5,
        },
    ]

    # 4. Run backtests and build curves
    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=160, facecolor="#ffffff")
    ax.set_facecolor("#fafbfc")

    days_series = None
    results_summary = []
    max_equity_val = 0.0

    for item in strategies_to_plot:
        fn = item["fn"]
        res = engine.backtest(fn, data, start_day=plot_start, verbose=False)
        log = res["log"]

        mask = (log.index > plot_start) & (log.index <= plot_end)
        daily_ret = log.loc[mask, "return"]

        equity = engine.STARTING_CAPITAL * (1.0 + daily_ret).cumprod()
        if days_series is None:
            days_series = [plot_start] + list(daily_ret.index)

        curve = [float(engine.STARTING_CAPITAL)] + list(equity.values)
        metrics = engine.metrics(log.loc[mask])
        max_equity_val = max(max_equity_val, max(curve))

        results_summary.append({
            "name": item["name"],
            "metrics": metrics,
            "final_eq": curve[-1],
        })

        lbl = f"{item['name']} (Sharpe: {metrics['sharpe']:.2f}, MaxDD: {metrics['max_drawdown_%']:.1f}%)"
        ax.plot(
            days_series,
            curve,
            label=lbl,
            color=item["color"],
            linewidth=item["linewidth"],
            linestyle=item["linestyle"],
            alpha=item["alpha"],
            zorder=item["zorder"],
        )

    # 5. Shaded training window
    shade_start = max(plot_start, train_start)
    shade_end = min(plot_end, train_end)
    ax.axvspan(
        shade_start,
        shade_end,
        color="#3b82f6",
        alpha=0.07,
        label=f"Training Window (In-Sample: Days {train_start}–{train_end})",
        zorder=1,
    )

    # If there is out-of-sample data within the last 10 years, shade and label it
    if train_end < plot_end:
        ax.axvspan(
            train_end,
            plot_end,
            color="#f59e0b",
            alpha=0.10,
            label=f"Out-of-Sample Historical Validation (Days {train_end + 1}–{plot_end})",
            zorder=1,
        )

    # Mark secret evaluation boundary post Day 3707
    top_final_val = results_summary[-1]["final_eq"]
    ax.axvline(plot_end, color="#dc2626", linestyle=":", linewidth=1.5, alpha=0.8, zorder=6)
    ax.annotate(
        "Secret Test Period\n(Day 3708+)",
        xy=(plot_end, top_final_val * 0.92),
        xytext=(plot_end - 290, top_final_val * 0.76),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        fontsize=9,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#fca5a5", alpha=0.9),
        zorder=7,
    )

    # 6. Formatting axes and titles
    ax.set_title(
        f"Cumulative Portfolio Equity: Top 3 Strategies vs Equal-Weight Benchmark\n"
        f"Stressed Transaction Cost (0.3% / trade) · Days {plot_start} to {plot_end} ({eval_days_count} Days) · ₹1,00,000 Base",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.set_xlabel("Trading Day", fontsize=11, fontweight="bold", color="#334155", labelpad=8)
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11, fontweight="bold", color="#334155", labelpad=8)

    # Currency formatting on Y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"₹{x:,.0f}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # Grid and borders
    ax.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    # 7. Summary Table / Callout Box
    table_lines = [f"Performance Summary (Cost Rate = {engine.COST_PER_TRADE * 100:.1f}%):"]
    for s in results_summary:
        m = s["metrics"]
        short_name = s["name"].split("(")[0].strip()
        table_lines.append(
            f"• {short_name:<24} | Ret: {m['total_return_%']:>+6.1f}% | Sh: {m['sharpe']:>4.2f} | Cal: {m['calmar']:>4.2f} | DD: {m['max_drawdown_%']:>5.1f}% | Cost: {m['costs_paid_%']:>5.2f}%"
        )
    summary_text = "\n".join(table_lines)

    ax.text(
        0.025,
        0.54,
        summary_text,
        transform=ax.transAxes,
        fontsize=8.2,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.95),
        zorder=6,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.95,
        fontsize=9,
    )
    ax.set_xlim(plot_start, plot_end + 15)

    plt.tight_layout()

    # Save PNG deliverable
    output_png = base_dir / "portfolio_growth.png"
    plt.savefig(output_png, dpi=160)
    print(f"Saved top 3 strategies visualization to {output_png}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
    except Exception:
        plt.close(fig)


if __name__ == "__main__":
    main()
