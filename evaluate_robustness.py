"""
Robustness and Walk-Forward Evaluation Harness for Alpha Arena.
Tests strategies and training windows across:
1. Walk-Forward Chronological Folds (5 folds across all 15 years).
2. Common Validation Block (Days 3528 to 3707) for unbiased window comparison.
3. Cost Sensitivity Analysis (0.1% vs 0.2% trade fee).
4. Bootstrap Confidence Intervals on Sharpe difference vs Equal-Weight Benchmark.
"""

import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd

import engine
import strategy

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "train.csv"


def run_backtest_with_cost(decide_fn, data, start_day=50, cost_rate=0.001):
    """
    Run backtest with configurable transaction cost rate (e.g., 0.001 or 0.002).
    """
    close = data["close"]
    stocks = list(close.columns)
    days = list(close.index)
    start = days.index(start_day)
    fwd_ret = close.pct_change().shift(-1).fillna(0.0).to_numpy()

    w_prev = np.zeros(len(stocks))
    log, weights = [], []
    for i in range(start, len(days) - 1):
        lo = max(0, i + 1 - engine.LOOKBACK_DAYS)
        history = {f: data[f].iloc[lo: i + 1] for f in engine.FIELDS}
        try:
            w = engine.clean_weights(decide_fn(history), stocks)
        except Exception:
            w = w_prev
        cost = cost_rate * np.abs(w - w_prev).sum()
        held = w != 0
        hits = int((w[held] * fwd_ret[i][held] > 0).sum())
        log.append((float(w @ fwd_ret[i] - cost), cost, int(held.sum()), hits, float(np.abs(w).sum()), 0))
        weights.append(w)
        w_prev = w

    log_df = pd.DataFrame(
        log,
        index=pd.Index(days[start + 1:], name="day"),
        columns=["return", "cost", "positions", "hits", "exposure", "error"]
    )
    return log_df


def bootstrap_sharpe_diff(r_strat, r_bench, n_boot=1000, ci=95):
    """
    Bootstrap 95% confidence interval for (Sharpe_strat - Sharpe_bench).
    Uses block bootstrap (block size = 10 days) to preserve return autocorrelation.
    """
    T = len(r_strat)
    if T < 20:
        return 0.0, (0.0, 0.0)
    
    diffs = []
    block_size = 10
    n_blocks = int(np.ceil(T / block_size))
    
    # Pre-split into blocks
    blocks_strat = [r_strat.iloc[i:i + block_size].values for i in range(0, T, block_size)]
    blocks_bench = [r_bench.iloc[i:i + block_size].values for i in range(0, T, block_size)]
    num_blocks = len(blocks_strat)
    
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, num_blocks, size=n_blocks)
        s_sample = np.concatenate([blocks_strat[b] for b in idx])[:T]
        b_sample = np.concatenate([blocks_bench[b] for b in idx])[:T]
        
        std_s = np.std(s_sample)
        std_b = np.std(b_sample)
        sh_s = np.mean(s_sample) / std_s * np.sqrt(252) if std_s > 1e-6 else 0.0
        sh_b = np.mean(b_sample) / std_b * np.sqrt(252) if std_b > 1e-6 else 0.0
        diffs.append(sh_s - sh_b)
        
    diffs = np.array(diffs)
    alpha = (100 - ci) / 2
    lower = float(np.percentile(diffs, alpha))
    upper = float(np.percentile(diffs, 100 - alpha))
    mean_diff = float(np.mean(diffs))
    return mean_diff, (lower, upper)


def run_walk_forward(strategy_module, decide_fn, data, cost_rate=0.001):
    """
    5-fold walk-forward validation across the full 3,708 days.
    Folds:
      Fold 1: Train 0-707,       Test 708-1307  (~600 days)
      Fold 2: Train 0-1307,      Test 1308-1907 (~600 days)
      Fold 3: Train 0-1907,      Test 1908-2507 (~600 days)
      Fold 4: Train 508-2507,    Test 2508-3107 (~600 days, 2000d train)
      Fold 5: Train 1108-3107,   Test 3108-3707 (~600 days, 2000d train)
    """
    folds = [
        {"name": "Fold 1", "train": (0, 707),     "test": (708, 1307)},
        {"name": "Fold 2", "train": (0, 1307),    "test": (1308, 1907)},
        {"name": "Fold 3", "train": (0, 1907),    "test": (1908, 2507)},
        {"name": "Fold 4", "train": (508, 2507),  "test": (2508, 3107)},
        {"name": "Fold 5", "train": (1108, 3107), "test": (3108, 3707)},
    ]
    
    rows = []
    # Also evaluate equal weight benchmark
    bench_log = run_backtest_with_cost(engine.equal_weight, data, start_day=50, cost_rate=cost_rate)
    
    for f in folds:
        t_start, t_end = f["train"]
        v_start, v_end = f["test"]
        
        # Train agent on fold's training slice
        engine.train_agent(strategy_module, data, window=(t_start, t_end))
        
        # Run test from v_start (or earlier warmup) to v_end
        warmup_start = max(0, v_start - 252)
        strat_log = run_backtest_with_cost(decide_fn, data, start_day=warmup_start, cost_rate=cost_rate)
        
        # Slice exact test period
        test_mask = (strat_log.index >= v_start) & (strat_log.index <= v_end)
        b_mask = (bench_log.index >= v_start) & (bench_log.index <= v_end)
        
        m = engine.metrics(strat_log[test_mask])
        m_b = engine.metrics(bench_log[b_mask])
        
        r_strat = strat_log.loc[test_mask, "return"]
        r_bench = bench_log.loc[b_mask, "return"]
        
        d_sharpe, (ci_lo, ci_hi) = bootstrap_sharpe_diff(r_strat, r_bench)
        
        rows.append({
            "Fold": f["name"],
            "Train": f"{t_start}-{t_end}",
            "Test": f"{v_start}-{v_end}",
            "Return%": m["total_return_%"],
            "Sharpe": m["sharpe"],
            "BenchSh": m_b["sharpe"],
            "DiffSh": m["sharpe"] - m_b["sharpe"],
            "BootCI_95": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "Sortino": m["sortino"],
            "Calmar": m["calmar"],
            "MaxDD%": m["max_drawdown_%"],
            "Cost%": m["costs_paid_%"],
            "Acc%": m["trading_accuracy_%"],
            "Exp%": m["avg_exposure_%"],
        })
        
    df = pd.DataFrame(rows)
    return df


def run_common_block_eval(strategy_module, data, cost_rate=0.001):
    """
    Evaluates all 6 candidate 10-year windows on a fixed COMMON validation block:
    Days 3528 to 3707 (final 180 trading days).
    Note: Windows 0 through 4 end on or before day 3527, so this block is strictly
    OUT-OF-SAMPLE for Windows 0-4.
    For Window 5 (1188-3707), days 3528-3707 were inside its 10-year training window.
    """
    common_start, common_end = 3528, 3707
    days = data["close"].index
    last_start = days[-1] - engine.MAX_TRAIN_DAYS + 1
    starts = sorted(set(range(days[0], last_start, 252)) | {last_start})
    windows = [(s, s + engine.MAX_TRAIN_DAYS - 1) for s in starts]
    
    rows = []
    # Benchmark on common block
    bench_log = run_backtest_with_cost(engine.equal_weight, data, start_day=common_start - 252, cost_rate=cost_rate)
    b_mask = (bench_log.index >= common_start) & (bench_log.index <= common_end)
    m_b = engine.metrics(bench_log[b_mask])
    
    for w in windows:
        engine.train_agent(strategy_module, data, window=w)
        strat_log = run_backtest_with_cost(strategy_module.decide, data, start_day=common_start - 252, cost_rate=cost_rate)
        mask = (strat_log.index >= common_start) & (strat_log.index <= common_end)
        m = engine.metrics(strat_log[mask])
        
        is_in_sample = w[1] >= common_start
        rows.append({
            "Window": f"days {w[0]}-{w[1]}",
            "Is OOS on Block?": "NO (in-sample)" if is_in_sample else "YES (out-of-sample)",
            "Block Return%": m["total_return_%"],
            "Block Sharpe": m["sharpe"],
            "Block Calmar": m["calmar"],
            "Block MaxDD%": m["max_drawdown_%"],
            "Block Cost%": m["costs_paid_%"],
            "Block Acc%": m["trading_accuracy_%"],
            "Block Exp%": m["avg_exposure_%"],
        })
        
    df = pd.DataFrame(rows)
    # Restore agent's original window
    engine.train_agent(strategy_module, data)
    return df, m_b


def run_cost_sensitivity(strategy_module, decide_fn, data):
    """
    Compares metrics at 0.1% vs 0.2% cost across full dataset and validation.
    """
    engine.train_agent(strategy_module, data)
    log_01 = run_backtest_with_cost(decide_fn, data, start_day=50, cost_rate=0.001)
    log_02 = run_backtest_with_cost(decide_fn, data, start_day=50, cost_rate=0.002)
    
    w = strategy_module.TRAIN_WINDOW
    inside = (log_01.index >= w[0]) & (log_01.index <= w[1])
    
    m_val_01 = engine.metrics(log_01[~inside])
    m_val_02 = engine.metrics(log_02[~inside])
    
    df = pd.DataFrame({
        "Metric": ["Total Return %", "Sharpe Ratio", "Calmar Ratio", "Max Drawdown %", "Costs Paid %", "Trading Accuracy %"],
        "Cost Rate 0.1% (Standard)": [
            f"{m_val_01['total_return_%']:.2f}%",
            f"{m_val_01['sharpe']:.2f}",
            f"{m_val_01['calmar']:.2f}",
            f"{m_val_01['max_drawdown_%']:.2f}%",
            f"{m_val_01['costs_paid_%']:.2f}%",
            f"{m_val_01['trading_accuracy_%']:.2f}%"
        ],
        "Cost Rate 0.2% (2x Stressed)": [
            f"{m_val_02['total_return_%']:.2f}%",
            f"{m_val_02['sharpe']:.2f}",
            f"{m_val_02['calmar']:.2f}",
            f"{m_val_02['max_drawdown_%']:.2f}%",
            f"{m_val_02['costs_paid_%']:.2f}%",
            f"{m_val_02['trading_accuracy_%']:.2f}%"
        ],
    })
    return df


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 15)
    data = engine.load(DATA_PATH)
    
    print("=" * 80)
    print("PHASE 1: WALK-FORWARD & COMMON-BLOCK ROBUSTNESS EVALUATION HARNESS")
    print("=" * 80)
    
    print("\n--- 1. 5-Fold Walk-Forward Performance (Cost = 0.1%) ---")
    wf_df = run_walk_forward(strategy, strategy.decide, data, cost_rate=0.001)
    print(wf_df.to_string(index=False))
    
    print("\n--- Walk-Forward Summary Statistics ---")
    numeric_cols = ["Return%", "Sharpe", "BenchSh", "DiffSh", "Sortino", "Calmar", "MaxDD%", "Cost%", "Acc%", "Exp%"]
    med = wf_df[numeric_cols].median()
    worst = wf_df[numeric_cols].min()
    # For MaxDD and Cost, worst is min (most negative DD) or max (highest cost)
    worst["MaxDD%"] = wf_df["MaxDD%"].min()
    worst["Cost%"] = wf_df["Cost%"].max()
    
    summary_df = pd.DataFrame({"Median": med, "Worst Fold": worst}).T
    print(summary_df.round(2).to_string())
    
    print("\n--- 2. Common Block (Days 3528-3707) Window Evaluation ---")
    common_df, bench_common = run_common_block_eval(strategy, data, cost_rate=0.001)
    print(common_df.to_string(index=False))
    print(f"\nEqual-Weight Benchmark on Common Block: Return={bench_common['total_return_%']:.2f}%, "
          f"Sharpe={bench_common['sharpe']:.2f}, MaxDD={bench_common['max_drawdown_%']:.2f}%")
    
    print("\n--- 3. Cost Sensitivity Analysis (0.1% vs 0.2%) ---")
    sens_df = run_cost_sensitivity(strategy, strategy.decide, data)
    print(sens_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

