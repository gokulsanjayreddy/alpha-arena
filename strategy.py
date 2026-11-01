"""
Alpha Arena Competition Strategy: AlphaQuant-Prime
Team: AlphaQuant-Prime

An algorithmic trading agent implementing four distinct quantitative strategies:
- decide_momentum (Final Selection): Core equal-weight + momentum tilt with
  stateless recursive hysteresis replay and smooth market breadth exposure.
- decide_trend_following: Dual moving-average breakout with ATR trailing stops.
- decide_mean_reversion: RSI and Bollinger Z-score pullbacks in secular uptrends.
- decide_ml: Regularized Ridge regression on multi-period technical features.

Rules Compliance:
- Training Window: (1188, 3707), exactly 2,520 trading days (10.0 years).
- Dependencies: Python standard library, NumPy, pandas, scikit-learn.
- No day-number logic, no external I/O, no internet access, no hardcoded prices.
- Minimum gross exposure >= 40% (comfortably above the 30% competition floor).
- Zero crash guarantee via defensive wrapper.
"""

import functools
import math
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")

TEAM_NAME = "AlphaQuant-Prime"

# Selected 10-year training window: Days 1188 to 3707 (exactly 2,520 trading days)
# Captures the most recent market regime up to the edge of the secret evaluation period.
TRAIN_WINDOW = (1188, 3707)

# Global model dictionary populated by train(data)
MODEL = {
    "mom_trend_ma": 100,
    "top_k_enter": 4,
    "top_k_exit": 6,
    "tilt_strength": 0.35,
    "min_exposure": 0.40,
    "ml_model": None,
    "stocks": None,
}


def _safe_decide(func):
    """
    Decorator that guarantees strict competition compliance:
    1. Catches any unexpected exception and falls back to equal-weight.
    2. Sanitizes NaNs, infinities, and missing stock keys.
    3. Clips individual weights to [-1.0, 1.0].
    4. Enforces sum(|w|) <= 1.0.
    5. Enforces minimum exposure floor >= 40% (strictly above the 30% rule).
    """
    @functools.wraps(func)
    def wrapper(history):
        stocks = list(history["close"].columns)
        N = len(stocks)
        try:
            raw_weights = func(history)
            if not isinstance(raw_weights, dict):
                raw_weights = {}
        except Exception:
            # Fallback to equal-weight on any unforeseen error
            return {s: 1.0 / N for s in stocks}

        cleaned = {}
        for s in stocks:
            val = raw_weights.get(s, 0.0)
            try:
                fval = float(val)
                if not math.isfinite(fval):
                    fval = 0.0
            except (ValueError, TypeError):
                fval = 0.0
            cleaned[s] = max(-1.0, min(1.0, fval))

        gross = sum(abs(w) for w in cleaned.values())

        # Cap gross exposure at 1.0
        if gross > 1.0:
            scale = 1.0 / gross
            cleaned = {s: w * scale for s, w in cleaned.items()}
            gross = 1.0

        # Enforce minimum exposure floor of 40% (competition requires >= 30%)
        min_floor = MODEL.get("min_exposure", 0.40)
        if gross < min_floor:
            if gross > 1e-6:
                scale = min_floor / gross
                cleaned = {s: w * scale for s, w in cleaned.items()}
            else:
                cleaned = {s: min_floor / N for s in stocks}

        return cleaned

    return wrapper


def _compute_atr(h, l, c, period=20):
    """Calculate Average True Range across all stocks."""
    if len(c) < 2:
        return np.ones(c.shape[1], dtype=np.float64)
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    lookback = min(period, len(tr))
    return np.maximum(np.mean(tr[-lookback:], axis=0), 1e-4)


def _compute_rsi(c, period=14):
    """Calculate Relative Strength Index across all stocks."""
    if len(c) <= period:
        return np.full(c.shape[1], 50.0, dtype=np.float64)
    deltas = np.diff(c[-period - 1:], axis=0)
    avg_gain = np.mean(np.maximum(deltas, 0.0), axis=0)
    avg_loss = np.mean(np.maximum(-deltas, 0.0), axis=0)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def _fast_score_matrix(c, trend_len=100):
    """
    Computes multi-horizon momentum scores and trend filter in vectorized NumPy.
    Returns: score_matrix [T, N], trend_ok [T, N], vol_matrix [T, N].
    """
    T, N = c.shape
    r21 = np.full((T, N), -999.0)
    r63 = np.full((T, N), -999.0)
    r126 = np.full((T, N), -999.0)

    if T > 21:
        r21[21:] = c[21:] / c[:-21] - 1.0
    if T > 63:
        r63[63:] = c[63:] / c[:-63] - 1.0
    if T > 126:
        r126[126:] = c[126:] / c[:-126] - 1.0

    d_ret = np.zeros((T, N))
    if T > 1:
        d_ret[1:] = c[1:] / c[:-1] - 1.0

    # Rolling 63-day volatility via cumulative sums
    k = 63
    cs1 = np.vstack([np.zeros((1, N)), np.cumsum(d_ret, axis=0)])
    cs2 = np.vstack([np.zeros((1, N)), np.cumsum(d_ret**2, axis=0)])

    vol = np.full((T, N), 0.02)
    if T > k:
        sum1 = cs1[k:] - cs1[:-k]
        sum2 = cs2[k:] - cs2[:-k]
        mean1 = sum1 / k
        var1 = np.maximum(sum2 / k - mean1**2, 1e-8)
        vol[k - 1:] = np.sqrt(var1)

    # Rolling trend moving average via cumulative sums
    ma = np.copy(c)
    if T > trend_len:
        cs_c = np.vstack([np.zeros((1, N)), np.cumsum(c, axis=0)])
        ma[trend_len - 1:] = (cs_c[trend_len:] - cs_c[:-trend_len]) / trend_len

    score = (0.2 * r21 + 0.4 * r63 + 0.4 * r126) / np.maximum(vol, 1e-4)
    trend_ok = c > ma
    score[~trend_ok] = -999.0
    return score, trend_ok, vol


def train(data):
    """
    Called once by the engine with data inside TRAIN_WINDOW.
    Calibrates trend filter length (look-ahead free) and fits Ridge ML model.
    Wrapped in try/except to guarantee zero unhandled runtime crashes.
    """
    try:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        stocks = list(close.columns)
        MODEL["stocks"] = stocks
        c_arr = close.to_numpy()
        h_arr = high.to_numpy()
        l_arr = low.to_numpy()
        v_arr = volume.to_numpy()
        T, N = c_arr.shape

        # 1. Look-ahead free MA length selection
        # Signal uses close[t] > MA[t], return is forward: (close[t+1] - close[t]) / close[t]
        best_sharpe = -np.inf
        best_ma = 100
        for ma_len in [60, 80, 100, 120]:
            if T > ma_len + 50:
                ma_series = pd.DataFrame(c_arr).rolling(ma_len).mean().to_numpy()
                signal = c_arr[ma_len:-1] > ma_series[ma_len:-1]
                fwd_ret = (c_arr[ma_len + 1:] - c_arr[ma_len:-1]) / c_arr[ma_len:-1]
                strat_daily = np.where(signal, fwd_ret, 0.0).mean(axis=1)
                std_val = np.std(strat_daily)
                if std_val > 1e-6:
                    sh = np.mean(strat_daily) / std_val
                    if sh > best_sharpe:
                        best_sharpe = sh
                        best_ma = ma_len
        MODEL["mom_trend_ma"] = best_ma

        # 2. Supervised Ridge Regression Model Training
        min_lb = 130
        horizon = 5
        X_list, y_list = [], []
        for t in range(min_lb, T - horizon):
            c_sub = c_arr[:t + 1]
            h_sub = h_arr[:t + 1]
            l_sub = l_arr[:t + 1]
            v_sub = v_arr[:t + 1]

            r5 = c_sub[-1] / c_sub[-5] - 1.0
            r10 = c_sub[-1] / c_sub[-10] - 1.0
            r21 = c_sub[-1] / c_sub[-21] - 1.0
            r63 = c_sub[-1] / c_sub[-63] - 1.0
            r126 = c_sub[-1] / c_sub[-126] - 1.0

            d_ret = c_sub[-21:] / c_sub[-22:-1] - 1.0
            vol21 = np.std(d_ret, axis=0)

            ma20 = np.mean(c_sub[-20:], axis=0)
            ma60 = np.mean(c_sub[-60:], axis=0)
            ma120 = np.mean(c_sub[-120:], axis=0)
            ratio20 = c_sub[-1] / ma20 - 1.0
            ratio60 = c_sub[-1] / ma60 - 1.0
            ratio120 = c_sub[-1] / ma120 - 1.0

            hl_range = (h_sub[-1] - l_sub[-1]) / c_sub[-1]
            v_ma20 = np.mean(v_sub[-20:], axis=0)
            v_ratio = v_sub[-1] / (v_ma20 + 1e-4) - 1.0

            target = c_arr[t + horizon] / c_arr[t] - 1.0
            feats = np.column_stack([
                r5, r10, r21, r63, r126, vol21, ratio20, ratio60, ratio120, hl_range, v_ratio
            ])
            X_list.append(feats)
            y_list.append(target)

        if len(X_list) > 0:
            X_all = np.nan_to_num(np.vstack(X_list), nan=0.0, posinf=0.0, neginf=0.0)
            y_all = np.nan_to_num(np.concatenate(y_list), nan=0.0, posinf=0.0, neginf=0.0)
            ridge = Ridge(alpha=150.0)
            ridge.fit(X_all, y_all)
            MODEL["ml_model"] = ridge

    except Exception:
        # Fallback to safe defaults if any unexpected error occurs
        MODEL["mom_trend_ma"] = 100
        MODEL["ml_model"] = None


# =============================================================================
# STRATEGY A: CORE EQUAL-WEIGHT + MOMENTUM TILT (SUBMITTED STRATEGY)
# =============================================================================
def _raw_decide_momentum(history):
    """
    Core Equal-Weight + Momentum Tilt with Stateless Recursive Hysteresis Replay.
    - Base: (1 - tilt) allocated equally across all 12 stocks (broad diversification).
    - Tilt: tilt allocated to top K momentum stocks (cross-sectional alpha).
    - Hysteresis: Stateless replay over trailing 40 days to converge to true holdings.
    - Exposure: Smooth continuous ramp based on trailing 10-day market breadth.
    """
    close = history["close"]
    stocks = list(close.columns)
    c = close.to_numpy()
    T, N = c.shape

    if T < 140:
        return {s: 1.0 / N for s in stocks}

    trend_len = MODEL.get("mom_trend_ma", 100)
    top_enter = MODEL.get("top_k_enter", 4)
    top_exit = MODEL.get("top_k_exit", 6)
    tilt = MODEL.get("tilt_strength", 0.35)
    min_exp = MODEL.get("min_exposure", 0.40)

    score_mat, trend_ok, vol_mat = _fast_score_matrix(c, trend_len=trend_len)

    # Stateless recursive hysteresis replay over trailing 40 days
    replay_len = min(40, T - 130)
    start_t = T - replay_len

    held = list(np.argsort(-score_mat[start_t])[:top_enter])
    for t in range(start_t, T):
        sc_t = score_mat[t]
        ord_t = np.argsort(-sc_t)
        new_held = []
        for idx in held:
            rank = np.where(ord_t == idx)[0][0]
            if rank < top_exit and sc_t[idx] > -500.0:
                new_held.append(idx)
        for idx in ord_t:
            if len(new_held) >= top_enter:
                break
            if idx not in new_held and sc_t[idx] > -500.0:
                new_held.append(idx)
        held = new_held

    # Smooth market breadth exposure scaling (continuous linear ramp)
    breadth_10 = np.mean(trend_ok[-10:], axis=1)
    avg_breadth = np.mean(breadth_10)

    if avg_breadth <= 0.20:
        target_gross = min_exp
    elif avg_breadth >= 0.60:
        target_gross = 1.00
    else:
        target_gross = min_exp + (1.00 - min_exp) * ((avg_breadth - 0.20) / (0.60 - 0.20))

    # Portfolio construction: core + momentum tilt
    w_core = np.full(N, (1.0 - tilt) / N)
    w_tilt = np.zeros(N)

    if len(held) > 0:
        w_each = tilt / len(held)
        for idx in held:
            w_tilt[idx] = w_each
    else:
        w_core = np.full(N, 1.0 / N)

    total_w = (w_core + w_tilt) * target_gross
    return {stocks[i]: float(total_w[i]) for i in range(N)}


# =============================================================================
# STRATEGY B: MEAN REVERSION WITH VOLATILITY AND RSI FILTERS
# =============================================================================
def _raw_decide_mean_reversion(history):
    """
    Statistical Mean Reversion: Dips within secular uptrends.
    Buys stocks above 100-day MA experiencing short-term oversold deviations (Z < -0.5).
    """
    close = history["close"]
    stocks = list(close.columns)
    c = close.to_numpy()
    T, N = c.shape

    if T < 105:
        return {s: 1.0 / N for s in stocks}

    ma100 = np.mean(c[-100:], axis=0)
    in_uptrend = c[-1] > ma100

    ma20 = np.mean(c[-20:], axis=0)
    std20 = np.maximum(np.std(c[-20:], axis=0), 1e-4)
    z = (c[-1] - ma20) / std20
    rsi = _compute_rsi(c, period=14)

    score = np.full(N, -999.0)
    for i in range(N):
        if in_uptrend[i] and z[i] < -0.5:
            score[i] = -z[i] * 2.0 + (50.0 - rsi[i]) * 0.1
        elif in_uptrend[i] and z[i] < 0.5:
            score[i] = 0.5 - z[i]

    order = np.argsort(-score)
    picks = [i for i in order[:4] if score[i] > -500.0]

    breadth = np.mean(in_uptrend)
    target_gross = 0.40 if breadth < 0.25 else (0.65 if breadth < 0.50 else 0.90)

    if len(picks) == 0:
        low_z = np.argsort(z)[:2]
        half = target_gross / 2.0
        return {stocks[low_z[0]]: half, stocks[low_z[1]]: half}

    w_each = target_gross / len(picks)
    return {stocks[idx]: float(w_each) for idx in picks}


# =============================================================================
# STRATEGY C: VOLATILITY-ADJUSTED TREND FOLLOWING
# =============================================================================
def _raw_decide_trend_following(history):
    """
    Systematic Trend Following with 2.5x ATR Trailing Stop.
    Sizes positions inversely proportional to percentage ATR risk.
    """
    close = history["close"]
    high = history["high"]
    low = history["low"]
    stocks = list(close.columns)
    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    T, N = c.shape

    if T < 105:
        return {s: 1.0 / N for s in stocks}

    atr = _compute_atr(h, l, c, 20)
    ma50 = np.mean(c[-50:], axis=0)
    ma120 = np.mean(c[-120:], axis=0) if T >= 120 else ma50

    trend_up = (c[-1] > ma50) & (ma50 > ma120)
    peak_40 = np.max(h[-40:], axis=0)
    drop_atr = (peak_40 - c[-1]) / atr
    valid = trend_up & (drop_atr < 2.5)

    breadth = np.mean(c[-1] > ma50)
    target_gross = 0.40 if breadth < 0.25 else (0.65 if breadth < 0.50 else 1.00)

    score = (c[-1] / ma50 - 1.0) / (atr / c[-1] + 1e-4)
    score[~valid] = -999.0

    order = np.argsort(-score)
    picks = [i for i in order[:4] if score[i] > -500.0]

    if len(picks) == 0:
        low_risk = np.argsort(atr / c[-1])[:2]
        half = target_gross / 2.0
        return {stocks[low_risk[0]]: half, stocks[low_risk[1]]: half}

    w_each = target_gross / len(picks)
    return {stocks[idx]: float(w_each) for idx in picks}


# =============================================================================
# STRATEGY D: MACHINE LEARNING RETURN PREDICTION
# =============================================================================
def _raw_decide_ml(history):
    """
    Supervised Machine Learning using Regularized Ridge Regression.
    Predicts forward 5-day returns across 11 normalized technical features.
    """
    close = history["close"]
    high = history["high"]
    low = history["low"]
    volume = history["volume"]
    stocks = list(close.columns)

    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    v = volume.to_numpy()
    T, N = c.shape

    model = MODEL.get("ml_model")
    if T < 130 or model is None:
        return {s: 1.0 / N for s in stocks}

    r5 = c[-1] / c[-5] - 1.0
    r10 = c[-1] / c[-10] - 1.0
    r21 = c[-1] / c[-21] - 1.0
    r63 = c[-1] / c[-63] - 1.0
    r126 = c[-1] / c[-126] - 1.0

    d_ret = c[-21:] / c[-22:-1] - 1.0
    vol21 = np.std(d_ret, axis=0)

    ma20 = np.mean(c[-20:], axis=0)
    ma60 = np.mean(c[-60:], axis=0)
    ma120 = np.mean(c[-120:], axis=0)
    ratio20 = c[-1] / ma20 - 1.0
    ratio60 = c[-1] / ma60 - 1.0
    ratio120 = c[-1] / ma120 - 1.0

    hl_range = (h[-1] - l[-1]) / c[-1]
    v_ma20 = np.mean(v[-20:], axis=0)
    v_ratio = v[-1] / (v_ma20 + 1e-4) - 1.0

    feats = np.column_stack([
        r5, r10, r21, r63, r126, vol21, ratio20, ratio60, ratio120, hl_range, v_ratio
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    pred = model.predict(feats)
    ma100 = np.mean(c[-100:], axis=0)
    trend_ok = c[-1] > ma100
    pred[~trend_ok] = -999.0

    breadth = np.mean(trend_ok)
    target_gross = 0.40 if breadth < 0.25 else (0.65 if breadth < 0.50 else 0.95)

    order = np.argsort(-pred)
    picks = [i for i in order[:4] if pred[i] > -500.0]

    if len(picks) == 0:
        low_vol = np.argsort(vol21)[:2]
        half = target_gross / 2.0
        return {stocks[low_vol[0]]: half, stocks[low_vol[1]]: half}

    w_each = target_gross / len(picks)
    return {stocks[idx]: float(w_each) for idx in picks}


# =============================================================================
# EXPORTED STRATEGY FUNCTIONS WITH DEFENSIVE WRAPPERS
# =============================================================================
decide_momentum = _safe_decide(_raw_decide_momentum)
decide_mean_reversion = _safe_decide(_raw_decide_mean_reversion)
decide_trend_following = _safe_decide(_raw_decide_trend_following)
decide_ml = _safe_decide(_raw_decide_ml)

# Champion strategy submission
decide = decide_momentum


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    data_path = Path(__file__).resolve().parent / "data" / "train.csv"
    data = engine.load(data_path)

    print("=" * 60)
    print(f"Alpha Arena Evaluation: {TEAM_NAME}")
    print(f"Active Submission: {decide.__name__}")
    print("=" * 60)

    window = engine.train_agent(sys.modules[__name__], data)
    backtest_result = engine.backtest(decide, data, verbose=False)
    log = backtest_result["log"]

    inside = (log.index >= window[0]) & (log.index <= window[1])
    in_metrics = engine.metrics(log[inside])
    out_metrics = engine.metrics(log[~inside])

    print(f"\nTrained on days {window[0]} - {window[1]} ({window[1] - window[0] + 1} days)")
    print("\n[IN-SAMPLE] Inside Training Window:")
    engine.print_metrics(in_metrics)

    print("\n[OUT-OF-SAMPLE] Outside Training Window (Validation):")
    engine.print_metrics(out_metrics)

    gen = out_metrics["sharpe"] - in_metrics["sharpe"]
    print(f"Generalization (Val Sharpe - Fit Sharpe): {gen:+.2f}")
    print("=" * 60)
