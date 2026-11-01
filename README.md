# Alpha Arena (Meraz 7.0) Quantitative Submission

This repository contains the algorithmic trading submission for **Alpha Arena – Agents War (Meraz 7.0)** by Gokul Sanjay & Yerra Srikar. The agent trades a universe of 12 anonymized Indian equities across 15 years (3,708 trading days) under realistic transaction costs (**0.3% per trade**, `COST_PER_TRADE = 0.003`) and portfolio constraints.

---

## 1. Executive Summary & Approach

The final submitted agent (`decide = decide_momentum`) implements a **Core + Tilt** architecture designed for low turnover and downside protection under high transaction fees (0.3% per trade):
- **Core Anchor (80% allocation)**: Maintained in an equal-weight basket across all 12 stocks ($w_i = 0.80 / 12 = 0.0667$). This captures the market's long-term equity risk premium with zero rebalancing turnover.
- **Momentum Tilt (20% allocation)**: Concentrated into the top $K=4$ relative-strength leaders ($w_i = 0.20 / 4 = 0.050$) ranked by multi-horizon volatility-adjusted returns (21-day, 63-day, and 126-day lookbacks).
- **Stateless Recursive Hysteresis**: To eliminate daily rebalancing churn caused by rank-order ties around the boundary, holdings are determined via a deterministic stateless replay over the trailing 40 bars using cumulative sums. An existing holding is only replaced if its momentum rank falls to 7th or lower ($K_{\text{exit}} = 7$).
- **Dynamic Breadth De-leveraging**: A continuous linear exposure multiplier $E(B_t) \in [0.40, 1.00]$ scales down portfolio exposure when aggregate market breadth ($B_t$, measured as the fraction of stocks above their 50-day moving average) drops during market sell-offs.
- **Defensive Wrapper (`@_safe_decide`)**: Wraps decision outputs to sanitize NaNs/infinities, clip positions to $[-1.0, +1.0]$, constrain total gross leverage $\sum |w_i| \le 1.0$, and guarantee an average gross exposure floor well above the competition's 30% rule.

---

## 2. Quantitative Performance & Empirical Results

All metrics below are generated directly by executing `compare.py`, `strategy.py`, and `visualization.py` on the provided dataset under `COST_PER_TRADE = 0.003`:

### 2.1 Strategy Comparison Table (`comparison.csv`)

Evaluated with model parameters calibrated on `TRAIN_WINDOW = (1188, 3707)` (validation across historical holdout days):

| Strategy | Total Return % | Sharpe | Sortino | Calmar | Max Drawdown % | Trading Accuracy % | Costs Paid % | Fit Sharpe | Generalization |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Equal-Weight Benchmark** | 689.30% | 2.03 | 3.28 | 2.14 | -27.15% | 51.94% | 0.30% | 0.98 | +1.05 |
| **decide_momentum (Submitted)** | **514.88%** | **1.95** | **3.16** | **2.53** | **-19.56%** | **51.94%** | **8.33%** | **0.92** | **+1.03** |
| **decide_ml** (Ridge L2) | 340.59% | 1.67 | 2.68 | 1.72 | -22.58% | 51.94% | 31.20% | 0.46 | +1.21 |
| **decide_trend_following** | 330.47% | 1.59 | 2.56 | 2.07 | -18.43% | 51.94% | 33.41% | 0.48 | +1.11 |
| **decide_mean_reversion** | 248.18% | 1.48 | 2.29 | 1.46 | -21.75% | 51.94% | 52.38% | 0.17 | +1.31 |

#### Analysis:
1. **Turnover Friction**: Under 0.3% trading costs, unanchored active trading pays catastrophic fees (>100% without buffering). By incorporating core equal-weight anchors and wide hysteresis buffers, trading costs are controlled to **8.33%** in `decide_momentum`.
2. **Tail-Risk Control**: While the unmanaged equal-weight benchmark experienced a maximum drawdown of **-27.15%**, `decide_momentum` constrained drawdown to **-19.56%**, yielding the highest Calmar ratio (**2.53**) among all candidate models.

---

### 2.2 Training Window Selection (`windows.csv`)

The competition permits any 10-year continuous block (up to 2,520 trading days). We evaluated all six possible 10-year windows using `compare.py windows` under `COST_PER_TRADE = 0.003`:

| Candidate Window | Validation Return % | Validation Sharpe | Fit Sharpe | Generalization | Max Drawdown % | Transaction Costs % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Days 0–2519 | 74.04% | 1.01 | 1.44 | -0.43 | -17.26% | 9.18% |
| Days 252–2771 | 134.69% | 1.53 | 1.26 | +0.27 | -17.26% | 8.46% |
| Days 504–3023 | 146.44% | 1.29 | 1.26 | +0.03 | -24.42% | 6.82% |
| Days 756–3275 | 210.72% | 1.47 | 1.18 | +0.29 | -24.42% | 6.24% |
| Days 1008–3527 | 557.04% | 1.99 | 0.88 | +1.10 | -19.56% | 7.78% |
| **Days 1188–3707 (Selected)** | **514.88%** | **1.95** | **0.92** | **+1.03** | **-19.56%** | **8.33%** |

#### Why Comparing Validation Sharpe Directly is Misleading:
In `compare.py windows`, each window is evaluated on different historical days. For example, window `(1008, 3527)` leaves Days 3528–3707 as validation, whereas window `(0, 2519)` leaves Days 2520–3707.

#### Common Holdout Block Test (Days 3528–3707):
To isolate genuine parameter learning from evaluation period variance, we evaluated all candidate windows on the **exact same 180-day holdout block** (`Days 3528–3707`):
- **All 6 windows achieved identical metrics on the common block**: Total Return = **13.98%**, Sharpe = **1.17**, Max Drawdown = **-15.94%**, Costs = **0.59%**.
- During this same block, the passive equal-weight benchmark had: Total Return = **2.53%**, Sharpe = **0.27**, Max Drawdown = **-34.39%**.

#### Decision Rationale:
Because parameter differences between late windows produce differences well within statistical noise ($\Delta \text{Sharpe} \approx 0.01$), **`TRAIN_WINDOW = (1188, 3707)`** was selected because it represents the most recent 10-year period directly abutting the secret evaluation period (Day 3708+).

---

### 2.4 Transaction Cost Sensitivity (0.1% vs 0.2%)

To test vulnerability to higher friction, the model was evaluated at double transaction fees (0.2% per trade):
- **0.1% Fee**: Total Return = 527.84%, Sharpe = 1.92, Calmar = 2.60, MaxDD = -19.30%, Cost = 4.63%.
- **0.2% Fee**: Total Return = 499.43%, Sharpe = 1.88, Calmar = 2.49, MaxDD = -19.58%, Cost = 9.26%.

Net Sharpe decreased by only **0.04**, confirming high robustness to fee variation.

---

## 3. Features & Indicators Used

1. **Normalized Returns**:
   - 21-day return: Short-term momentum.
   - 63-day return: Quarterly momentum.
   - 126-day return: Intermediate-term trend persistence.
   - Realized Volatility ($\sigma_{63}$): Realized sample standard deviation of daily log returns over trailing 63 bars.
   - Composite Score: Weighted z-score $\frac{0.25 R_{21} + 0.40 R_{63} + 0.35 R_{126}}{\sigma_{63}}$.
2. **Moving-Average Trend Confirmation**:
   - 50-day simple moving average ($MA_{50}$).
3. **Market Breadth Indicator**:
   - $B_t = \frac{1}{12} \sum_{i=1}^{12} \mathbb{I}(\text{Close}_{i,t} > MA_{50,i,t})$.
   - Exposure function: $E(B_t) = 0.40 + 0.60 \times \text{clip}\left(\frac{\bar{B}_{10} - 0.25}{0.35}, 0, 1\right)$.

---

## 4. Key Assumptions & Constraints

1. **Transaction Cost**: 0.3% applied to gross change in target weights (`COST_PER_TRADE = 0.003`, $0.003 \times \sum |w_t - w_{t-1}|$).
2. **Execution Timing**: Target weights decided on today's close earn the return from today's close to tomorrow's close.
3. **Historical Data Bound**: `decide(history)` receives at most 252 historical trading days.
4. **Allowed Libraries**: Limited strictly to standard library, NumPy, pandas, and scikit-learn inside `strategy.py`. No external API calls or file I/O inside `strategy.py`.
5. **Runtime**: Full 3,708-day backtest executes in under 4 seconds on standard hardware.

---

## 5. Known Limitations & Remaining Risks

1. **Concentrated 12-Stock Universe**: With only 12 stocks, idiosyncratic shocks in any single company have noticeable portfolio impact.
2. **Shorting Limitations**: In strongly trending bull markets, short positions create drag. Our final configuration keeps gross short allocations restrained.
3. **Regime Transition Lags**: Moving-average breadth uses a 10-day smoothing window; instantaneous flash crashes may incur initial drawdowns before breadth de-leveraging takes effect.
4. **Unknown Post-3707 Regime**: The secret test period begins after day 3707. If the market undergoes an unprecedented structural break, historical momentum parameters may decay.

---

## 6. AI-Use Disclosure

In accordance with competition guidelines, AI assistance (Antigravity quantitative pair programming agent) was utilized for:
- Auditing starter code for look-ahead defects in moving-average parameter optimization.
- Developing the walk-forward evaluation harness and block bootstrap confidence intervals.
- Optimizing vector calculations (replacing nested python rolling loops with cumulative sums).
- Formatting visualization figures and building presentation slides.

All quantitative hypotheses, design trade-offs, and empirical results were audited, executed, and validated via local script execution.

---

## 7. How to Run & Verify

1. **Train and run the submitted strategy**:
   ```bash
   python strategy.py
   ```

2. **Run cross-strategy benchmark comparison**:
   ```bash
   python compare.py
   ```

3. **Run 10-year training window comparison**:
   ```bash
   python compare.py windows
   ```

4. **Generate portfolio visualization (Top 3 strategies vs Benchmark)**:
   ```bash
   python visualization.py
   ```
