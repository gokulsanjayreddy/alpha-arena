# Agents War — Alpha Arena (Meraz 7.0)

You get 15 years of daily prices for 12 Indian stocks. Build a trading agent that decides,
every day, how to split your Rs 1 lakh (virtual) between those stocks. After the event we let
your agent trade a **secret market period it has never seen**, right after the training data ends.
We don't reveal which years the data or the secret period come from, so your agent has to cope
with whatever the market does. The best agent wins.

**Getting one agent to run takes 10 minutes. Winning takes thinking:** try several ideas,
test them honestly, understand why they work because you also have to present your idea and why it is best, and control your risk.

## What's in this kit
| File | What it is |
|---|---|
| `data/train.csv` | Prices: `day, stock, open, high, low, close, volume`. Stock names are hidden (STOCK_01..12) and every stock starts at 100. |
| `strategy.py` | **Your agent** (we run this one). `python strategy.py` trains it and prints its performance metrics. |
| `compare.py` | `python compare.py` compares all your `decide_xxx` ideas; `python compare.py windows` compares 10-year training windows. Saves tables for your slides. |
| `visualization.py` | **You write this:** your portfolio growth graph (see What to submit). |
| `starter.ipynb` | The same steps as a notebook (works in Google Colab / Jupyter). |
| `engine.py` | The simulator. Don't edit it; we score you with the same file. |

Setup, either way works:
* **Google Colab (nothing to install):** File → Open notebook → GitHub tab → paste this repo's link → open `starter.ipynb`,
  then run the cells. The first cell downloads the rest of the kit.
* **Laptop:** `git clone <this repo>` (or Code → Download ZIP), then `pip install pandas numpy matplotlib scikit-learn`.

## What you write (in `strategy.py`)
```python
TRAIN_WINDOW = (756, 3275)           # the block of days your agent learns from: ANY block, max 10 years (2520 days)

def train(data):                     # called ONCE, with only the days inside TRAIN_WINDOW
    ...                              # learn parameters / fit models, store them in MODEL

def decide_my_idea(history):         # called every day; history = only the LAST YEAR (252 days) up to today
    close = history["close"]         # DataFrame: rows = days, columns = stocks
    ...
    return {"STOCK_04": 0.5, "STOCK_09": -0.2}   # 50% long STOCK_04, 20% short STOCK_09, 30% cash

decide = decide_my_idea              # the one we run on the secret period
```
* **The 10-year rule:** your agent may learn from at most 10 years of the 15. You pick which 10:
  days 0–2519, 252–2771, 756–3275, 1188–3707 ... any block of up to 2520 days. Every day outside your
  window is your validation data. A `TRAIN_WINDOW` longer than 2520 days is rejected automatically.
* `decide` only ever sees the last year, so all long-term learning has to happen in `train`.
* `decide` never sees the future. Weights go from -1 to 1, and the total of |weights| must be ≤ 1 (if it's more, it gets scaled down).
* Changing a weight costs **0.1%** of the amount traded, so trading every day gets expensive.

## How to work
1. Write 3 or more different ideas as `decide_xxx` functions.
2. `python compare.py` trains on your `TRAIN_WINDOW` and compares all your ideas on the days **outside** it (validation).
3. `python compare.py windows` trains your final agent on **6 different 10-year windows** and shows how each one does
   outside its window. A good team tries several windows and can explain why it picked one.
   Careful: each window is checked on different days, so think about what the numbers really tell you.
4. Pick your final agent (`decide = decide_xxx`) and `TRAIN_WINDOW`. Be wary of an agent that is great inside its window but weak outside it.

## Rules
1. `TRAIN_WINDOW` may cover at most **10 years (2520 days)**, otherwise the submission is disqualified.
   Don't collect prices day by day inside `decide` to get around the 1-year limit.
2. Submit `strategy.py` using only python, numpy, pandas and scikit-learn. No internet or file reading inside it.
3. No hard-coded prices or day numbers: your agent must not act differently just because of which day number it is.
4. On average your agent must keep **at least 30%** of the money invested. An agent that sits in cash is disqualified.
5. We run the agent over the training days plus the secret period. That must finish within **3 minutes**.
6. If the agent crashes on more than **5%** of days, it is disqualified. On any day it crashes, it keeps yesterday's weights.
7. No sharing code between teams. AI assistants are allowed; say so in your explanation.

## What to submit
1. **`strategy.py`**: your agent (`TRAIN_WINDOW`, `train`, all the `decide_xxx` ideas you tried, and `decide` pointing
   at your final one). Running `python strategy.py` must print its performance metrics.
2. **`visualization.py`**: a cumulative portfolio growth graph of your strategy.
   * X-axis: Day (the data has no dates)
   * Y-axis: Portfolio value in ₹, starting from ₹1,00,000
   * At most 10 years on one graph, with the equal-weight benchmark as a second line
3. **`README.md`** (your own): your approach, the training window (data) you selected and why, the features
   you used, and your assumptions.
4. **PPT**, 3–5 slides:
   1. Your final agent in one sentence.
   2. The strategies you tried (your `compare.py` table).
   3. Which training window you picked and why, with the evidence from validation (`windows.csv`).
   4. Risk: how your agent limits losses when the market crashes.
   5. What you learnt / what you would do with more time.

Working in the notebook / Colab? Its last cell downloads `strategy.py` and `visualization.py` for you.
Submit the `.py` files, not the notebook.

## Scoring — 100 backtest marks (measured on the secret period)
| Metric | Marks | What it means |
|---|---|---|
| Net Profit / Return | 20 | total return |
| Sharpe Ratio | 15 | return per unit of risk (ups and downs) |
| Calmar Ratio | 15 | yearly return ÷ worst fall from a peak |
| Maximum Drawdown | 10 | worst fall from a peak (smaller is better) |
| Generalization | 15 | secret-period Sharpe minus the Sharpe inside your training window (a small drop is better) |
| Trading Accuracy | 10 | % of held positions that moved the right way the next day |
| Transaction Cost Efficiency | 10 | total trading costs you paid (less is better) |
| Technical Reliability | 5 | full marks if the agent never crashes |

Every metric except reliability is **ranked**: your percentile among all teams, so the best team on a
metric gets full marks. The tie-break is a higher Sharpe ratio. If presentations are held, the final score
is 70% backtest + 30% slides. `python strategy.py` and `compare.py` print every one of these numbers
(plus the Sortino ratio, which is like Sharpe but only counts the falls).
