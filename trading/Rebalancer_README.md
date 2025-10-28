# SoftObjectiveRebalancer

This README explains the logic behind the **SoftObjectiveRebalancer** class — a minimal-update, explainable rebalancing engine aligned with the Day‑1 white‑paper model.

---

## 1. Overview

The rebalancer minimizes a soft objective function:

```
U = TrackingError² + tax_weight × TaxCost
```

subject to trading and cash constraints.  
It uses a greedy, explainable approach that balances tax efficiency and risk control in three main steps:

1. **Sell losers first** (harvest losses)  
2. **Sell gainers** if it improves the objective  
3. **Buy underweights** without breaching minimum cash thresholds  

---

## 2. Inputs

- `positions`: list of dicts, each containing `{ticker, target_weight, quantity, cost_basis, price}`
- One position **must** have `ticker="CASH"`, `price=1.0`, and `quantity` equal to the cash dollars.
- `cov_matrix`: NxN covariance matrix for non-cash assets.
- `tax_weight`: weight applied to tax penalty vs. tracking error.
- `ltcg_rate`: long‑term capital gains tax rate (e.g. 0.15).
- `integer_shares`: toggle for integer or fractional trading.
- `min_cash_pct`: minimum cash percent of total portfolio value.

---

## 3. Cash Handling

Cash is treated as a **real position**, not a derived parameter.  
The initial cash percent is computed as:

```
cash_pct = cash / total_portfolio_value
```

Cash participates in weight calculations for target and current allocations, but is excluded from tracking error since it is risk‑free.

---

## 4. Helper Functions

| Function | Description |
|-----------|--------------|
| `portfolio_weights()` | Returns security weights and cash weight as % of total |
| `te2()` | Computes tracking error squared (ex‑cash) |
| `refresh_deltas()` | Computes current vs. target holdings (in dollar terms) |

---

## 5. Step‑by‑Step Greedy Algorithm

### Step 1 — Sell Losers First
Sells overweight positions where `price <= cost_basis`.  
This realizes capital losses and increases cash.

### Step 2 — Sell Gainers (Soft Objective)
For each overweight gainer, estimate ΔTE² per dollar and tax per dollar, then compute:

```
ΔU_per_$ = ΔTE²_per_$ + tax_weight × (ltcg_rate × gain_per_$)
```

If ΔU_per_$ < 0, the sale improves the overall utility and the position is sold down to target.

### Step 3 — Buy Underweights (Respect Min Cash)
Redistributes available cash to underweight securities proportionally.  
Never spends cash below the minimum cash threshold.

---

## 6. Outputs

Returned dictionary includes:

- `initial_tracking_error`, `final_tracking_error`
- `realized_net_gains`, `estimated_tax_cost`
- `cash_start_pct`, `cash_end_pct`
- `total_traded`, `trades`, `post_allocation`

---

## 7. Key Design Choices

- Cash is part of the portfolio but excluded from risk calculations.  
- Soft objective enables flexible tax‑aware decisions.  
- Greedy algorithm provides transparency and explainability.  
- Minimum cash boundary ensures liquidity and stability.

---

## 8. Example Workflow

```python
positions = [
  {'ticker': 'AAPL', 'target_weight': 0.20, 'quantity': 50, 'cost_basis': 130, 'price': 180},
  {'ticker': 'MSFT', 'target_weight': 0.25, 'quantity': 40, 'cost_basis': 250, 'price': 380},
  {'ticker': 'BND',  'target_weight': 0.51, 'quantity': 600, 'cost_basis': 75,  'price': 70},
  {'ticker': 'CASH', 'target_weight': 0.04, 'quantity': 8000, 'cost_basis': 1,  'price': 1}
]

Sigma = np.array([
  [0.04, 0.01, 0.00],
  [0.01, 0.03, 0.00],
  [0.00, 0.00, 0.01]
])

reb = SoftObjectiveRebalancer(Sigma, tax_weight=1.0, ltcg_rate=0.15, min_cash_pct=0.02)
result = reb.rebalance(positions)
```

---

## 9. Summary

The **SoftObjectiveRebalancer** provides an explainable, modular, tax‑aware rebalancing framework that can evolve into a more advanced agentic trading module.  
It demonstrates how soft objectives can balance risk control and tax efficiency while maintaining liquidity discipline.
