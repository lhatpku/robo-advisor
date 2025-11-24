from typing import List, Dict, Any, Tuple
import math
import numpy as np


class SoftObjectiveRebalancer:
    """
    Minimal-update rebalancer based on original flow, now with:
      • CASH as part of positions (ticker="CASH")
      • Tracking Error (TE) excludes CASH from covariance
      • Portfolio % calculations include CASH

    Flow:
      1) Sell losers first (only if overweight)
      2) Sell gainers if the soft objective improves (ΔTE² + tax_weight * tax_per_$ < 0)
      3) Buy underweights proportionally using cash, but never drop below min_cash_pct
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,        # (m x m) covariance for NON-CASH tickers
        tax_weight: float = 1.0,       # relative weight on tax vs. TE²
        ltcg_rate: float = 0.15,       # expected long-term gains tax rate
        integer_shares: bool = False,  # True → integer shares only
        min_cash_pct: float = 0.02     # minimum cash % of total portfolio
    ):
        self.Sigma = np.array(cov_matrix, dtype=float)
        if self.Sigma.shape[0] != self.Sigma.shape[1]:
            raise ValueError("cov_matrix must be square")
        self.tax_weight = float(tax_weight)
        self.ltcg = float(ltcg_rate)
        self.integer_shares = bool(integer_shares)
        self.min_cash_pct = float(min_cash_pct)

    # -------------------- main function --------------------
    def rebalance(self, positions: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        positions: list of dicts with keys:
          {ticker, target_weight, quantity, cost_basis, price}

        CASH row:
          ticker == "CASH"
          price = 1.0
          quantity = current cash amount (dollars)
          target_weight = target % of total (e.g., 0.04)

        Returns summary dictionary
        """

        # --- Split CASH and securities ---
        cash_idx = None
        for i, p in enumerate(positions):
            if str(p["ticker"]).upper() == "CASH":
                cash_idx = i
                break
        if cash_idx is None:
            raise ValueError('Missing position with ticker == "CASH"')

        cash_pos = positions[cash_idx]
        cash = float(cash_pos["quantity"]) * 1.0
        sec_positions = [p for i, p in enumerate(positions) if i != cash_idx]
        tickers = [p["ticker"] for p in sec_positions]
        n = len(tickers)
        if n == 0:
            raise ValueError("No securities found besides CASH")
        if self.Sigma.shape[0] != n:
            raise ValueError("cov_matrix dimension must match number of NON-CASH tickers")

        tgt_w_all = np.array([float(p["target_weight"]) for p in positions], dtype=float)
        qty_sec = np.array([float(p["quantity"]) for p in sec_positions], dtype=float)
        basis = np.array([float(p["cost_basis"]) for p in sec_positions], dtype=float)
        price = np.array([float(p["price"]) for p in sec_positions], dtype=float)

        # --- Compute total and percentages (include cash) ---
        sec_val = float(np.dot(qty_sec, price))
        total_val = sec_val + cash
        cash_start_pct = cash / total_val if total_val > 1e-12 else 0.0

        # separate target weights
        tgt_w_sec = np.array([float(p["target_weight"]) for p in sec_positions], dtype=float)
        tgt_w_cash = float(cash_pos["target_weight"])
        tgt_w_sum = float(np.sum(tgt_w_sec) + tgt_w_cash)
        if abs(tgt_w_sum - 1.0) > 1e-4:
            print(f"⚠️ Warning: target weights sum to {tgt_w_sum:.3f}, not 1.0 (CASH + securities).")

        # --- Helpers ---
        def portfolio_weights(q_sec: np.ndarray, cash_amt: float) -> Tuple[np.ndarray, float]:
            sec_value = float(np.dot(q_sec, price))
            tot_value = sec_value + cash_amt
            if tot_value <= 1e-12:
                return np.zeros_like(q_sec), 0.0
            w_sec = (q_sec * price) / tot_value
            w_cash = cash_amt / tot_value
            return w_sec, w_cash

        def te2(q_sec: np.ndarray) -> float:
            """Tracking error squared, excluding cash"""
            w_sec, _ = portfolio_weights(q_sec, cash)
            diff = w_sec - tgt_w_sec  # cash ignored here
            return float(diff.T @ self.Sigma @ diff)

        def refresh_deltas(q: np.ndarray, cash_amt: float):
            """Recalculate holdings, total, and deltas (in $ terms)"""
            sec_value = float(np.dot(q, price))
            tot_value = sec_value + cash_amt
            curr_w_sec = (q * price) / tot_value
            tgt_val_sec = tgt_w_sec * tot_value
            curr_val_sec = q * price
            holdings_delta = tgt_val_sec - curr_val_sec  # + => need to buy
            return holdings_delta, sec_value, tot_value

        # --- Initial state ---
        initial_te = math.sqrt(max(te2(qty_sec), 0.0))
        realized_gains_total = 0.0
        trades = []

        holdings_delta, sec_val, total_val = refresh_deltas(qty_sec, cash)

        # ---------------- STEP 1: Sell losers (if overweight) ----------------
        for i in range(n):
            if price[i] <= basis[i] and holdings_delta[i] < -1e-12:
                desired_sell_value = min(-holdings_delta[i], qty_sec[i] * price[i])
                shares_to_sell = desired_sell_value / price[i]
                if self.integer_shares:
                    shares_to_sell = math.floor(min(shares_to_sell, qty_sec[i]))
                    if shares_to_sell <= 0:
                        continue
                else:
                    shares_to_sell = min(shares_to_sell, qty_sec[i])

                proceeds = shares_to_sell * price[i]
                rg = shares_to_sell * (price[i] - basis[i])  # likely ≤ 0
                qty_sec[i] -= shares_to_sell
                cash += proceeds
                realized_gains_total += rg

                trades.append({
                    "Ticker": tickers[i],
                    "Side": "SELL",
                    "Shares": int(shares_to_sell) if self.integer_shares else shares_to_sell,
                    "Price": price[i],
                    "Proceeds": proceeds,
                    "RealizedGain": rg
                })
                holdings_delta, sec_val, total_val = refresh_deltas(qty_sec, cash)

        # ---------------- STEP 2: Sell gainers if ΔU < 0 ----------------
        def delta_te2_per_dollar_sell(i: int, q: np.ndarray) -> float:
            if q[i] <= 0 or price[i] <= 0:
                return 0.0
            d_shares = 1.0 / price[i]
            q_new = q.copy()
            q_new[i] = max(0.0, q_new[i] - d_shares)
            return te2(q_new) - te2(q)

        def tax_per_dollar_sell(i: int) -> float:
            if price[i] <= 0:
                return 0.0
            gain_per_share = price[i] - basis[i]
            return self.ltcg * (gain_per_share / price[i])

        def objective_delta_per_dollar_sell(i: int, q: np.ndarray) -> float:
            return delta_te2_per_dollar_sell(i, q) + self.tax_weight * tax_per_dollar_sell(i)

        gainers = [i for i in range(n) if price[i] > basis[i] and holdings_delta[i] < -1e-12]
        per_dollar_scores = [(objective_delta_per_dollar_sell(i, qty_sec), i) for i in gainers]
        per_dollar_scores.sort(key=lambda x: x[0])

        for dU_per_dollar, i in per_dollar_scores:
            if dU_per_dollar >= 0:
                continue
            desired_sell_value = min(-holdings_delta[i], qty_sec[i] * price[i])
            if desired_sell_value <= 1e-9:
                continue
            shares_to_sell = desired_sell_value / price[i]
            if self.integer_shares:
                shares_to_sell = math.floor(min(shares_to_sell, qty_sec[i]))
                if shares_to_sell <= 0:
                    continue
            else:
                shares_to_sell = min(shares_to_sell, qty_sec[i])

            proceeds = shares_to_sell * price[i]
            rg = shares_to_sell * (price[i] - basis[i])
            qty_sec[i] -= shares_to_sell
            cash += proceeds
            realized_gains_total += rg

            trades.append({
                "Ticker": tickers[i],
                "Side": "SELL",
                "Shares": int(shares_to_sell) if self.integer_shares else shares_to_sell,
                "Price": price[i],
                "Proceeds": proceeds,
                "RealizedGain": rg
            })
            holdings_delta, sec_val, total_val = refresh_deltas(qty_sec, cash)

        # ---------------- STEP 3: Buy underweights (respect min cash %) ----------------
        min_cash_abs = self.min_cash_pct * total_val
        spendable_cash = max(0.0, cash - min_cash_abs)
        under_dollar = np.maximum(holdings_delta, 0.0)
        total_under = float(np.sum(under_dollar))

        if total_under > 1e-9 and spendable_cash > 1e-9:
            if not self.integer_shares:
                for i in range(n):
                    if under_dollar[i] <= 0:
                        continue
                    alloc = spendable_cash * (under_dollar[i] / total_under)
                    shares_to_buy = alloc / price[i]
                    cost = shares_to_buy * price[i]
                    qty_sec[i] += shares_to_buy
                    cash -= cost
                    trades.append({
                        "Ticker": tickers[i],
                        "Side": "BUY",
                        "Shares": shares_to_buy,
                        "Price": price[i],
                        "Proceeds": -cost,
                        "RealizedGain": 0.0
                    })
            else:
                for _ in range(5000):
                    sec_val = float(np.dot(qty_sec, price))
                    total_val = sec_val + cash
                    min_cash_abs = self.min_cash_pct * total_val
                    spendable_cash = cash - min_cash_abs
                    if spendable_cash < min(price):
                        break
                    holdings_delta, _, _ = refresh_deltas(qty_sec, cash)
                    under_dollar = np.maximum(holdings_delta, 0.0)
                    if np.sum(under_dollar) <= 1e-9:
                        break
                    gaps = [(i, (tgt_w_sec[i] - (qty_sec[i] * price[i]) / total_val) / price[i])
                            for i in range(n) if under_dollar[i] > 0 and price[i] <= spendable_cash]
                    if not gaps:
                        break
                    gaps.sort(key=lambda x: x[1], reverse=True)
                    i_pick = gaps[0][0]
                    qty_sec[i_pick] += 1.0
                    cash -= price[i_pick]
                    trades.append({
                        "Ticker": tickers[i_pick],
                        "Side": "BUY",
                        "Shares": 1,
                        "Price": price[i_pick],
                        "Proceeds": -price[i_pick],
                        "RealizedGain": 0.0
                    })

        # ---------------- Summary ----------------
        final_te = math.sqrt(max(te2(qty_sec), 0.0))
        sec_val_final = float(np.dot(qty_sec, price))
        total_val_final = sec_val_final + cash
        cash_end_pct = cash / total_val_final if total_val_final > 1e-12 else 0.0
        realized_gain = realized_gains_total
        est_tax_cost = self.ltcg * realized_gain

        post_alloc = []
        for i, t in enumerate(tickers):
            post_alloc.append({
                "Ticker": t,
                "FinalQty": qty_sec[i],
                "Price": price[i],
                "Final$": qty_sec[i] * price[i],
                "FinalWeight": (qty_sec[i] * price[i]) / total_val_final,
                "TargetWeight": tgt_w_sec[i]
            })
        post_alloc.append({
            "Ticker": "CASH",
            "FinalQty": cash,
            "Price": 1.0,
            "Final$": cash,
            "FinalWeight": cash_end_pct,
            "TargetWeight": tgt_w_cash
        })

        return {
            "initial_tracking_error": initial_te,
            "final_tracking_error": final_te,
            "realized_net_gains": realized_gain,
            "estimated_tax_cost": est_tax_cost,
            "cash_start_pct": cash_start_pct,
            "cash_end_pct": cash_end_pct,
            "total_traded": sum(abs(tr["Proceeds"]) for tr in trades),
            "trades": trades,
            "post_allocation": post_alloc
        }
