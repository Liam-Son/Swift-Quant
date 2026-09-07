"""
crypto_wf_engine.py

Single-file crypto perp/spot backtest engine with:
- data validation
- walk-forward testing
- tradable ranking
- concentration limits
- historical or static funding
- fees, spread, slippage
- portfolio equity tracking
- maintenance margin
- partial liquidation
- liquidation penalties
- OOS reporting
- pytest-style tests
- optional CLI runner

Save as: crypto_wf_engine.py
Run tests: pytest crypto_wf_engine.py -q
Run CLI:   python crypto_wf_engine.py --input data.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

ASSETS = ["XRP", "XLM", "XDC", "PI"]

TRAIN_DAYS = 365
TEST_DAYS = 90
STEP_DAYS = 90

THRESHOLDS = np.round(np.arange(0.70, 0.96, 0.01), 2).tolist()
NEIGHBOR_EPS = 0.03

TARGET_GROSS_LEV = 1.0
LONG_N = 1
SHORT_N = 1

STRESS_Q = 0.85
SKIP_Q = 0.95

FEE_BPS = 10.0
SPREAD_BPS = 5.0
BASE_SLIP_BPS = 5.0
MAX_SLIP_MULT = 3.0

USE_FUNDING = True
FUNDING_BPS_PER_8H = 0.0
BARS_PER_FUNDING = 8

MAINT_MARGIN_RATE = 0.10
LIQ_PENALTY_RATE = 0.005
PARTIAL_LIQ_TARGET = 1.20

MAX_SINGLE_WEIGHT = 0.40
MAX_TOP2_WEIGHT = 0.70

MIN_OBS_PER_FOLD = 120

OUTPUT_DIR = "output"
ENGINE_VERSION = "2.1.0"


# =========================
# HELPERS
# =========================

def sharpe(x, periods_per_year=365):
    s = pd.Series(x).dropna()
    if len(s) < 5 or s.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * s.mean() / s.std()

def max_drawdown(eq):
    eq = pd.Series(eq).dropna()
    if eq.empty:
        return np.nan
    # Include starting capital so a loss on the first evaluated bar counts.
    eq = pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)], ignore_index=True)
    return (eq / eq.cummax() - 1).min()

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_df(df, name):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    return path


def engine_metadata():
    """Return the assumptions needed to reproduce an engine run."""
    return {
        "engine_version": ENGINE_VERSION,
        "assets": ASSETS,
        "walk_forward": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "step_days": STEP_DAYS},
        "thresholds": THRESHOLDS,
        "target_gross_leverage": TARGET_GROSS_LEV,
        "selection": {"long_n": LONG_N, "short_n": SHORT_N},
        "costs_bps": {"fee": FEE_BPS, "spread": SPREAD_BPS, "base_slippage": BASE_SLIP_BPS},
        "funding": {"enabled": USE_FUNDING, "static_bps_per_8h": FUNDING_BPS_PER_8H},
        "risk": {
            "maintenance_margin_rate": MAINT_MARGIN_RATE,
            "liquidation_penalty_rate": LIQ_PENALTY_RATE,
            "max_single_weight": MAX_SINGLE_WEIGHT,
            "max_top2_weight": MAX_TOP2_WEIGHT,
        },
    }


def save_metadata(name):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(engine_metadata(), handle, indent=2)
    return path


# =========================
# DATA VALIDATION
# =========================

def validate_input(df):
    req = {"date", "asset", "open", "high", "low", "close", "volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df.copy()
    x = x[x["asset"].isin(ASSETS)].copy()
    missing_assets = set(ASSETS) - set(x["asset"].unique())
    if missing_assets:
        raise ValueError(f"Configured assets missing from input: {sorted(missing_assets)}")
    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
    if x["date"].isna().any():
        raise ValueError("Invalid dates found.")

    dup = x.duplicated(["asset", "date"], keep=False)
    if dup.any():
        sample = x.loc[dup, ["asset", "date"]].head(20)
        raise ValueError(f"Duplicate asset/date rows found:\n{sample}")

    if (x[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC prices found.")

    bad_ohlc = ~(
        (x["high"] >= x["low"]) &
        (x["high"] >= x["open"]) &
        (x["high"] >= x["close"]) &
        (x["low"] <= x["open"]) &
        (x["low"] <= x["close"])
    )
    if bad_ohlc.any():
        sample = x.loc[bad_ohlc, ["asset", "date", "open", "high", "low", "close"]].head(20)
        raise ValueError(f"Invalid OHLC rows found:\n{sample}")

    if (x["volume"] < 0).any():
        raise ValueError("Negative volume found.")

    coverage = x.groupby("date")["asset"].nunique()
    if coverage.empty or coverage.min() < len(ASSETS):
        raise ValueError("Incomplete asset coverage on at least one date.")

    return x.sort_values(["date", "asset"]).reset_index(drop=True)


# =========================
# FEATURES
# =========================

def add_features(df):
    x = df.sort_values(["asset", "date"]).copy()
    g = x.groupby("asset", group_keys=False)

    x["ret_1"] = g["close"].pct_change(1)
    x["mom_7"] = g["close"].pct_change(7)
    x["mom_20"] = g["close"].pct_change(20)
    x["ma_20"] = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    x["mr_20"] = -((x["close"] / x["ma_20"]) - 1.0)
    x["vol_20"] = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    x["vol_ref"] = g["vol_20"].rolling(60).median().reset_index(level=0, drop=True)
    x["vol_ratio"] = (x["vol_20"] / (x["vol_ref"] + 1e-12)).replace([np.inf, -np.inf], np.nan)
    return x


# =========================
# CONCENTRATION
# =========================

def enforce_concentration_limits(
    df,
    date_col="date",
    asset_col="asset",
    position_col="position",
    max_single_weight=MAX_SINGLE_WEIGHT,
    max_top2_weight=MAX_TOP2_WEIGHT,
    target_gross_leverage=TARGET_GROSS_LEV,
):
    x = df.copy().sort_values([date_col, asset_col]).reset_index(drop=True)
    out = []

    for dt, g in x.groupby(date_col, sort=True):
        g = g.copy()
        w = g[position_col].astype(float).abs()
        gross = w.sum()
        if gross <= 0:
            out.append(g)
            continue

        g[position_col] = g[position_col] * (target_gross_leverage / gross)
        w = g[position_col].abs()

        # Caps can make target leverage infeasible (for example, two selected
        # assets with a 70% top-two cap). Preserve zero positions and allow the
        # portfolio to run below target instead of inventing exposure.
        signs = np.sign(g[position_col])
        weights = g[position_col].abs().clip(upper=max_single_weight)
        active = weights > 0
        if active.any():
            feasible_gross = min(target_gross_leverage, max_single_weight * active.sum())
            for _ in range(len(weights) + 1):
                deficit = feasible_gross - weights.sum()
                room = (max_single_weight - weights).where(active, 0.0).clip(lower=0)
                if deficit <= 1e-12 or room.sum() <= 1e-12:
                    break
                weights += room / room.sum() * min(deficit, room.sum())

        top = weights.sort_values(ascending=False).index[:2]
        excess = max(0.0, weights.loc[top].sum() - max_top2_weight)
        if excess:
            top_total = weights.loc[top].sum()
            weights.loc[top] *= (top_total - excess) / top_total

        g[position_col] = signs * weights

        out.append(g)

    return pd.concat(out, ignore_index=True)


# =========================
# SIGNALS / SIZING
# =========================

def slippage_bps_frame(df):
    mult = df["vol_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, MAX_SLIP_MULT)
    return BASE_SLIP_BPS * mult

def make_positions(df, stress_q=STRESS_Q, skip_q=SKIP_Q, target_gross_lev=TARGET_GROSS_LEV):
    x = df.copy()

    vol_ok = x["volume"] > x.groupby("asset")["volume"].transform(lambda s: s.rolling(20).mean())
    stress_cut = x.groupby("asset")["vol_20"].transform(lambda s: s.rolling(60).quantile(stress_q))
    skip_cut = x.groupby("asset")["vol_20"].transform(lambda s: s.rolling(60).quantile(skip_q))

    x["regime"] = np.where(
        (x["vol_20"] >= skip_cut) | (~vol_ok),
        "skip",
        np.where(x["vol_20"] >= stress_cut, "stress", "normal")
    )

    tradable = x["regime"] != "skip"

    x["score_raw"] = np.where(
        x["regime"].eq("stress"),
        0.3 * (0.6 * x["mom_7"] + 0.4 * x["mom_20"]) + 0.7 * x["mr_20"],
        0.7 * (0.6 * x["mom_7"] + 0.4 * x["mom_20"]) + 0.3 * x["mr_20"],
    )
    x["score"] = x["score_raw"] / (x["vol_20"].replace(0, np.nan) + 1e-12)

    x["rank"] = np.nan
    x.loc[tradable, "rank"] = x.loc[tradable].groupby("date")["score"].rank(method="first", ascending=False)

    n_tradable = x.groupby("date")["rank"].transform(lambda s: s.notna().sum())
    x["position"] = 0.0
    enough_assets = n_tradable >= LONG_N + SHORT_N
    x.loc[tradable & enough_assets & (x["rank"] <= LONG_N), "position"] = 1.0 / LONG_N
    x.loc[tradable & enough_assets & (x["rank"] > n_tradable - SHORT_N), "position"] = -1.0 / SHORT_N

    gross = x.groupby("date")["position"].transform(lambda s: s.abs().sum())
    scale = np.where(gross > 0, target_gross_lev / gross, 0.0)
    x["position"] = x["position"] * scale

    return x


# =========================
# FUNDING
# =========================

def apply_funding(portfolio, use_funding=False, funding_df=None, funding_rate_col="funding_rate", funding_bps_per_8h=0.0):
    x = portfolio.copy()
    if not use_funding:
        x["funding_rate"] = 0.0
        x["funding_pnl"] = 0.0
        return x

    if funding_df is not None:
        f = funding_df.copy()
        f["date"] = pd.to_datetime(f["date"], utc=True)
        if "asset" not in f.columns:
            raise ValueError("funding_df must contain asset")
        if "fundingRate" in f.columns and funding_rate_col not in f.columns:
            f = f.rename(columns={"fundingRate": funding_rate_col})
        elif funding_rate_col not in f.columns:
            raise ValueError("funding_df must contain fundingRate or funding_rate")
        x = x.drop(columns=[funding_rate_col], errors="ignore")
        x = x.merge(f[["date", "asset", funding_rate_col]], on=["date", "asset"], how="left")
        x[funding_rate_col] = x.groupby("asset")[funding_rate_col].ffill().fillna(0.0)
        x["funding_rate"] = x[funding_rate_col]
    else:
        x["funding_rate"] = funding_bps_per_8h / 10000.0

    x["funding_pnl"] = -x["prev_position"] * x["funding_rate"]
    return x


# =========================
# COSTS / RETURNS
# =========================

def apply_costs(
    df,
    fee_bps=FEE_BPS,
    spread_bps=SPREAD_BPS,
    base_slip_bps=BASE_SLIP_BPS,
    use_funding=USE_FUNDING,
    funding_df=None,
    funding_bps_per_8h=FUNDING_BPS_PER_8H,
):
    x = df.copy().sort_values(["asset", "date"])
    x["prev_position"] = x.groupby("asset")["position"].shift(1).fillna(0.0)
    x["asset_ret"] = x.groupby("asset")["close"].pct_change()
    x["turnover"] = (x["position"] - x["prev_position"]).abs()

    slip = slippage_bps_frame(x)
    total_cost_bps = fee_bps + spread_bps + slip
    x["trade_cost"] = x["turnover"] * (total_cost_bps / 10000.0)

    x = apply_funding(x, use_funding=use_funding, funding_df=funding_df, funding_bps_per_8h=funding_bps_per_8h)

    # Yesterday's close-position earns today's close-to-close return.
    x["gross_ret"] = x["prev_position"] * x["asset_ret"]
    x["net_ret"] = x["gross_ret"] - x["trade_cost"] + x["funding_pnl"]
    return x


def aggregate_portfolio(bt):
    """Collapse asset rows into one investable portfolio return per timestamp."""
    if bt.empty:
        return pd.DataFrame(columns=["date", "gross_ret", "trade_cost", "funding_pnl", "net_ret", "turnover"])
    return (
        bt.groupby("date", sort=True, as_index=False)
        .agg(
            gross_ret=("gross_ret", "sum"),
            trade_cost=("trade_cost", "sum"),
            funding_pnl=("funding_pnl", "sum"),
            net_ret=("net_ret", "sum"),
            turnover=("turnover", "sum"),
        )
    )


# =========================
# MARGIN / LIQUIDATION
# =========================

def _calc_state(pos, price, entry_price, cash, maint_rate):
    equity = cash + sum(pos[a] * (price[a] - entry_price[a]) for a in pos)
    maintenance = sum(abs(pos[a]) * price[a] for a in pos) * maint_rate
    return equity, maintenance

def apply_margin(
    portfolio,
    maintenance_margin_rate=MAINT_MARGIN_RATE,
    liquidation_penalty_rate=LIQ_PENALTY_RATE,
    partial_liquidation=True,
    partial_liquidation_target=PARTIAL_LIQ_TARGET,
    date_col="date",
    asset_col="asset",
    price_col="close",
):
    x = portfolio.copy().sort_values([date_col, asset_col]).reset_index(drop=True)

    if "cash" not in x.columns:
        x["cash"] = 0.0
    if "position" not in x.columns:
        raise ValueError("portfolio must contain a position column")

    out = []
    entry_price = x.groupby(asset_col)[price_col].first().astype(float).to_dict()

    for dt, g in x.groupby(date_col, sort=True):
        g = g.copy().sort_values(asset_col).reset_index(drop=True)
        cash = float(g["cash"].iloc[0])
        pos = g.set_index(asset_col)["position"].astype(float).to_dict()
        price = g.set_index(asset_col)[price_col].astype(float).to_dict()

        equity, maintenance = _calc_state(pos, price, entry_price, cash, maintenance_margin_rate)
        liquidated = False
        total_penalty = 0.0
        events = []

        while equity <= maintenance and any(abs(v) > 0 for v in pos.values()):
            liquidated = True

            biggest = max(pos.keys(), key=lambda a: abs(pos[a]) * price[a])
            q = pos[biggest]
            p = price[biggest]

            current_equity, current_maintenance = _calc_state(pos, price, entry_price, cash, maintenance_margin_rate)
            target_equity = current_maintenance * partial_liquidation_target if partial_liquidation else current_maintenance
            deficit = max(0.0, target_equity - current_equity)

            per_unit_relief = p * (1.0 - liquidation_penalty_rate - maintenance_margin_rate)
            if per_unit_relief <= 0:
                close_amt = abs(q)
            else:
                close_amt = min(abs(q), deficit / per_unit_relief if deficit > 0 else abs(q))

            if close_amt <= 0:
                close_amt = abs(q)

            penalty = close_amt * p * liquidation_penalty_rate
            signed_close = np.sign(q) * close_amt
            cash += signed_close * (p - entry_price[biggest])
            cash -= penalty
            total_penalty += penalty
            pos[biggest] = q - np.sign(q) * close_amt

            equity, maintenance = _calc_state(pos, price, entry_price, cash, maintenance_margin_rate)
            events.append({"date": dt, "asset": biggest, "close_amt": close_amt, "penalty": penalty})

            if not partial_liquidation:
                for a in pos:
                    pos[a] = 0.0
                equity, maintenance = _calc_state(pos, price, entry_price, cash, maintenance_margin_rate)
                break

            if equity > maintenance:
                break

        row = g.copy()
        row["cash"] = cash
        row["equity"] = equity
        row["maintenance_margin"] = maintenance
        row["liquidated"] = liquidated
        row["liquidation_penalty"] = total_penalty

        for a in pos:
            row.loc[row[asset_col] == a, "position"] = pos[a]

        out.append(row)

    out = pd.concat(out, ignore_index=True)
    unrealized = out.apply(lambda r: r["position"] * (r[price_col] - entry_price[r[asset_col]]), axis=1)
    out["portfolio_equity"] = out.groupby(date_col)["cash"].transform("first") + unrealized.groupby(out[date_col]).transform("sum")
    notionals = out["position"].abs() * out[price_col]
    out["portfolio_maintenance_margin"] = notionals.groupby(out[date_col]).transform("sum") * maintenance_margin_rate

    return out


# =========================
# SIMULATOR
# =========================

def simulate_portfolio(
    df,
    initial_cash=1000.0,
    leverage=1.0,
    position_col="position",
    price_col="close",
    asset_col="asset",
    date_col="date",
    use_funding=False,
    funding_rate_col="funding_rate",
    funding_df=None,
    fee_bps=FEE_BPS,
    spread_bps=SPREAD_BPS,
    base_slip_bps=BASE_SLIP_BPS,
):
    x = df.copy().sort_values([date_col, asset_col]).reset_index(drop=True)

    req = {date_col, asset_col, price_col, position_col}
    missing = req - set(x.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = x.copy()
    if "cash" not in x.columns:
        x["cash"] = np.nan

    out = []
    cash = float(initial_cash)
    prev_positions: Dict[str, float] = {}

    for dt, g in x.groupby(date_col, sort=True):
        g = g.copy().sort_values(asset_col).reset_index(drop=True)
        rows = []

        for _, row in g.iterrows():
            asset = row[asset_col]
            price = float(row[price_col])
            target_pos = float(row[position_col])
            prev_pos = float(prev_positions.get(asset, 0.0))
            delta_pos = target_pos - prev_pos

            trade_notional = abs(delta_pos) * price
            trade_cost = trade_notional * ((fee_bps + spread_bps) / 10000.0)

            if use_funding:
                funding_rate = float(row[funding_rate_col]) if funding_rate_col in row and pd.notna(row[funding_rate_col]) else 0.0
                funding_pnl = -prev_pos * funding_rate
            else:
                funding_rate = 0.0
                funding_pnl = 0.0

            cash -= trade_cost
            cash += funding_pnl

            rows.append({
                date_col: dt,
                asset_col: asset,
                price_col: price,
                "position": target_pos,
                "prev_position": prev_pos,
                "delta_position": delta_pos,
                "trade_notional": trade_notional,
                "trade_cost": trade_cost,
                "funding_rate": funding_rate,
                "funding_pnl": funding_pnl,
                "cash_pre_margin": cash,
            })

            prev_positions[asset] = target_pos

        day = pd.DataFrame(rows)
        day["position_value"] = day["position"] * day[price_col]
        portfolio_cash = float(day["cash_pre_margin"].iloc[-1])
        day["cash"] = portfolio_cash
        day["portfolio_equity_pre_margin"] = portfolio_cash + day["position_value"].sum()

        out.append(day)

    portfolio = pd.concat(out, ignore_index=True)

    if funding_df is not None or use_funding:
        portfolio = apply_funding(portfolio, use_funding=use_funding, funding_df=funding_df, funding_rate_col=funding_rate_col)

    portfolio = apply_margin(portfolio)
    return portfolio


# =========================
# PLATEAU
# =========================

def plateau_score(grid_df, metric_col="train_sharpe", eps=NEIGHBOR_EPS):
    rows = []
    qs = sorted(grid_df["threshold"].dropna().unique())
    for q in qs:
        neigh = grid_df[grid_df["threshold"].between(q - eps, q + eps)]
        if neigh.empty:
            continue
        center = neigh[np.isclose(neigh["threshold"], q)]
        if center.empty:
            continue
        rows.append({
            "threshold": q,
            "neighbor_mean": neigh[metric_col].mean(),
            "neighbor_median": neigh[metric_col].median(),
            "center_metric": center.iloc[0][metric_col],
            "plateau_width": neigh["threshold"].nunique(),
            "plateau_score": 0.7 * neigh[metric_col].mean() + 0.3 * center.iloc[0][metric_col],
        })
    return pd.DataFrame(rows)

def choose_plateau(grid_df):
    p = plateau_score(grid_df, eps=NEIGHBOR_EPS)
    if p.empty:
        best = grid_df.sort_values("train_sharpe", ascending=False).iloc[0]
        return float(best["threshold"]), p
    best = p.sort_values(["plateau_score", "plateau_width", "center_metric"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), p


# =========================
# EVALUATION
# =========================

def evaluate_threshold(df, q, evaluation_dates=None):
    pos = make_positions(df, stress_q=q, skip_q=SKIP_Q, target_gross_lev=TARGET_GROSS_LEV)
    pos = enforce_concentration_limits(pos)
    bt = apply_costs(pos)
    if evaluation_dates is not None:
        bt = bt[bt["date"].isin(evaluation_dates)].copy()
    daily = aggregate_portfolio(bt).set_index("date")
    s = daily["net_ret"].dropna()
    eq = (1 + s).cumprod()

    return {
        "threshold": q,
        "train_sharpe": sharpe(s),
        "train_ret": eq.iloc[-1] - 1 if len(eq) else np.nan,
        "train_mdd": max_drawdown(eq),
        "train_turnover": daily["turnover"].mean(),
        "trade_count": int((bt["turnover"] > 0).sum()),
        "skip_rate": float((bt["regime"] == "skip").mean()),
        "mean_trade_cost": daily["trade_cost"].mean(),
        "mean_funding_pnl": daily["funding_pnl"].mean(),
    }

def summarize_oos(oos_table, equity_table=None):
    if oos_table.empty:
        return pd.DataFrame()

    out = {
        "folds": len(oos_table),
        "mean_oos_sharpe": oos_table["oos_sharpe"].mean(),
        "median_oos_sharpe": oos_table["oos_sharpe"].median(),
        "mean_oos_ret": oos_table["oos_ret"].mean(),
        "median_oos_ret": oos_table["oos_ret"].median(),
        "mean_oos_mdd": oos_table["oos_mdd"].mean(),
        "mean_oos_turnover": oos_table["oos_turnover"].mean(),
        "mean_skip_rate": oos_table["skip_rate"].mean(),
        "mean_oos_trade_cost": oos_table["mean_trade_cost"].mean(),
        "mean_oos_funding_pnl": oos_table["mean_funding_pnl"].mean(),
        "positive_folds": int((oos_table["oos_sharpe"] > 0).sum()),
    }
    if equity_table is not None and not equity_table.empty:
        stitched = equity_table.sort_values("date").drop_duplicates("date")
        returns = stitched["net_ret"].dropna()
        equity = (1.0 + returns).cumprod()
        out.update({
            "stitched_oos_sharpe": sharpe(returns),
            "stitched_oos_ret": equity.iloc[-1] - 1.0 if len(equity) else np.nan,
            "stitched_oos_mdd": max_drawdown(equity),
            "oos_days": len(returns),
        })
    return pd.DataFrame([out])


# =========================
# WALK-FORWARD
# =========================

def walk_forward_engine(df):
    d = validate_input(df)
    d = add_features(d).dropna().reset_index(drop=True)
    dates = pd.Index(sorted(d["date"].unique()))

    grid_rows, chosen_rows, oos_rows, equity_rows = [], [], [], []
    start, fold = 0, 0

    while True:
        tr0, tr1 = start, start + TRAIN_DAYS
        te0, te1 = tr1, tr1 + TEST_DAYS
        if te1 > len(dates):
            break

        train_dates = dates[tr0:tr1]
        test_dates = dates[te0:te1]
        train = d[d["date"].isin(train_dates)].copy()
        test = d[d["date"].isin(test_dates)].copy()

        if len(train) < MIN_OBS_PER_FOLD or len(test) < 10:
            start += STEP_DAYS
            fold += 1
            continue

        fold_grid = []
        for q in THRESHOLDS:
            r = evaluate_threshold(train, q)
            r.update({"fold": fold, "split": "train"})
            fold_grid.append(r)

        fold_grid_df = pd.DataFrame(fold_grid)
        best_q, plateau_tbl = choose_plateau(fold_grid_df)

        chosen_rows.append({
            "fold": fold,
            "chosen_threshold": best_q,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
        })

        # Include trailing training history so OOS rolling signals do not restart.
        test_context = pd.concat([train.groupby("asset", group_keys=False).tail(60), test])
        test_res = evaluate_threshold(test_context, best_q, evaluation_dates=test_dates)
        test_res.update({
            "fold": fold,
            "chosen_threshold": best_q,
            "oos_sharpe": test_res.pop("train_sharpe"),
            "oos_ret": test_res.pop("train_ret"),
            "oos_mdd": test_res.pop("train_mdd"),
            "oos_turnover": test_res.pop("train_turnover"),
        })
        oos_rows.append(test_res)

        test_pos = make_positions(test_context, stress_q=best_q, skip_q=SKIP_Q, target_gross_lev=TARGET_GROSS_LEV)
        test_pos = enforce_concentration_limits(test_pos)
        test_bt = apply_costs(test_pos)
        test_bt = test_bt[test_bt["date"].isin(test_dates)].copy()
        test_bt["fold"] = fold
        test_bt["chosen_threshold"] = best_q
        daily_bt = aggregate_portfolio(test_bt)
        daily_bt["fold"] = fold
        daily_bt["chosen_threshold"] = best_q
        daily_bt["equity"] = (1 + daily_bt["net_ret"].fillna(0.0)).cumprod()
        equity_rows.append(daily_bt)

        if not plateau_tbl.empty:
            plateau_tbl = plateau_tbl.copy()
            plateau_tbl["fold"] = fold
            plateau_tbl["split"] = "plateau"
            grid_rows.append(plateau_tbl)
        grid_rows.append(fold_grid_df)

        start += STEP_DAYS
        fold += 1

    grid_table = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    chosen_table = pd.DataFrame(chosen_rows)
    oos_table = pd.DataFrame(oos_rows)
    equity_table = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    if not equity_table.empty:
        equity_table = equity_table.sort_values(["date", "fold"]).drop_duplicates("date", keep="first").reset_index(drop=True)
        equity_table["equity"] = (1.0 + equity_table["net_ret"].fillna(0.0)).cumprod()

    return grid_table, chosen_table, oos_table, equity_table


# =========================
# REPORT
# =========================

def run_engine(input_csv, out_prefix="crypto_wf"):
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    grid_table, chosen_table, oos_table, equity_table = walk_forward_engine(df)
    summary = summarize_oos(oos_table, equity_table)

    save_df(grid_table, f"{out_prefix}_threshold_grid.csv")
    save_df(chosen_table, f"{out_prefix}_chosen_thresholds.csv")
    save_df(oos_table, f"{out_prefix}_oos_results.csv")
    save_df(equity_table, f"{out_prefix}_oos_equity.csv")
    save_df(summary, f"{out_prefix}_summary.csv")
    save_metadata(f"{out_prefix}_metadata.json")

    return {
        "grid_table": grid_table,
        "chosen_table": chosen_table,
        "oos_table": oos_table,
        "equity_table": equity_table,
        "summary": summary,
    }


# =========================
# TESTS
# =========================

def _make_synth_df():
    dates = pd.date_range("2023-01-01", periods=600, freq="D", tz="UTC")
    assets = ASSETS
    rows = []
    for a in assets:
        p = 100.0 if a != "XLM" else 50.0
        for d in dates:
            ret = 0.001 if a in ["XRP", "XLM"] else -0.001
            p *= (1 + ret)
            rows.append({
                "date": d,
                "asset": a,
                "open": p * 0.99,
                "high": p * 1.01,
                "low": p * 0.98,
                "close": p,
                "volume": 1_000_000 + (d.dayofyear % 7) * 10_000,
            })
    return pd.DataFrame(rows)

def _make_synth_funding():
    dates = pd.date_range("2023-01-01", periods=600, freq="D", tz="UTC")
    rows = []
    for a in ASSETS:
        for i, d in enumerate(dates):
            rows.append({"date": d, "asset": a, "fundingRate": 0.0001 if i % 2 == 0 else -0.00005})
    return pd.DataFrame(rows)

def test_validate_input_accepts_clean_data():
    df = _make_synth_df()
    out = validate_input(df)
    assert len(out) == len(df)

def test_validate_input_rejects_duplicates():
    df = _make_synth_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    try:
        validate_input(df)
        assert False
    except ValueError:
        assert True

def test_concentration_caps_hold():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        "asset": ASSETS,
        "position": [0.8, -0.1, 0.05, 0.05],
    })
    out = enforce_concentration_limits(df, max_single_weight=0.40, max_top2_weight=0.70, target_gross_leverage=1.0)
    w = out["position"].abs()
    assert w.max() <= 0.40 + 1e-9
    assert w.sort_values(ascending=False).iloc[:2].sum() <= 0.70 + 1e-9
    assert w.sum() <= 1.0 + 1e-9
    assert abs(w.sum() - 1.0) <= 1e-9

def test_concentration_does_not_create_new_positions():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        "asset": ASSETS,
        "position": [0.5, -0.5, 0.0, 0.0],
    })
    out = enforce_concentration_limits(df)
    assert (out.loc[out["asset"].isin(["XDC", "PI"]), "position"] == 0).all()
    assert out["position"].abs().sum() <= MAX_TOP2_WEIGHT + 1e-9

def test_apply_costs_never_crosses_asset_boundaries():
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "asset": ["XRP"] * 3 + ["XLM"] * 3,
        "close": [100.0, 110.0, 121.0, 50.0, 50.0, 50.0],
        "position": [1.0] * 6,
        "vol_ratio": [1.0] * 6,
    })
    out = apply_costs(df, use_funding=False)
    xrp = out[out["asset"] == "XRP"].sort_values("date")
    xlm = out[out["asset"] == "XLM"].sort_values("date")
    assert np.allclose(xrp["asset_ret"].iloc[1:], 0.10)
    assert np.allclose(xlm["asset_ret"].iloc[1:], 0.0)

def test_portfolio_aggregation_produces_one_row_per_date():
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bt = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "gross_ret": [0.01, -0.005] * 3,
        "trade_cost": [0.001, 0.001] * 3,
        "funding_pnl": [0.0] * 6,
        "net_ret": [0.009, -0.006] * 3,
        "turnover": [0.5, 0.5] * 3,
    })
    daily = aggregate_portfolio(bt)
    assert len(daily) == 3
    assert np.allclose(daily["net_ret"], 0.003)

def test_drawdown_includes_first_period_loss():
    assert np.isclose(max_drawdown(pd.Series([0.90, 0.95])), -0.10)

def test_metadata_captures_reproducibility_assumptions():
    metadata = engine_metadata()
    assert metadata["engine_version"] == ENGINE_VERSION
    assert metadata["assets"] == ASSETS
    assert metadata["walk_forward"]["test_days"] == TEST_DAYS
    assert metadata["costs_bps"]["fee"] == FEE_BPS

def test_negative_funding_hurts_shorts_with_zero_price_returns():
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": dates,
        "asset": ["XRP"] * 4,
        "open": [100.0] * 4,
        "high": [100.0] * 4,
        "low": [100.0] * 4,
        "close": [100.0] * 4,
        "volume": [1_000_000] * 4,
        "position": [1.0, -1.0, 1.0, -1.0],
    })
    df["prev_position"] = df["position"].shift(1).fillna(0.0)
    df["funding_rate"] = [-0.001, -0.001, 0.001, 0.001]
    df["funding_pnl"] = -df["prev_position"] * df["funding_rate"]
    short_rows = df["prev_position"] < 0
    long_rows = df["prev_position"] > 0
    assert df.loc[short_rows & (df["funding_rate"] < 0), "funding_pnl"].sum() < df.loc[short_rows & (df["funding_rate"] > 0), "funding_pnl"].sum()
    assert df.loc[long_rows & (df["funding_rate"] < 0), "funding_pnl"].sum() > df.loc[long_rows & (df["funding_rate"] > 0), "funding_pnl"].sum()

def test_historical_funding_series_merges_by_symbol_and_date():
    df = add_features(_make_synth_df()).dropna().reset_index(drop=True)
    pos = make_positions(df)
    funding = _make_synth_funding()
    out = apply_costs(pos, use_funding=True, funding_df=funding)
    assert "funding_rate" in out.columns
    assert out["funding_rate"].isna().sum() == 0

def test_liquidation_triggers_with_penalty_fee():
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": dates,
        "asset": ["XRP"] * 4,
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [100.0, 100.0, 100.0, 100.0],
        "low": [100.0, 92.0, 90.0, 85.0],
        "close": [100.0, 92.0, 90.0, 85.0],
        "volume": [1_000_000] * 4,
        "position": [10.0] * 4,
        "cash": [200.0] * 4,
    })
    out = apply_margin(df, maintenance_margin_rate=0.10, liquidation_penalty_rate=0.005, partial_liquidation=True)
    assert "liquidated" in out.columns
    assert "liquidation_penalty" in out.columns
    liq = out[out["liquidated"]]
    assert len(liq) > 0
    assert liq["liquidation_penalty"].sum() > 0

def test_partial_liquidation_reduces_position_not_full_close():
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "asset": ["XRP"] * 4 + ["XLM"] * 4,
        "open": [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
        "high": [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
        "low": [100.0, 92.0, 90.0, 95.0, 50.0, 52.0, 55.0, 58.0],
        "close": [100.0, 92.0, 90.0, 95.0, 50.0, 52.0, 55.0, 58.0],
        "volume": [1_000_000] * 8,
        "position": [10.0] * 4 + [-10.0] * 4,
        "cash": [290.0] * 8,
    })
    out = apply_margin(df, maintenance_margin_rate=0.10, liquidation_penalty_rate=0.005, partial_liquidation=True)
    liq = out[out["liquidated"]]
    assert len(liq) > 0
    assert (liq["position"].abs() < 10.0).any()

def test_multi_asset_portfolio_liquidation_triggers_on_combined_equity_breach():
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "asset": ["XRP"] * 4 + ["XLM"] * 4,
        "open":  [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
        "high":  [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
        "low":   [100.0, 92.0, 90.0, 85.0, 50.0, 52.0, 55.0, 58.0],
        "close": [100.0, 92.0, 90.0, 85.0, 50.0, 52.0, 55.0, 58.0],
        "volume": [1_000_000] * 8,
        "position": [10.0] * 4 + [-10.0] * 4,
        "cash": [300.0] * 8,
    })
    out = apply_margin(df, maintenance_margin_rate=0.10, liquidation_penalty_rate=0.005, partial_liquidation=True)
    assert "portfolio_equity" in out.columns
    assert "portfolio_maintenance_margin" in out.columns
    assert (out["portfolio_equity"] > out["portfolio_maintenance_margin"]).any()

def test_walk_forward_runs():
    df = _make_synth_df()
    grid, chosen, oos, eq = walk_forward_engine(df)
    assert isinstance(grid, pd.DataFrame)
    assert isinstance(chosen, pd.DataFrame)
    assert isinstance(oos, pd.DataFrame)
    assert isinstance(eq, pd.DataFrame)
    assert not grid.empty
    assert not chosen.empty
    assert not oos.empty
    assert not eq.empty
    assert eq["date"].is_unique
    assert eq["turnover"].sum() > 0


# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", action="version", version=f"%(prog)s {ENGINE_VERSION}")
    p.add_argument("--input", required=False, help="CSV input with date, asset, open, high, low, close, volume")
    p.add_argument("--out-prefix", default="crypto_wf")
    args = p.parse_args()

    if args.input:
        result = run_engine(args.input, out_prefix=args.out_prefix)
        print(result["summary"].to_string(index=False))
    else:
        df = _make_synth_df()
        grid, chosen, oos, eq = walk_forward_engine(df)
        summary = summarize_oos(oos, eq)
        print(summary.to_string(index=False))
        save_df(grid, f"{args.out_prefix}_threshold_grid.csv")
        save_df(chosen, f"{args.out_prefix}_chosen_thresholds.csv")
        save_df(oos, f"{args.out_prefix}_oos_results.csv")
        save_df(eq, f"{args.out_prefix}_oos_equity.csv")
        save_df(summary, f"{args.out_prefix}_summary.csv")
        save_metadata(f"{args.out_prefix}_metadata.json")

if __name__ == "__main__":
    main()
