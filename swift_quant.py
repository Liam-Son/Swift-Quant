"""Walk-forward crypto strategy engine."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ASSETS = ["XRP", "XLM", "XDC", "PI"]
TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 365, 90, 90
THRESHOLDS = [0.80, 0.85, 0.90]
NEIGHBOR_EPS = 0.05
MOM_WEIGHTS_NORMAL = (0.70, 0.30)
MOM_WEIGHTS_STRESS = (0.30, 0.70)
FEE_BPS, SPREAD_BPS, BASE_SLIP_BPS = 10, 5, 5
MAX_SLIP_MULT, STRESS_Q, SKIP_Q = 3.0, 0.85, 0.95
LONG_N, SHORT_N, MIN_OBS_PER_FOLD = 1, 1, 120
REQUIRED_COLUMNS = {"date", "asset", "open", "high", "low", "close", "volume"}


def sharpe(values):
    values = pd.Series(values).dropna()
    if len(values) < 5 or values.std() == 0:
        return np.nan
    return np.sqrt(252) * values.mean() / values.std()


def max_drawdown(equity):
    equity = pd.Series(equity).dropna()
    return np.nan if equity.empty else (equity / equity.cummax() - 1).min()


def zscore(series, window=20, eps=1e-12):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + eps)


def add_features(df):
    x = df.sort_values(["asset", "date"]).copy()
    g = x.groupby("asset", group_keys=False)
    x["ret_1"] = g["close"].pct_change(1)
    x["logret"] = np.log(x["close"]).groupby(x["asset"]).diff()
    x["mom_7"] = g["close"].pct_change(7)
    x["mom_20"] = g["close"].pct_change(20)
    x["ma_20"] = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    x["mr_20"] = -(x["close"] / x["ma_20"] - 1.0)
    x["vol_20"] = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    x["vol_ref"] = x.groupby("asset")["vol_20"].rolling(60).median().reset_index(level=0, drop=True)
    x["vol_z"] = x.groupby("asset")["vol_20"].transform(lambda s: zscore(s, 60))
    x["vol_surprise"] = (x["vol_20"] / (x["vol_ref"] + 1e-12)).replace([np.inf, -np.inf], np.nan)
    return x


def make_positions(df, stress_q=STRESS_Q, skip_q=SKIP_Q):
    x = df.copy()
    by_asset = x.groupby("asset")
    stress_cut = by_asset["vol_20"].transform(lambda s: s.rolling(60).quantile(stress_q))
    skip_cut = by_asset["vol_20"].transform(lambda s: s.rolling(60).quantile(skip_q))
    vol_ok = x["volume"] > by_asset["volume"].transform(lambda s: s.rolling(20).mean())
    x["regime"] = np.where((x["vol_20"] >= skip_cut) | ~vol_ok, "skip", np.where(x["vol_20"] >= stress_cut, "stress", "normal"))
    x["score_mom"] = 0.6 * x["mom_7"] + 0.4 * x["mom_20"]
    wm = np.where(x["regime"].eq("stress"), MOM_WEIGHTS_STRESS[0], MOM_WEIGHTS_NORMAL[0])
    wr = np.where(x["regime"].eq("stress"), MOM_WEIGHTS_STRESS[1], MOM_WEIGHTS_NORMAL[1])
    x["score"] = (wm * x["score_mom"] + wr * x["mr_20"]) / (x["vol_20"].replace(0, np.nan) + 1e-12)
    x["rank"] = x.groupby("date")["score"].rank(method="first", ascending=False)
    count = x.groupby("date")["asset"].transform("count")
    x["position"] = 0.0
    active = x["regime"] != "skip"
    x.loc[active & (x["rank"] <= LONG_N), "position"] = 1.0 / LONG_N
    x.loc[active & (x["rank"] > count - SHORT_N), "position"] = -1.0 / SHORT_N
    return x


def apply_costs(df):
    x = df.copy()
    x["pos_lag"] = x.groupby("asset")["position"].shift(1).fillna(0.0)
    x["fwd_ret"] = x.groupby("asset")["close"].pct_change().shift(-1)
    x["turnover"] = x.groupby("asset")["position"].diff().abs().fillna(0.0)
    ratio = x["vol_surprise"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, MAX_SLIP_MULT)
    x["cost"] = x["turnover"] * ((FEE_BPS + SPREAD_BPS + BASE_SLIP_BPS * ratio) / 10000.0)
    x["gross_ret"] = x["pos_lag"] * x["fwd_ret"]
    x["net_ret"] = x["gross_ret"] - x["cost"]
    return x


def evaluate_threshold(df, threshold):
    bt = apply_costs(make_positions(df, stress_q=threshold))
    valid = bt["net_ret"].dropna()
    equity = (1 + valid).cumprod()
    return {"threshold": threshold, "sharpe": sharpe(valid), "ret": equity.iloc[-1] - 1 if len(equity) else np.nan,
            "mdd": max_drawdown(equity), "turnover": bt["turnover"].mean(),
            "trade_count": int((bt["turnover"] > 0).sum()), "skip_rate": float((bt["regime"] == "skip").mean())}


def choose_plateau(grid):
    candidates = []
    for q in sorted(grid["threshold"].unique()):
        rows = grid[grid["threshold"].between(q - NEIGHBOR_EPS, q + NEIGHBOR_EPS)]
        center = grid[grid["threshold"] == q].iloc[0]["train_sharpe"]
        candidates.append((0.7 * rows["train_sharpe"].mean() + 0.3 * center, rows["threshold"].nunique(), center, q))
    return float(max(candidates)[-1])


def walk_forward_engine(df):
    data = add_features(df).dropna().copy()
    dates = pd.Index(sorted(data["date"].unique()))
    grids, chosen, results, equities = [], [], [], []
    start = fold = 0
    while start + TRAIN_DAYS + TEST_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        test_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + TEST_DAYS]
        train, test = data[data["date"].isin(train_dates)], data[data["date"].isin(test_dates)]
        if len(train) >= MIN_OBS_PER_FOLD and len(test) >= 10:
            grid = pd.DataFrame([evaluate_threshold(train, q) for q in THRESHOLDS]).rename(columns={"sharpe": "train_sharpe"})
            grid["fold"], grid["split"] = fold, "train"
            best_q = choose_plateau(grid)
            grids.append(grid)
            chosen.append({"fold": fold, "chosen_threshold": best_q, "train_start": train_dates[0], "train_end": train_dates[-1], "test_start": test_dates[0], "test_end": test_dates[-1]})
            result = evaluate_threshold(test, best_q)
            result.update(fold=fold, chosen_threshold=best_q)
            results.append(result)
            bt = apply_costs(make_positions(test, stress_q=best_q))
            bt["fold"], bt["chosen_threshold"] = fold, best_q
            bt["equity"] = (1 + bt["net_ret"].fillna(0)).cumprod()
            equities.append(bt[["date", "asset", "fold", "chosen_threshold", "net_ret", "equity", "regime"]])
        start, fold = start + STEP_DAYS, fold + 1
    return (pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(), pd.DataFrame(chosen),
            pd.DataFrame(results), pd.concat(equities, ignore_index=True) if equities else pd.DataFrame())


def summarize_oos(oos):
    if oos.empty:
        return pd.DataFrame()
    return pd.DataFrame([{"folds": len(oos), "mean_oos_sharpe": oos["sharpe"].mean(), "median_oos_sharpe": oos["sharpe"].median(),
                          "mean_oos_ret": oos["ret"].mean(), "median_oos_ret": oos["ret"].median(), "mean_oos_mdd": oos["mdd"].mean(),
                          "mean_oos_turnover": oos["turnover"].mean(), "mean_skip_rate": oos["skip_rate"].mean(),
                          "positive_folds": int((oos["sharpe"] > 0).sum())}])


def run_engine(input_csv, out_prefix="output/engine"):
    df = pd.read_csv(input_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["asset"].isin(ASSETS)]
    if df.empty:
        raise ValueError(f"No rows found for configured assets: {ASSETS}")
    tables = walk_forward_engine(df)
    summary = summarize_oos(tables[2])
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    names = ["threshold_grid", "chosen_thresholds", "oos_results", "oos_equity"]
    for name, table in zip(names, tables):
        table.to_csv(f"{prefix}_{name}.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Swift-Quant walk-forward backtest.")
    parser.add_argument("input_csv", help="CSV with date, asset, OHLC, and volume columns")
    parser.add_argument("--out-prefix", default="output/crypto_wf", help="Output path prefix")
    args = parser.parse_args()
    summary = run_engine(args.input_csv, args.out_prefix)
    print("Backtest complete.")
    print(summary.to_string(index=False) if not summary.empty else "No complete walk-forward folds were available.")


if __name__ == "__main__":
    main()
