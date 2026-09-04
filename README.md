# Swift-Quant

A walk-forward crypto backtesting engine for XRP, XLM, XDC, and PI.

## Run

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Provide a daily market-data CSV with these columns:

```text
date,asset,open,high,low,close,volume
```

Then launch the engine:

```powershell
python swift_quant.py your_data.csv --out-prefix output/crypto_wf
```

The engine writes threshold-selection, out-of-sample performance, equity, and summary CSV files under `output/`.
