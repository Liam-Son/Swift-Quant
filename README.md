# Swift-Quant

A walk-forward crypto spot/perpetual backtesting engine for XRP, XLM, XDC, and PI.

The expanded engine includes input validation, tradable-only ranking, concentration
limits, funding, trading costs, portfolio-level returns, maintenance margin,
partial liquidation, walk-forward selection, reporting, and embedded tests.

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
python crypto_wf_engine.py --input your_data.csv --out-prefix crypto_wf
```

The engine writes threshold-selection, out-of-sample performance, equity, and summary CSV files under `output/`.

Run the embedded test suite with:

```powershell
python -m pytest crypto_wf_engine.py -q
```

The earlier `swift_quant.py` prototype remains available for comparison.
