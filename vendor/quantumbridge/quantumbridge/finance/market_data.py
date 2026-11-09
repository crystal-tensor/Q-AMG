# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Verified, offline real-stock snapshots; never substitute synthetic prices."""
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from importlib.resources import files
from datetime import date

import numpy as np


@dataclass(frozen=True)
class StockHistory:
    dates: tuple
    symbols: tuple
    prices: np.ndarray
    provenance: dict

    def returns(self):
        return self.prices[1:] / self.prices[:-1] - 1


def load_stock_history(start="2015-01-01", end="2015-12-31",
                       symbols=("MSFT", "IBM", "SBUX", "AAPL")):
    """Load aligned source prices; dates are inclusive, no imputation is used."""
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("start must not be after end")
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("unique nonempty symbols are required")
    folder = files("quantumbridge.finance").joinpath("data")
    provenance = json.loads(folder.joinpath("provenance.json").read_text())
    raw = folder.joinpath("plotly_stockdata.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["sha256"]:
        raise ValueError("Stock source checksum mismatch; refusing unverified prices")
    if set(symbols) - set(provenance["symbols"]):
        raise ValueError("symbol is not in the frozen real-stock dataset")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    dates = [row["Date"] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Source dates must be strictly ordered and unique")
    selected = [row for row in rows if start <= row["Date"] <= end]
    if len(selected) < 3:
        raise ValueError("At least three observed dates are required; no synthetic fallback")
    prices = np.array([[float(row[s]) for s in symbols] for row in selected])
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Stock prices must be finite and positive")
    prices.setflags(write=False)
    return StockHistory(tuple(row["Date"] for row in selected), symbols, prices, provenance)
