"""scripts/adapter.py

Small adapter to load a CSV dataset and produce a standardized CSV with
columns the analysis notebook expects: 'text', 'summary', 'url', 'date', 'label'.

Usage examples:
  python scripts/adapter.py --input data/processed/train.csv --out data/processed/standard_train.csv
  python scripts/adapter.py --input data/processed/train.csv --sample 10000 --out data/processed/sample_train.csv

This script is intentionally conservative: it will warn about missing source
columns and fill them with nulls where appropriate.
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
import sys


DEFAULT_MAP = {
    "text": "text",
    "summary": "summary",
    "url": "url",
    "date": "date",
    "label": "label",
}


def load_and_map(in_path: Path, col_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(in_path)

    out = pd.DataFrame()
    for target, src in col_map.items():
        if src in df.columns:
            out[target] = df[src]
        else:
            print(f"Warning: source column '{src}' not found; filling '{target}' with nulls", file=sys.stderr)
            out[target] = pd.NA

    # Basic cleaning
    if "text" in out.columns:
        # strip whitespace and coerce to string
        out["text"] = out["text"].astype(str).str.strip()
    if "summary" in out.columns:
        out["summary"] = out["summary"].astype(str).str.strip()

    # Convert date if present
    if "date" in out.columns:
        try:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
        except Exception:
            pass

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to input CSV")
    p.add_argument("--out", "-o", required=True, help="Path to write standardized CSV")
    p.add_argument("--sample", "-s", type=int, default=0, help="If >0, randomly sample this many rows")
    p.add_argument("--map", "-m", nargs="*", help="Optional column mapping target:source (e.g. text:body summary:summary)")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 2

    # build mapping
    col_map = DEFAULT_MAP.copy()
    if args.map:
        for item in args.map:
            if ":" in item:
                tgt, src = item.split(":", 1)
                col_map[tgt] = src

    df = load_and_map(in_path, col_map)

    if args.sample and args.sample > 0:
        n = min(len(df), args.sample)
        df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # Drop rows without text
    if "text" in df.columns:
        df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote standardized CSV to {out_path} (rows: {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
