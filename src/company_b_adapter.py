"""Company B invoice-line adapter. Inference only: never opens seed_labels.csv.

Inputs: invoice_lines.csv, fx_rates_usd.csv, taxonomy_B.csv.
Outputs: final_spend.csv, processed classification audit, pipeline_summary.json.

Conservative policy: keep credits/negative values; only exclude uniquely
matched exact reversal pairs, invalid amounts and unconvertible currencies.
Document-type meanings are unknown, so no speculative type exclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import (CONFIG, apply_item_consistency, build_similarity,
                                load_taxonomy, normalize_text)

REQUIRED = {"line_id", "source_system", "invoice_number", "posting_date",
            "line_amount", "currency", "vendor_id", "material_number",
            "line_description", "gl_account_description", "document_type"}
OUTPUT = ["record_id", "record_source", "include", "exclusion_reason",
          "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4",
          "confidence", "classification_review", "classification_method"]


def strings(frame, col):
    return frame[col].astype("string").fillna("").str.strip()


def hash_key(prefix, text):
    return prefix + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16].upper()


def reversal_detection(df):
    """Exclude only one-negative/one-positive pairs with identical strong keys.

    This is deliberately narrower than general reversal matching. A reversal
    pair is recognized only when invoice number, source, vendor, material,
    invoice line and posting date are all present and identical.
    """
    required = ["source_system", "invoice_number", "vendor_id", "material_number",
                "invoice_line", "posting_date", "currency"]
    keys = [strings(df, col) if col in df else pd.Series("", index=df.index)
            for col in required]
    complete = pd.concat(keys, axis=1).ne("").all(axis=1)
    amt = df["amount_local"]
    candidates = df.loc[complete & amt.notna() & amt.ne(0)].copy()
    if candidates.empty:
        return set(), set()
    candidates["_amount_abs_cents"] = (candidates["amount_local"].abs() * 100).round().astype("int64")
    candidates["_sign"] = np.where(candidates["amount_local"] > 0, 1, -1)
    group_cols = required + ["_amount_abs_cents"]
    matched = set()
    ambiguous = set()
    for _, group in candidates.groupby(group_cols, dropna=False, sort=False):
        positive = group.index[group["_sign"].eq(1)].tolist()
        negative = group.index[group["_sign"].eq(-1)].tolist()
        if positive and negative:
            if len(positive) == len(negative) == 1:
                matched.update((positive[0], negative[0]))
            else:
                ambiguous.update(group.index.tolist())
    return matched, ambiguous


def run(args):
    for path in (args.lines, args.fx, args.taxonomy):
        if not path.is_file():
            raise ValueError(f"Input not found: {path}")
    df = pd.read_csv(args.lines, dtype="string", keep_default_na=False, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Company B input missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Company B input has no rows")
    ids = strings(df, "line_id")
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Company B line_id must be nonblank and unique")
    df["record_id"] = "ERP:" + ids
    df["record_source"] = "ERP"
    df["currency"] = strings(df, "currency").str.upper()
    df["posting_month"] = pd.to_datetime(strings(df, "posting_date"), errors="coerce").dt.strftime("%Y-%m")
    if df["posting_month"].isna().any():
        raise ValueError(f"Unparseable posting_date rows: {int(df['posting_month'].isna().sum())}")
    raw_amount = strings(df, "line_amount").str.replace(",", "", regex=False)
    df["amount_local"] = pd.to_numeric(raw_amount, errors="coerce")
    df["invalid_amount"] = df["amount_local"].isna() | ~np.isfinite(df["amount_local"].fillna(0))

    fx = pd.read_csv(args.fx, dtype="string", keep_default_na=False)
    if not {"currency", "month", "units_per_usd"}.issubset(fx.columns):
        raise ValueError("FX file needs currency, month, units_per_usd")
    fx["currency"] = strings(fx, "currency").str.upper()
    fx["month"] = strings(fx, "month")
    fx["units_per_usd"] = pd.to_numeric(strings(fx, "units_per_usd").str.replace(",", "", regex=False), errors="coerce")
    if fx.duplicated(["currency", "month"]).any():
        raise ValueError("FX has duplicate currency-month rates")
    if fx["units_per_usd"].isna().any() or (~np.isfinite(fx["units_per_usd"])).any() or fx["units_per_usd"].le(0).any():
        raise ValueError("FX rates must be finite positive numbers")
    df = df.merge(fx.rename(columns={"month": "posting_month", "units_per_usd": "fx_units_per_usd"}),
                  on=["currency", "posting_month"], how="left", validate="many_to_one", sort=False)
    df.loc[df["currency"].eq("USD"), "fx_units_per_usd"] = 1.0
    df["missing_fx"] = df["fx_units_per_usd"].isna() | df["fx_units_per_usd"].le(0)
    df["amount_usd"] = df["amount_local"] / df["fx_units_per_usd"]

    matched, ambiguous = reversal_detection(df)
    df["include"] = True
    df["exclusion_reason"] = ""
    df["reversal_review"] = df.index.isin(ambiguous)
    df.loc[df.index.isin(matched), ["include", "exclusion_reason"]] = [False, "REVERSAL_PAIR"]
    df.loc[df["missing_fx"], ["include", "exclusion_reason"]] = [False, "MISSING_FX"]
    df.loc[df["invalid_amount"], ["include", "exclusion_reason"]] = [False, "INVALID_AMOUNT"]
    df["spend"] = df["amount_usd"].where(df["include"], 0.0).fillna(0.0)

    source = strings(df, "source_system")
    vendor = strings(df, "vendor_id")
    df["supplier_key"] = [hash_key("SUP_B_", f"{s}|{v}" if v else f"{s}|UNKNOWN|{i}")
                          for i, (s, v) in enumerate(zip(source, vendor))]
    material = strings(df, "material_number")
    description = strings(df, "line_description")
    gl = strings(df, "gl_account_description")
    # Do not classify on opaque material IDs; use human-readable line and GL text.
    df["classification_text_raw"] = (description + " " + gl).str.strip()
    df["classification_text"] = df["classification_text_raw"].map(normalize_text)
    df["item_key"] = [hash_key("ITEM_B_", f"{s}|MAT|{m}" if m else
                               f"{s}|VENDOR|{v}|DESC|{d}" if d else f"{s}|ROW|{i}")
                      for i, (s, m, v, d) in enumerate(zip(source, material, vendor, df["classification_text"]))]

    taxonomy = load_taxonomy(args.taxonomy)
    unique_texts = df["classification_text"].drop_duplicates().tolist()
    predictions = build_similarity(taxonomy, unique_texts)
    df["_original_order"] = np.arange(len(df))
    df = df.merge(predictions.drop(columns=["classification_text_raw"], errors="ignore"),
                  on="classification_text", how="left", validate="many_to_one", sort=False)
    df = df.sort_values("_original_order", kind="stable").reset_index(drop=True)
    df = apply_item_consistency(df, taxonomy)
    path_lookup = taxonomy.set_index("taxonomy_path_id")[["l1", "l2", "l3", "l4"]]
    df = df.join(path_lookup, on="taxonomy_path_id", validate="many_to_one")
    df["classification_review"] = (df["classification_review"].astype(bool) |
                                   df["reversal_review"] | df["missing_fx"] |
                                   df["invalid_amount"] | df["amount_local"].eq(0))
    if df[OUTPUT].isna().any().any():
        raise AssertionError("Missing final output field")
    valid = set(map(tuple, taxonomy[["l1", "l2", "l3", "l4"]].itertuples(index=False, name=None)))
    actual = set(map(tuple, df[["l1", "l2", "l3", "l4"]].itertuples(index=False, name=None)))
    if not actual.issubset(valid):
        raise AssertionError("Predicted path not in taxonomy B")
    if len(df) != len(ids) or not df["record_id"].equals("ERP:" + ids.reset_index(drop=True)):
        raise AssertionError("Input records changed or reordered")
    if df["supplier_key"].eq("").any() or df["item_key"].eq("").any():
        raise AssertionError("Missing entity keys")
    if df.loc[~df["include"], "spend"].abs().gt(1e-7).any():
        raise AssertionError("Excluded spend is not zero")
    if not df["confidence"].between(0, 1).all():
        raise AssertionError("Invalid confidence")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    df[OUTPUT].to_csv(args.output_dir.parent.parent / "final_spend.csv", index=False)
    audit = ["record_id", "source_system", "document_type", "invoice_number", "invoice_line",
             "posting_date", "currency", "amount_local", "fx_units_per_usd", "amount_usd",
             "include", "exclusion_reason", "reversal_review", "supplier_key", "item_key",
             "classification_text_raw", "classification_text", "l1", "l2", "l3", "l4",
             "confidence", "classification_review", "classification_method", "spend"]
    df[[c for c in audit if c in df]].to_csv(args.output_dir / "company_b_audit.csv", index=False)
    summary = {
        "company": "B", "total_records": int(len(df)),
        "included_records": int(df["include"].sum()),
        "excluded_records": int((~df["include"]).sum()),
        "included_net_spend_usd": float(df.loc[df["include"], "spend"].sum()),
        "excluded_spend_usd": float(df.loc[~df["include"], "spend"].sum()),
        "reversal_pair_records": int(df["exclusion_reason"].eq("REVERSAL_PAIR").sum()),
        "ambiguous_reversal_review_records": int(df["reversal_review"].sum()),
        "invalid_amount_records": int(df["invalid_amount"].sum()),
        "missing_fx_records": int(df["missing_fx"].sum()),
        "review_flagged_records": int(df["classification_review"].sum()),
        "taxonomy_paths_used": int(df["taxonomy_path_id"].nunique()),
        "labels_used_for_inference": False,
        "currency": "USD",
        "policy_note": "No document-type exclusions without business mapping; unmatched credits retained. Reversal matching is conservative. Confidence is heuristic.",
    }
    (args.report_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VALIDATION PASSED: Company B row count, IDs, USD conversion, zero excluded spend, taxonomy paths")
    print(json.dumps(summary, indent=2))
    print("Final CSV:", args.output_dir.parent.parent / "final_spend.csv")


def main():
    p = argparse.ArgumentParser(description="Company B invoice-line processing (no seed labels)")
    p.add_argument("--lines", type=Path, required=True)
    p.add_argument("--fx", type=Path, required=True)
    p.add_argument("--taxonomy", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
