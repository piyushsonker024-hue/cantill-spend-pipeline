
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


def main():

    invoices = pd.read_csv(
        DATA / "invoices_eligible.csv",
        low_memory=False,
        dtype={
            "po_number_normalized": "string",
            "duplicate_erp_line_id": "string",
        }
    )

    duplicates = invoices.loc[
        invoices["exclusion_reason"] == "DUPLICATE"
    ].copy()

    review = invoices.loc[
        invoices["duplicate_review"].fillna(False).astype(bool)
    ].copy()

    print("=" * 65)
    print("DUPLICATE MATCH AUDIT")
    print("=" * 65)

    print(f"\nExcluded duplicates: {len(duplicates):,}")
    print(f"Flagged for review: {len(review):,}")

    print("\nDUPLICATE TEXT SIMILARITY")
    print("-" * 55)

    print(
        duplicates["duplicate_text_similarity"]
        .describe()
        .to_string()
    )

    print("\nDUPLICATE AMOUNT DIFFERENCES")
    print("-" * 55)

    print(
        duplicates["duplicate_amount_difference"]
        .describe()
        .to_string()
    )

    print("\nERP RECORDS MATCHED BY MULTIPLE INVOICES")
    print("-" * 55)

    match_counts = (
        duplicates["duplicate_erp_line_id"]
        .dropna()
        .value_counts()
    )

    repeated = match_counts[match_counts > 1]

    print(
        "ERP records with multiple matched invoices:",
        f"{len(repeated):,}"
    )

    print(
        "Invoices involved in repeated ERP matches:",
        f"{int(repeated.sum()):,}"
    )

    print("\nMOST FREQUENT REPEATED MATCHES")

    print(
        repeated.head(15).to_string()
    )

    print("\nLOWEST-SIMILARITY EXCLUDED DUPLICATES")
    print("-" * 55)

    columns = [
        "tool_record_id",
        "po_number_normalized",
        "supplier_name_normalized",
        "invoice_net_spend",
        "line_text",
        "duplicate_erp_line_id",
        "duplicate_text_similarity",
        "duplicate_amount_difference",
    ]

    print(
        duplicates
        .sort_values("duplicate_text_similarity")
        [columns]
        .head(15)
        .to_string(index=False)
    )

    print("\nREVIEW CANDIDATE TYPES")
    print("-" * 55)

    print(
        review["duplicate_match_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
