
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


def main():
    erp = pd.read_csv(
        DATA / "erp_normalized.csv",
        low_memory=False,
        dtype={"po_number_normalized": "string"},
    )

    invoices = pd.read_csv(
        DATA / "invoices_normalized.csv",
        low_memory=False,
        dtype={"po_number_normalized": "string"},
    )

    # Convert to strings BEFORE replacing missing values.
    for df in (erp, invoices):
        df["po_number_normalized"] = (
            df["po_number_normalized"]
            .astype("string")
            .fillna("")
            .str.strip()
        )

        df["supplier_name_normalized"] = (
            df["supplier_name_normalized"]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    print("=" * 65)
    print("ERP-INVOICE DUPLICATE DIAGNOSTIC")
    print("=" * 65)

    erp_po = set(
        erp.loc[
            erp["po_number_normalized"].ne(""),
            "po_number_normalized"
        ]
    )

    invoice_po = invoices["po_number_normalized"]

    po_matches = invoice_po.isin(erp_po)

    print(
        "\nInvoices with PO found in ERP:",
        f"{int(po_matches.sum()):,}"
    )

    # Check exact (PO, supplier) combinations.
    erp_pairs = set(
        zip(
            erp["po_number_normalized"],
            erp["supplier_name_normalized"]
        )
    )

    invoice_pairs = list(
        zip(
            invoices["po_number_normalized"],
            invoices["supplier_name_normalized"]
        )
    )

    pair_matches = [
        po != "" and (po, supplier) in erp_pairs
        for po, supplier in invoice_pairs
    ]

    print(
        "Invoices matching both PO and supplier:",
        f"{sum(pair_matches):,}"
    )

    print(
        "PO matches but supplier does not:",
        f"{int(po_matches.sum()) - sum(pair_matches):,}"
    )

    # Show representative supplier differences.
    erp_supplier_by_po = (
        erp.loc[
            erp["po_number_normalized"].ne(""),
            ["po_number_normalized", "supplier_name_normalized"]
        ]
        .drop_duplicates()
        .groupby("po_number_normalized")[
            "supplier_name_normalized"
        ]
        .first()
    )

    examples = invoices.loc[
        po_matches & ~pd.Series(
            pair_matches,
            index=invoices.index
        ),
        [
            "po_number_normalized",
            "supplier_name_normalized",
            "invoice_net_spend",
        ]
    ].copy()

    examples["erp_supplier"] = (
        examples["po_number_normalized"]
        .map(erp_supplier_by_po)
    )

    print("\nEXAMPLES OF SUPPLIER MISMATCHES")
    print("-" * 65)

    print(
        examples.head(15).to_string(index=False)
    )

    print("\nDIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
