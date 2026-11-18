
from pathlib import Path
import json
import re

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND DATA LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

ERP_FILE = RAW_DATA_DIR / "erp_goods_receipts.csv"
INVOICE_FILE = RAW_DATA_DIR / "invoice_tool_export.csv"
LABELS_FILE = RAW_DATA_DIR / "labels_A.csv"
TAXONOMY_FILE = RAW_DATA_DIR / "taxonomy_A.csv"


def load_data():
    """Load all Company A input files."""

    erp = pd.read_csv(ERP_FILE)
    invoices = pd.read_csv(INVOICE_FILE)
    labels = pd.read_csv(LABELS_FILE)
    taxonomy = pd.read_csv(TAXONOMY_FILE)

    return erp, invoices, labels, taxonomy


# --------------------------------------------------
# 2. BASIC DATASET INFORMATION
# --------------------------------------------------

def basic_information(erp, invoices, labels, taxonomy):

    datasets = {
        "ERP": erp,
        "Invoices": invoices,
        "Labels": labels,
        "Taxonomy": taxonomy,
    }

    print("\n" + "=" * 70)
    print("BASIC DATASET INFORMATION")
    print("=" * 70)

    for name, df in datasets.items():
        print(f"\n{name}")
        print("-" * 40)
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])
        print("Column names:")
        print(df.columns.tolist())


# --------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# --------------------------------------------------

def missing_value_analysis(erp, invoices, labels, taxonomy):

    datasets = {
        "ERP": erp,
        "Invoices": invoices,
        "Labels": labels,
        "Taxonomy": taxonomy,
    }

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    for name, df in datasets.items():

        print(f"\n{name}")
        print("-" * 40)

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("No missing values.")
        else:
            missing_percentage = (
                missing / len(df) * 100
            ).round(2)

            result = pd.DataFrame({
                "missing_count": missing,
                "missing_percentage": missing_percentage,
            })

            print(result.sort_values(
                by="missing_count",
                ascending=False
            ))


# --------------------------------------------------
# 4. DUPLICATE ANALYSIS
# --------------------------------------------------

def duplicate_analysis(erp, invoices, labels, taxonomy):

    datasets = {
        "ERP": erp,
        "Invoices": invoices,
        "Labels": labels,
        "Taxonomy": taxonomy,
    }

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    for name, df in datasets.items():

        print(f"\n{name}")
        print("-" * 40)

        print("Exact duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 5. IDENTIFIER ANALYSIS
# --------------------------------------------------

def identifier_analysis(erp, invoices, labels):

    print("\n" + "=" * 70)
    print("IDENTIFIER ANALYSIS")
    print("=" * 70)

    erp_columns = [
        "line_id",
        "vendor_id",
        "supplier_name",
        "po_number",
        "receipt_document",
        "material_number",
    ]

    print("\nERP identifiers")
    print("-" * 40)

    for col in erp_columns:
        if col in erp.columns:
            print(
                f"{col}: "
                f"{erp[col].nunique(dropna=True)} unique values"
            )

    label_columns = [
        "record_id",
        "supplier_key",
        "item_key",
    ]

    print("\nLabel identifiers")
    print("-" * 40)

    for col in label_columns:
        if col in labels.columns:
            print(
                f"{col}: "
                f"{labels[col].nunique(dropna=True)} unique values"
            )

    invoice_columns = [
        "invoice_number",
        "supplier_name_as_captured",
        "po_reference",
    ]

    print("\nInvoice identifiers")
    print("-" * 40)

    for col in invoice_columns:
        if col in invoices.columns:
            print(
                f"{col}: "
                f"{invoices[col].nunique(dropna=True)} unique values"
            )


# --------------------------------------------------
# 6. CATEGORY AND TAXONOMY ANALYSIS
# --------------------------------------------------

def category_analysis(labels, taxonomy):

    print("\n" + "=" * 70)
    print("CATEGORY AND TAXONOMY ANALYSIS")
    print("=" * 70)

    category_columns = [
        "category_l1",
        "category_l2",
        "category_l3",
    ]

    print("\nTaxonomy levels")
    print("-" * 40)

    for col in category_columns:
        if col in taxonomy.columns:
            print(
                f"{col}: "
                f"{taxonomy[col].nunique(dropna=True)} unique values"
            )

    print("\nLabel category distribution")
    print("-" * 40)

    print(
        labels["category_l1"]
        .value_counts(dropna=False)
    )

    print("\nChecking label categories against taxonomy...")

    label_paths = labels[
        ["category_l1", "category_l2", "category_l3"]
    ].drop_duplicates()

    taxonomy_paths = taxonomy[
        ["category_l1", "category_l2", "category_l3"]
    ].drop_duplicates()

    comparison = label_paths.merge(
        taxonomy_paths,
        on=["category_l1", "category_l2", "category_l3"],
        how="left",
        indicator=True,
    )

    invalid_paths = comparison[
        comparison["_merge"] == "left_only"
    ]

    print("Unique category paths in labels:", len(label_paths))
    print("Invalid category paths:", len(invalid_paths))

    if not invalid_paths.empty:
        print("\nInvalid paths:")
        print(invalid_paths)


# --------------------------------------------------
# 7. INVOICE STATUS ANALYSIS
# --------------------------------------------------

def invoice_status_analysis(invoices):

    print("\n" + "=" * 70)
    print("INVOICE STATUS ANALYSIS")
    print("=" * 70)

    if "status" not in invoices.columns:
        print("Status column not found.")
        return

    print("\nInvoice status counts:")
    print(invoices["status"].value_counts(dropna=False))

    print("\nStatus percentages:")
    print(
        (invoices["status"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nStatuses relevant to the assignment:")
    print("- Approved and Paid: eligible for inclusion")
    print("- Cancelled: CANCELLED")
    print("- On Hold and Rejected: NOT_APPROVED")


# --------------------------------------------------
# 8. PURCHASE ORDER NORMALIZATION AND MATCHING
# --------------------------------------------------

def normalize_po(value):

    """Extract a 10-digit purchase order number."""

    if pd.isna(value):
        return None

    text = str(value).strip()

    # Remove common spreadsheet-style decimal suffix.
    text = re.sub(r"\.0$", "", text)

    # Extract a 10-digit PO number.
    match = re.search(r"\b\d{10}\b", text)

    if match:
        return match.group(0)

    return None


def po_matching_analysis(erp, invoices):

    print("\n" + "=" * 70)
    print("PURCHASE ORDER MATCHING ANALYSIS")
    print("=" * 70)

    if "po_number" not in erp.columns:
        print("ERP PO number column not found.")
        return

    if "po_reference" not in invoices.columns:
        print("Invoice PO reference column not found.")
        return

    erp_copy = erp.copy()
    invoice_copy = invoices.copy()

    erp_copy["normalized_po"] = (
        erp_copy["po_number"].apply(normalize_po)
    )

    invoice_copy["normalized_po"] = (
        invoice_copy["po_reference"].apply(normalize_po)
    )

    erp_pos = set(
        erp_copy["normalized_po"].dropna().unique()
    )

    invoice_pos = set(
        invoice_copy["normalized_po"].dropna().unique()
    )

    matching_pos = erp_pos.intersection(invoice_pos)

    invoice_copy["po_found_in_erp"] = (
        invoice_copy["normalized_po"].isin(erp_pos)
    )

    print("Unique ERP PO numbers:", len(erp_pos))
    print("Unique invoice PO numbers:", len(invoice_pos))
    print("Matching unique PO numbers:", len(matching_pos))

    print(
        "Invoice rows with a PO number found in ERP:",
        invoice_copy["po_found_in_erp"].sum()
    )

    print(
        "Invoice rows with a PO reference:",
        invoice_copy["normalized_po"].notna().sum()
    )

    print(
        "\nImportant: A matching PO is only a candidate match. "
        "It does not prove that the invoice is a duplicate."
    )


# --------------------------------------------------
# 9. TRANSACTION FIELD ANALYSIS
# --------------------------------------------------

def transaction_field_analysis(erp, invoices):

    print("\n" + "=" * 70)
    print("TRANSACTION FIELD ANALYSIS")
    print("=" * 70)

    # ERP date examples
    if "posting_date" in erp.columns:

        print("\nERP posting date examples:")
        print(erp["posting_date"].dropna().head(10).tolist())

    # Invoice date examples
    if "invoice_date" in invoices.columns:

        print("\nInvoice date examples:")
        print(invoices["invoice_date"].dropna().head(10).tolist())

    # Invoice gross amount
    if "gross_amount" in invoices.columns:

        print("\nInvoice gross amount analysis:")
        print("Data type:", invoices["gross_amount"].dtype)

        gross_numeric = pd.to_numeric(
            invoices["gross_amount"]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )

        print("Successfully parsed:", gross_numeric.notna().sum())
        print("Failed or missing:", gross_numeric.isna().sum())

        print("\nGross amount statistics:")
        print(gross_numeric.describe())

        print(
            "Zero or negative gross amounts:",
            (gross_numeric <= 0).sum()
        )

    # VAT rate distribution
    if "vat_rate" in invoices.columns:

        print("\nVAT rate examples:")
        print(invoices["vat_rate"].value_counts(dropna=False).head(20))

    # ERP spend value
    if "value_zar" in erp.columns:

        print("\nERP value_zar analysis:")

        erp_value = pd.to_numeric(
            erp["value_zar"],
            errors="coerce",
        )

        print(erp_value.describe())

        print(
            "Zero ERP values:",
            (erp_value == 0).sum()
        )

        print(
            "Negative ERP values:",
            (erp_value < 0).sum()
        )


# --------------------------------------------------
# 10. LABEL-TO-ERP ALIGNMENT
# --------------------------------------------------

def label_alignment_analysis(erp, labels):

    print("\n" + "=" * 70)
    print("LABEL-TO-ERP ALIGNMENT ANALYSIS")
    print("=" * 70)

    if "line_id" not in erp.columns:
        print("ERP line_id column not found.")
        return

    if "record_id" not in labels.columns:
        print("Label record_id column not found.")
        return

    erp_ids = (
        erp["line_id"]
        .astype(str)
        .str.strip()
    )

    label_ids = (
        labels["record_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"^ERP:", "", regex=True)
    )

    erp_id_set = set(erp_ids)
    label_id_set = set(label_ids)

    matched = label_id_set.intersection(erp_id_set)
    labels_not_in_erp = label_id_set - erp_id_set
    erp_without_labels = erp_id_set - label_id_set

    print("Unique ERP line IDs:", len(erp_id_set))
    print("Unique label record IDs:", len(label_id_set))
    print("Matched IDs:", len(matched))
    print("Labels without matching ERP record:", len(labels_not_in_erp))
    print("ERP records without matching label:", len(erp_without_labels))

    if labels_not_in_erp:
        print("\nExamples of label IDs not found in ERP:")
        print(list(labels_not_in_erp)[:10])

    if erp_without_labels:
        print("\nExamples of ERP IDs without labels:")
        print(list(erp_without_labels)[:10])


# --------------------------------------------------
# 11. SPEND TYPE ANALYSIS
# --------------------------------------------------

def spend_type_analysis(erp):

    print("\n" + "=" * 70)
    print("SPEND TYPE ANALYSIS")
    print("=" * 70)

    if "spend_type" not in erp.columns:
        print("Spend type column not found.")
        return

    print("\nSpend type record counts:")
    print(erp["spend_type"].value_counts(dropna=False))

    if "value_zar" in erp.columns:

        erp_copy = erp.copy()

        erp_copy["value_zar_numeric"] = pd.to_numeric(
            erp_copy["value_zar"],
            errors="coerce",
        )

        print("\nSpend type value statistics:")

        print(
            erp_copy.groupby("spend_type", dropna=False)[
                "value_zar_numeric"
            ].agg(
                record_count="count",
                total_spend="sum",
                average_spend="mean",
                median_spend="median",
            )
        )

    if "supplier_name" in erp.columns:

        print("\nUnique suppliers by spend type:")

        print(
            erp.groupby("spend_type", dropna=False)[
                "supplier_name"
            ].nunique()
        )

    description_columns = [
        "material_short_text",
        "service_short_text",
        "material_number",
    ]

    available_columns = [
        col for col in description_columns
        if col in erp.columns
    ]

    if available_columns:

        print("\nMissing description/identifier fields by spend type:")

        print(
            erp.groupby("spend_type", dropna=False)[
                available_columns
            ].apply(lambda group: group.isna().sum())
        )


# --------------------------------------------------
# 12. DATA QUALITY SUMMARY
# --------------------------------------------------

def data_quality_summary(erp, invoices, labels, taxonomy):

    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)

    summary = {
        "erp_rows": int(len(erp)),
        "invoice_rows": int(len(invoices)),
        "label_rows": int(len(labels)),
        "taxonomy_rows": int(len(taxonomy)),
        "erp_missing_cells": int(erp.isna().sum().sum()),
        "invoice_missing_cells": int(invoices.isna().sum().sum()),
        "label_missing_cells": int(labels.isna().sum().sum()),
        "taxonomy_missing_cells": int(taxonomy.isna().sum().sum()),
        "unique_erp_supplier_names": int(
            erp["supplier_name"].nunique()
        ) if "supplier_name" in erp.columns else None,
        "unique_invoice_supplier_names": int(
            invoices["supplier_name_as_captured"].nunique()
        ) if "supplier_name_as_captured" in invoices.columns else None,
        "unique_supplier_keys": int(
            labels["supplier_key"].nunique()
        ) if "supplier_key" in labels.columns else None,
        "unique_item_keys": int(
            labels["item_key"].nunique()
        ) if "item_key" in labels.columns else None,
        "taxonomy_l1_count": int(
            taxonomy["category_l1"].nunique()
        ) if "category_l1" in taxonomy.columns else None,
        "taxonomy_l2_count": int(
            taxonomy["category_l2"].nunique()
        ) if "category_l2" in taxonomy.columns else None,
        "taxonomy_l3_count": int(
            taxonomy["category_l3"].nunique()
        ) if "category_l3" in taxonomy.columns else None,
    }

    print(json.dumps(summary, indent=4))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORTS_DIR / "data_quality_summary.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nSummary saved to:", output_file)


# --------------------------------------------------
# 13. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nCOMPANY A - DATA PROFILING STARTED")

    erp, invoices, labels, taxonomy = load_data()

    basic_information(
        erp, invoices, labels, taxonomy
    )

    missing_value_analysis(
        erp, invoices, labels, taxonomy
    )

    duplicate_analysis(
        erp, invoices, labels, taxonomy
    )

    identifier_analysis(
        erp, invoices, labels
    )

    category_analysis(
        labels, taxonomy
    )

    invoice_status_analysis(
        invoices
    )

    po_matching_analysis(
        erp, invoices
    )

    transaction_field_analysis(
        erp, invoices
    )

    label_alignment_analysis(
        erp, labels
    )

    spend_type_analysis(
        erp
    )

    data_quality_summary(
        erp, invoices, labels, taxonomy
    )

    print("\nCOMPANY A - DATA PROFILING COMPLETED")


if __name__ == "__main__":
    main()
