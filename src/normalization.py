
from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ERP_INPUT_FILE = RAW_DATA_DIR / "erp_goods_receipts.csv"
INVOICE_INPUT_FILE = RAW_DATA_DIR / "invoice_tool_export.csv"

ERP_OUTPUT_FILE = PROCESSED_DATA_DIR / "erp_normalized.csv"
INVOICE_OUTPUT_FILE = PROCESSED_DATA_DIR / "invoices_normalized.csv"

DEFAULT_VAT_RATE = 0.15


# ============================================================
# 2. SUPPLIER NAME NORMALIZATION
# ============================================================

def normalize_supplier_name(value):
    """
    Standardize supplier names for later matching.

    Examples:
        ' ABC (Pty) Ltd. ' -> 'ABC PTY LTD'
        'Acme   Supplies' -> 'ACME SUPPLIES'

    This is a conservative normalization:
    it does not assume two different names are the same supplier.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    # Replace punctuation with spaces.
    text = re.sub(r"[^A-Z0-9]+", " ", text)

    # Collapse repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. PURCHASE ORDER NORMALIZATION
# ============================================================

def normalize_po(value):
    """
    Extract a 10-digit purchase order number.

    Examples:
        '4510085877'       -> '4510085877'
        '4510085877/10.0'  -> '4510085877'
        'PO 4510085877'    -> '4510085877'

    Returns an empty string if no 10-digit PO is found.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    match = re.search(r"(?<!\d)\d{10}(?!\d)", text)

    if match:
        return match.group(0)

    return ""


# ============================================================
# 4. AMOUNT NORMALIZATION
# ============================================================

def normalize_amount(value):
    """
    Convert amount values into numeric floats.

    Handles:
        1250
        '1,250.50'
        ' 1250.50 '
        negative values

    Missing or invalid values become NaN.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if not text:
        return np.nan

    text = text.replace(",", "").replace(" ", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


# ============================================================
# 5. VAT NORMALIZATION
# ============================================================

def normalize_vat_rate(value, default_rate=DEFAULT_VAT_RATE):
    """
    Convert different VAT representations into decimal rates.

    Examples:
        15       -> 0.15
        0.15     -> 0.15
        '15%'    -> 0.15
        0        -> 0.00
        missing  -> 0.15 (assignment default)
    """

    if pd.isna(value):
        return default_rate

    text = str(value).strip()

    if not text:
        return default_rate

    text = text.replace("%", "").replace(",", "")

    try:
        rate = float(text)
    except ValueError:
        return default_rate

    if not np.isfinite(rate) or rate < 0:
        return default_rate

    if rate > 1:
        rate = rate / 100.0

    return rate


# ============================================================
# 6. DATE NORMALIZATION - UPDATED
# ============================================================

def normalize_date_column(series, dayfirst=True):
    """
    Parse dates while protecting ISO-formatted dates.

    IMPORTANT:
    ISO dates must always be interpreted as YYYY-MM-DD.

    Examples:
        '2021-07-01' -> 2021-07-01
        '27-Mar-24'  -> 2024-03-27

    Slash-formatted dates are provisionally parsed with
    dayfirst=True. Their ambiguity is reported separately.

    Missing or invalid dates become NaT.
    """

    text = series.astype("string").str.strip()

    text = text.replace("", pd.NA)

    # Identify ISO dates such as:
    # 2021-07-01
    # 2021-07-01 14:30:00
    # 2021-07-01T14:30:00
    iso_mask = text.str.match(
        r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$",
        na=False
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # ISO dates: never use dayfirst=True.
    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(
            text.loc[iso_mask],
            format="mixed",
            dayfirst=False,
            errors="coerce"
        )

    # Non-ISO dates: use the provisional day-first convention.
    non_iso_mask = ~iso_mask & text.notna()

    if non_iso_mask.any():
        result.loc[non_iso_mask] = pd.to_datetime(
            text.loc[non_iso_mask],
            format="mixed",
            dayfirst=dayfirst,
            errors="coerce"
        )

    return result


# ============================================================
# 7. ERP NORMALIZATION
# ============================================================

def normalize_erp(erp):
    """
    Normalize ERP goods receipt data.

    Assignment rule:
        ERP value_zar is already NET spend.

    Therefore, no VAT deduction is applied to ERP amounts.
    """

    df = erp.copy()

    # Keep original columns unchanged and add normalized columns.
    df["supplier_name_normalized"] = (
    df["supplier_name"].apply(normalize_supplier_name)
    )

    df["po_number_normalized"] = (
        df["po_number"].apply(normalize_po)
    )

    df["erp_net_spend"] = (
        df["value_zar"].apply(normalize_amount)
    )

    df["posting_date_normalized"] = normalize_date_column(
        df["posting_date"],
        dayfirst=True
    )

    return df


# ============================================================
# 8. INVOICE NORMALIZATION
# ============================================================

def normalize_invoices(invoices):
    """
    Normalize invoice-tool records.

    Assignment rule:
        Invoice gross_amount includes VAT.

    Net spend = gross_amount / (1 + VAT rate)

    Missing VAT defaults to 15%.
    """

    df = invoices.copy()

    df["supplier_name_normalized"] = (
    df["supplier_name_as_captured"].apply(normalize_supplier_name)
    )

    df["po_number_normalized"] = (
        df["po_reference"].apply(normalize_po)
    )

    df["gross_amount_numeric"] = (
        df["gross_amount"].apply(normalize_amount)
    )

    df["vat_rate_normalized"] = (
        df["vat_rate"].apply(normalize_vat_rate)
    )

    df["invoice_net_spend"] = (
        df["gross_amount_numeric"]
        / (1.0 + df["vat_rate_normalized"])
    )

    df["invoice_date_normalized"] = normalize_date_column(
        df["invoice_date"],
        dayfirst=True
    )

    return df


# ============================================================
# 9. DATE INTERPRETATION ANALYSIS - UPDATED
# ============================================================

def date_interpretation_analysis(erp, invoices):
    """
    Diagnose date-format ambiguity.

    ISO dates are no longer compared under day-first versus
    month-first rules, because their YYYY-MM-DD interpretation
    is fixed.

    Only slash-formatted dates are checked for ambiguity.
    """

    print("\n" + "=" * 65)
    print("DATE INTERPRETATION ANALYSIS")
    print("=" * 65)

    datasets = [
        ("ERP posting_date", erp["posting_date"]),
        ("Invoice invoice_date", invoices["invoice_date"]),
    ]

    for name, series in datasets:

        print(f"\n{name}")
        print("-" * 55)

        raw = series.astype("string").str.strip()

        non_missing = raw.notna() & raw.ne("")

        parsed_corrected = normalize_date_column(
            series,
            dayfirst=True
        )

        print(f"Non-missing raw dates: {int(non_missing.sum()):,}")
        print(
            "Successfully parsed with corrected parser: "
            f"{int(parsed_corrected.notna().sum()):,}"
        )

        iso_mask = raw.str.match(
            r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$",
            na=False
        )

        print(f"ISO-formatted dates: {int(iso_mask.sum()):,}")

        # Check only slash dates with a four-digit year.
        slash_mask = raw.str.match(
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            na=False
        )

        slash_dates = raw.loc[slash_mask]

        print(f"Slash-formatted dates: {len(slash_dates):,}")

        if slash_dates.empty:
            print("No slash-date ambiguity to analyze.")
            continue

        parsed_dayfirst = pd.to_datetime(
            slash_dates,
            format="%d/%m/%Y",
            errors="coerce"
        )

        parsed_monthfirst = pd.to_datetime(
            slash_dates,
            format="%m/%d/%Y",
            errors="coerce"
        )

        ambiguous_mask = (
            parsed_dayfirst.notna()
            & parsed_monthfirst.notna()
            & parsed_dayfirst.ne(parsed_monthfirst)
        )

        ambiguous_count = int(ambiguous_mask.sum())

        print(
            "Dates with two valid, different interpretations: "
            f"{ambiguous_count:,}"
        )

        print(
            "Note: Slash dates currently use day/month/year. "
            "Ambiguous values should not be treated as strong "
            "matching evidence until their format is confirmed."
        )

        if ambiguous_count > 0:

            examples = pd.DataFrame({
                "raw_date": slash_dates.loc[ambiguous_mask],
                "day_first": parsed_dayfirst.loc[ambiguous_mask],
                "month_first": parsed_monthfirst.loc[ambiguous_mask],
            })

            print("\nExamples of ambiguous slash dates:")
            print(
                examples.head(10).to_string(index=False)
            )


# ============================================================
# 10. NORMALIZATION SUMMARY
# ============================================================

def print_normalization_summary(erp, invoices):

    print("\n" + "=" * 65)
    print("NORMALIZATION SUMMARY")
    print("=" * 65)

    print(f"\nERP rows: {len(erp):,}")
    print(f"Invoice rows: {len(invoices):,}")

    print("\nERP NORMALIZATION")

    print(
        "Missing normalized supplier names:",
        int(erp["supplier_name_normalized"].eq("").sum())
    )

    print(
        "Missing normalized PO numbers:",
        int(erp["po_number_normalized"].eq("").sum())
    )

    print(
        "Missing ERP net spend:",
        int(erp["erp_net_spend"].isna().sum())
    )

    print(
        "Missing parsed ERP dates:",
        int(erp["posting_date_normalized"].isna().sum())
    )

    print("\nINVOICE NORMALIZATION")

    print(
        "Missing normalized supplier names:",
        int(invoices["supplier_name_normalized"].eq("").sum())
    )

    print(
        "Missing normalized PO numbers:",
        int(invoices["po_number_normalized"].eq("").sum())
    )

    print(
        "Missing numeric gross amounts:",
        int(invoices["gross_amount_numeric"].isna().sum())
    )

    print(
        "Missing normalized VAT rates:",
        int(invoices["vat_rate_normalized"].isna().sum())
    )

    print(
        "Missing calculated invoice net spend:",
        int(invoices["invoice_net_spend"].isna().sum())
    )

    print(
        "Missing parsed invoice dates:",
        int(invoices["invoice_date_normalized"].isna().sum())
    )

    print("\nNORMALIZED VAT DISTRIBUTION")

    print(
        invoices["vat_rate_normalized"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nSAMPLE ERP VALUES")

    erp_sample_columns = [
        "value_zar",
        "erp_net_spend",
        "posting_date",
        "posting_date_normalized",
    ]

    print(
        erp[erp_sample_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nSAMPLE INVOICE VALUES")

    invoice_sample_columns = [
        "gross_amount",
        "vat_rate",
        "gross_amount_numeric",
        "vat_rate_normalized",
        "invoice_net_spend",
        "invoice_date",
        "invoice_date_normalized",
    ]

    print(
        invoices[invoice_sample_columns]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 11. SAVE NORMALIZED DATA
# ============================================================

def save_normalized_data(erp, invoices):

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    erp.to_csv(
        ERP_OUTPUT_FILE,
        index=False
    )

    invoices.to_csv(
        INVOICE_OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("FILES SAVED SUCCESSFULLY")
    print("=" * 65)

    print(f"ERP output: {ERP_OUTPUT_FILE}")
    print(f"Invoice output: {INVOICE_OUTPUT_FILE}")


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COMPANY A - DATA NORMALIZATION")
    print("=" * 65)

    print("\nLoading raw datasets...")

    erp_raw = pd.read_csv(
        ERP_INPUT_FILE,
        low_memory=False
    )

    invoices_raw = pd.read_csv(
        INVOICE_INPUT_FILE,
        low_memory=False
    )

    print(f"ERP records loaded: {len(erp_raw):,}")
    print(f"Invoice records loaded: {len(invoices_raw):,}")

    print("\nNormalizing ERP records...")
    erp_normalized = normalize_erp(erp_raw)

    print("Normalizing invoice records...")
    invoices_normalized = normalize_invoices(invoices_raw)

    date_interpretation_analysis(
        erp_raw,
        invoices_raw
    )

    print_normalization_summary(
        erp_normalized,
        invoices_normalized
    )

    # Ensure normalization has not dropped or duplicated rows.
    assert len(erp_normalized) == len(erp_raw), (
        "ERP row count changed during normalization."
    )

    assert len(invoices_normalized) == len(invoices_raw), (
        "Invoice row count changed during normalization."
    )

    save_normalized_data(
        erp_normalized,
        invoices_normalized
    )

    print("\nNormalization completed successfully.")


if __name__ == "__main__":
    main()
