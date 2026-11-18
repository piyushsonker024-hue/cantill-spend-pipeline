
"""
Company A - Step 3: Eligibility and Deduplication

Run:
    python -m src.eligibility

Inputs:
    data/processed/erp_normalized.csv
    data/processed/invoices_normalized.csv

Outputs:
    data/processed/erp_eligible.csv
    data/processed/invoices_eligible.csv
    reports/eligibility_summary.json

Rules:
- Preserve every ERP and invoice record.
- ERP net spend is authoritative.
- Only Approved/Paid invoices are eligible.
- Cancelled invoices are excluded as CANCELLED.
- Other unapproved invoices are excluded as NOT_APPROVED.
- Clearly non-procurement invoices are excluded.
- ERP takes precedence over matching invoice-tool records.
- One ERP row can exclude at most one invoice.
- Automatic duplicates require:
    same PO,
    same normalized supplier,
    net amount difference <= R0.05,
    description similarity >= 0.85.
- Uncertain matches remain included and are flagged.
- Excluded spend is always zero.

No labels_A.csv data is used.
"""

from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

ERP_INPUT = PROCESSED_DIR / "erp_normalized.csv"
INVOICE_INPUT = PROCESSED_DIR / "invoices_normalized.csv"

ERP_OUTPUT = PROCESSED_DIR / "erp_eligible.csv"
INVOICE_OUTPUT = PROCESSED_DIR / "invoices_eligible.csv"

SUMMARY_OUTPUT = REPORTS_DIR / "eligibility_summary.json"


CONFIG = {
    "eligible_invoice_statuses": {
        "APPROVED",
        "PAID",
    },

    "cancelled_statuses": {
        "CANCELLED",
        "CANCELED",
    },

    "non_procurement_keywords": [
        "salary",
        "payroll",
        "wages",
        "employee reimbursement",
        "staff reimbursement",
        "income tax",
        "corporate tax",
        "vat payment",
        "dividend",
        "loan repayment",
        "interest payment",
        "bank charges",
    ],

    # Maximum absolute difference in ZAR.
    "amount_tolerance_zar": 0.05,

    # Automatic duplicate exclusion threshold.
    "minimum_text_similarity": 0.85,

    # Lower similarity is not sufficient for exclusion.
    "review_text_similarity": 0.60,
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def clean_text(value):
    """Normalize text for comparisons."""

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def normalize_status(value):
    """Normalize invoice status."""

    return clean_text(value)


def safe_numeric(series):
    """Convert a pandas Series to numeric values."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def clean_po_series(series):
    """
    Preserve PO identifiers as strings.

    This avoids the earlier numeric-to-string issue
    that produced values such as '4510106317.0'.
    """

    return (
        series
        .astype("string")
        .fillna("")
        .str.strip()
    )


def text_similarity(left, right):
    """
    Compare descriptions after sorting their tokens.

    This reduces sensitivity to word order.
    """

    a = clean_text(left)
    b = clean_text(right)

    if not a or not b:
        return 0.0

    a = " ".join(sorted(a.split()))
    b = " ".join(sorted(b.split()))

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def amount_difference(left, right):
    """
    Return absolute net amount difference.

    Returns infinity if either amount is missing.
    """

    if pd.isna(left) or pd.isna(right):
        return float("inf")

    return abs(float(left) - float(right))


# ============================================================
# 3. NON-PROCUREMENT DETECTION
# ============================================================

def is_non_procurement(line_text):
    """
    Detect clearly non-procurement transactions
    using configurable phrases.

    This is a conservative heuristic.
    """

    text = clean_text(line_text)

    if not text:
        return False

    for keyword in CONFIG["non_procurement_keywords"]:

        keyword_clean = clean_text(keyword)

        pattern = (
            rf"(?<![A-Z0-9])"
            rf"{re.escape(keyword_clean)}"
            rf"(?![A-Z0-9])"
        )

        if re.search(pattern, text):
            return True

    return False


# ============================================================
# 4. PREPARE ERP RECORDS
# ============================================================

def prepare_erp(erp):
    """
    Preserve and include all ERP records.

    ERP values are already net.
    """

    df = erp.copy()

    df["record_source"] = "ERP"

    df["include"] = True
    df["exclusion_reason"] = ""

    df["eligibility_note"] = "ERP source record"

    material_text = (
        df["material_short_text"]
        .astype("string")
        .fillna("")
    )

    service_text = (
        df["service_short_text"]
        .astype("string")
        .fillna("")
    )

    df["match_text"] = (
        material_text + " " + service_text
    ).apply(clean_text)

    df["match_amount"] = safe_numeric(
        df["erp_net_spend"]
    )

    df["match_supplier"] = (
        df["supplier_name_normalized"]
        .astype("string")
        .fillna("")
        .apply(clean_text)
    )

    df["match_po"] = clean_po_series(
        df["po_number_normalized"]
    )

    return df


# ============================================================
# 5. PREPARE INVOICE RECORDS
# ============================================================

def prepare_invoices(invoices):
    """
    Apply eligibility rules before duplicate detection.

    Exclusion precedence:
        CANCELLED
        NOT_APPROVED
        NON_PROCUREMENT
        DUPLICATE
    """

    df = invoices.copy()

    df["record_source"] = "INVOICE_TOOL"

    df["status_normalized"] = (
        df["status"].apply(normalize_status)
    )

    df["include"] = True
    df["exclusion_reason"] = ""
    df["eligibility_note"] = ""

    cancelled = df["status_normalized"].isin(
        CONFIG["cancelled_statuses"]
    )

    approved = df["status_normalized"].isin(
        CONFIG["eligible_invoice_statuses"]
    )

    not_approved = ~approved & ~cancelled

    non_procurement = (
        df["line_text"]
        .apply(is_non_procurement)
    )

    # Cancelled invoices.
    df.loc[cancelled, "include"] = False

    df.loc[
        cancelled,
        "exclusion_reason"
    ] = "CANCELLED"

    df.loc[
        cancelled,
        "eligibility_note"
    ] = "Invoice cancelled"

    # Other invoices not Approved/Paid.
    df.loc[not_approved, "include"] = False

    df.loc[
        not_approved,
        "exclusion_reason"
    ] = "NOT_APPROVED"

    df.loc[
        not_approved,
        "eligibility_note"
    ] = "Invoice not approved or paid"

    # Non-procurement check only for approved invoices.
    non_procurement_eligible = (
        approved & non_procurement
    )

    df.loc[
        non_procurement_eligible,
        "include"
    ] = False

    df.loc[
        non_procurement_eligible,
        "exclusion_reason"
    ] = "NON_PROCUREMENT"

    df.loc[
        non_procurement_eligible,
        "eligibility_note"
    ] = "Likely non-procurement transaction"

    df.loc[
        df["include"],
        "eligibility_note"
    ] = "Eligible invoice pending duplicate check"

    # Matching fields.
    df["match_text"] = (
        df["line_text"].apply(clean_text)
    )

    df["match_amount"] = safe_numeric(
        df["invoice_net_spend"]
    )

    df["match_supplier"] = (
        df["supplier_name_normalized"]
        .astype("string")
        .fillna("")
        .apply(clean_text)
    )

    df["match_po"] = clean_po_series(
        df["po_number_normalized"]
    )

    # Preserve previous audit columns.
    df["duplicate_candidate"] = False
    df["duplicate_review"] = False

    df["duplicate_match_type"] = ""
    df["duplicate_erp_line_id"] = ""

    df["duplicate_amount_difference"] = np.nan
    df["duplicate_text_similarity"] = np.nan

    return df


# ============================================================
# 6. BUILD ERP MATCHING INDEX
# ============================================================

def build_erp_index(erp):
    """
    Group ERP records by exact (PO, supplier).

    Each group is sorted by net amount so candidate
    searches can use the R0.05 tolerance efficiently.

    Returns:
        {
            (po, supplier): {
                "amounts": [...],
                "records": [...]
            }
        }
    """

    groups = defaultdict(list)

    for row in erp.itertuples(index=False):

        po = str(row.match_po)
        supplier = str(row.match_supplier)

        if not po or not supplier:
            continue

        if pd.isna(row.match_amount):
            continue

        groups[(po, supplier)].append({
            "line_id": str(row.line_id),
            "amount": float(row.match_amount),
            "text": str(row.match_text),
        })

    index = {}

    for key, records in groups.items():

        records.sort(
            key=lambda record: record["amount"]
        )

        index[key] = {
            "amounts": [
                record["amount"]
                for record in records
            ],
            "records": records,
        }

    return index


# ============================================================
# 7. GENERATE DUPLICATE CANDIDATES
# ============================================================

def generate_duplicate_candidates(erp, invoices):
    """
    Find candidate ERP/invoice pairs.

    Requirements for an automatic-match edge:
        - exact normalized PO
        - exact normalized supplier
        - absolute net amount difference <= R0.05
        - description similarity >= 0.85

    The function also records weaker matches for review.

    No invoice is excluded at this stage.
    """

    erp_index = build_erp_index(erp)

    tolerance = CONFIG["amount_tolerance_zar"]

    min_similarity = CONFIG["minimum_text_similarity"]

    strong_edges = []

    audit = {}

    eligible_indices = invoices.index[
        invoices["include"]
    ]

    for invoice_idx in eligible_indices:

        row = invoices.loc[invoice_idx]

        po = str(row["match_po"])
        supplier = str(row["match_supplier"])

        key = (po, supplier)

        if not po or not supplier:
            continue

        group = erp_index.get(key)

        if group is None:
            continue

        invoice_amount = row["match_amount"]

        # A shared PO and supplier is already
        # evidence of a possible duplicate.
        info = {
            "has_candidate": True,
            "has_amount_match": False,
            "best_erp_line_id": "",
            "best_similarity": np.nan,
            "best_amount_difference": np.nan,
            "strong_edge_count": 0,
        }

        audit[invoice_idx] = info

        if pd.isna(invoice_amount):
            continue

        invoice_amount = float(invoice_amount)

        amounts = group["amounts"]
        records = group["records"]

        # Only inspect ERP records within R0.05.
        lower = bisect_left(
            amounts,
            invoice_amount - tolerance - 1e-9
        )

        upper = bisect_right(
            amounts,
            invoice_amount + tolerance + 1e-9
        )

        best_score = None

        for candidate in records[lower:upper]:

            difference = amount_difference(
                candidate["amount"],
                invoice_amount
            )

            if difference > tolerance + 1e-9:
                continue

            similarity = text_similarity(
                candidate["text"],
                row["match_text"]
            )

            info["has_amount_match"] = True

            # Prefer greater similarity,
            # then smaller amount difference.
            score = (
                similarity,
                -difference
            )

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score

                info["best_erp_line_id"] = (
                    candidate["line_id"]
                )

                info["best_similarity"] = similarity

                info["best_amount_difference"] = (
                    difference
                )

            if similarity >= min_similarity:

                info["strong_edge_count"] += 1

                strong_edges.append({
                    "invoice_idx": invoice_idx,
                    "erp_line_id": candidate["line_id"],
                    "similarity": similarity,
                    "amount_difference": difference,
                })

    return strong_edges, audit


# ============================================================
# 8. ONE-TO-ONE DUPLICATE MATCHING
# ============================================================

def detect_duplicates(erp, invoices):
    """
    Perform greedy one-to-one duplicate matching.

    Strong candidate pairs are ordered by:
        1. Highest text similarity
        2. Lowest amount difference
        3. Stable ERP/invoice identifiers

    Each ERP row and invoice row can be assigned
    to at most one automatic duplicate pair.

    Unassigned or weak candidates remain included
    and are flagged for review.
    """

    df = invoices.copy()

    strong_edges, audit = (
        generate_duplicate_candidates(
            erp,
            df
        )
    )

    # Strongest candidate pairs first.
    strong_edges.sort(
        key=lambda edge: (
            -edge["similarity"],
            edge["amount_difference"],
            edge["erp_line_id"],
            str(
                df.at[
                    edge["invoice_idx"],
                    "tool_record_id"
                ]
            ),
        )
    )

    matched_erp = set()
    matched_invoices = set()

    # Assign one-to-one pairs.
    for edge in strong_edges:

        invoice_idx = edge["invoice_idx"]
        erp_line_id = edge["erp_line_id"]

        if invoice_idx in matched_invoices:
            continue

        if erp_line_id in matched_erp:
            continue

        matched_invoices.add(invoice_idx)
        matched_erp.add(erp_line_id)

        df.at[invoice_idx, "include"] = False

        df.at[
            invoice_idx,
            "exclusion_reason"
        ] = "DUPLICATE"

        df.at[
            invoice_idx,
            "eligibility_note"
        ] = "One-to-one high-confidence ERP duplicate"

        df.at[
            invoice_idx,
            "duplicate_match_type"
        ] = "PO_SUPPLIER_AMOUNT_TEXT_1TO1"

        df.at[
            invoice_idx,
            "duplicate_erp_line_id"
        ] = erp_line_id

        df.at[
            invoice_idx,
            "duplicate_amount_difference"
        ] = edge["amount_difference"]

        df.at[
            invoice_idx,
            "duplicate_text_similarity"
        ] = edge["similarity"]

    # Audit all eligible invoices that shared
    # a PO and supplier with ERP.
    review_count = 0
    candidate_count = 0

    for invoice_idx, info in audit.items():

        candidate_count += 1

        df.at[
            invoice_idx,
            "duplicate_candidate"
        ] = True

        # Successfully matched invoice.
        if invoice_idx in matched_invoices:
            continue

        # Otherwise keep it included for review.
        df.at[
            invoice_idx,
            "duplicate_review"
        ] = True

        review_count += 1

        if info["has_amount_match"]:

            df.at[
                invoice_idx,
                "duplicate_erp_line_id"
            ] = info["best_erp_line_id"]

            df.at[
                invoice_idx,
                "duplicate_amount_difference"
            ] = info["best_amount_difference"]

            df.at[
                invoice_idx,
                "duplicate_text_similarity"
            ] = info["best_similarity"]

            if info["strong_edge_count"] > 0:

                df.at[
                    invoice_idx,
                    "duplicate_match_type"
                ] = "ONE_TO_ONE_CONFLICT"

                df.at[
                    invoice_idx,
                    "eligibility_note"
                ] = (
                    "Strong ERP match exists but "
                    "ERP record already assigned"
                )

            elif (
                info["best_similarity"]
                >= CONFIG["review_text_similarity"]
            ):

                df.at[
                    invoice_idx,
                    "duplicate_match_type"
                ] = "POSSIBLE_ERP_DUPLICATE"

                df.at[
                    invoice_idx,
                    "eligibility_note"
                ] = (
                    "Amount matches ERP but "
                    "description evidence is insufficient"
                )

            else:

                df.at[
                    invoice_idx,
                    "duplicate_match_type"
                ] = "AMOUNT_MATCH_TEXT_MISMATCH"

                df.at[
                    invoice_idx,
                    "eligibility_note"
                ] = (
                    "ERP amount matches but "
                    "description differs"
                )

        else:

            df.at[
                invoice_idx,
                "duplicate_match_type"
            ] = "PO_SUPPLIER_ONLY"

            df.at[
                invoice_idx,
                "eligibility_note"
            ] = (
                "PO and supplier match ERP, "
                "but no ERP amount is within tolerance"
            )

    print("\nDUPLICATE MATCHING RESULTS")
    print("-" * 55)

    print(
        "Eligible invoices checked:",
        f"{int(invoices['include'].sum()):,}"
    )

    print(
        "Invoices sharing PO and supplier with ERP:",
        f"{candidate_count:,}"
    )

    print(
        "Strong candidate pairs generated:",
        f"{len(strong_edges):,}"
    )

    print(
        "One-to-one duplicates excluded:",
        f"{len(matched_invoices):,}"
    )

    print(
        "Possible duplicates flagged for review:",
        f"{review_count:,}"
    )

    print(
        "Unique ERP records used:",
        f"{len(matched_erp):,}"
    )

    return df


# ============================================================
# 9. FINALIZE SPEND
# ============================================================

def finalize_spend(df, amount_column):
    """
    Keep the original net amount for auditing.

    Set spend to zero for excluded records.
    """

    result = df.copy()

    result["spend_before_exclusion"] = (
        safe_numeric(result[amount_column])
    )

    result["spend"] = np.where(
        result["include"],
        result["spend_before_exclusion"],
        0.0
    )

    return result


# ============================================================
# 10. VALIDATION
# ============================================================

def validate_outputs(
    erp_original,
    invoices_original,
    erp_result,
    invoices_result
):

    assert len(erp_original) == len(erp_result), (
        "ERP row count changed."
    )

    assert len(invoices_original) == len(invoices_result), (
        "Invoice row count changed."
    )

    assert erp_result["line_id"].is_unique, (
        "ERP line_id values are not unique."
    )

    assert invoices_result["tool_record_id"].is_unique, (
        "Invoice tool_record_id values are not unique."
    )

    for name, df in [
        ("ERP", erp_result),
        ("INVOICE", invoices_result),
    ]:

        assert df["include"].notna().all(), (
            f"{name}: missing include values."
        )

        excluded = df.loc[
            ~df["include"]
        ]

        assert excluded["spend"].eq(0).all(), (
            f"{name}: excluded spend is not zero."
        )

        assert excluded[
            "exclusion_reason"
        ].ne("").all(), (
            f"{name}: missing exclusion reasons."
        )

        included = df.loc[
            df["include"]
        ]

        assert included[
            "exclusion_reason"
        ].eq("").all(), (
            f"{name}: included records have exclusion reasons."
        )

        assert df["spend"].notna().all(), (
            f"{name}: missing spend values."
        )

    # Verify the one-to-one invariant.
    duplicates = invoices_result.loc[
        invoices_result["exclusion_reason"] == "DUPLICATE"
    ]

    assert duplicates[
        "duplicate_erp_line_id"
    ].is_unique, (
        "One-to-one matching failed: "
        "an ERP row was assigned more than once."
    )

    assert duplicates[
        "duplicate_amount_difference"
    ].le(
        CONFIG["amount_tolerance_zar"] + 1e-9
    ).all(), (
        "Duplicate amount tolerance violated."
    )

    assert duplicates[
        "duplicate_text_similarity"
    ].ge(
        CONFIG["minimum_text_similarity"]
    ).all(), (
        "Duplicate text similarity threshold violated."
    )

    assert not invoices_result.loc[
        invoices_result["duplicate_review"],
        "exclusion_reason"
    ].eq("DUPLICATE").any(), (
        "Review candidates must not be excluded as duplicates."
    )

    print("\nVALIDATION PASSED")
    print("-" * 55)

    print("All input records preserved.")
    print("All excluded records have zero spend.")
    print("All excluded records have an exclusion reason.")
    print("Every automatic duplicate uses a unique ERP row.")
    print("All duplicate amounts satisfy R0.05 tolerance.")
    print("All duplicate text similarities meet threshold.")
    print("Uncertain candidates remain included.")


# ============================================================
# 11. SUMMARY REPORT
# ============================================================

def build_summary(erp, invoices):

    invoice_reasons = (
        invoices["exclusion_reason"]
        .replace("", "INCLUDED")
        .value_counts()
        .to_dict()
    )

    review_types = (
        invoices.loc[
            invoices["duplicate_review"],
            "duplicate_match_type"
        ]
        .value_counts()
        .to_dict()
    )

    duplicates = invoices.loc[
        invoices["exclusion_reason"] == "DUPLICATE"
    ]

    summary = {
        "matching_configuration": {
            "amount_tolerance_zar": (
                CONFIG["amount_tolerance_zar"]
            ),
            "minimum_text_similarity": (
                CONFIG["minimum_text_similarity"]
            ),
            "one_to_one_matching": True,
            "matching_strategy": (
                "greedy_strongest_pair_first"
            ),
            "supplier_matching": (
                "exact_normalized_name"
            ),
        },

        "erp": {
            "total_records": int(len(erp)),

            "included_records": int(
                erp["include"].sum()
            ),

            "excluded_records": int(
                (~erp["include"]).sum()
            ),

            "included_net_spend": float(
                erp["spend"].sum()
            ),
        },

        "invoice_tool": {
            "total_records": int(len(invoices)),

            "included_records": int(
                invoices["include"].sum()
            ),

            "excluded_records": int(
                (~invoices["include"]).sum()
            ),

            "included_net_spend": float(
                invoices["spend"].sum()
            ),

            "exclusion_breakdown": {
                str(key): int(value)
                for key, value
                in invoice_reasons.items()
            },

            "duplicate_candidates": int(
                invoices["duplicate_candidate"].sum()
            ),

            "duplicate_review_count": int(
                invoices["duplicate_review"].sum()
            ),

            "duplicate_review_breakdown": {
                str(key): int(value)
                for key, value
                in review_types.items()
            },

            "unique_erp_duplicate_matches": int(
                duplicates[
                    "duplicate_erp_line_id"
                ].nunique()
            ),
        },

        "combined": {
            "total_input_records": int(
                len(erp) + len(invoices)
            ),

            "included_records": int(
                erp["include"].sum()
                + invoices["include"].sum()
            ),

            "total_included_net_spend": float(
                erp["spend"].sum()
                + invoices["spend"].sum()
            ),
        },
    }

    return summary


def print_summary(summary):

    print("\n" + "=" * 65)
    print("ELIGIBILITY SUMMARY")
    print("=" * 65)

    erp = summary["erp"]
    inv = summary["invoice_tool"]
    combined = summary["combined"]

    print("\nERP")

    print(
        f"Total: {erp['total_records']:,}"
    )

    print(
        f"Included: {erp['included_records']:,}"
    )

    print(
        f"Excluded: {erp['excluded_records']:,}"
    )

    print("\nINVOICE TOOL")

    print(
        f"Total: {inv['total_records']:,}"
    )

    print(
        f"Included: {inv['included_records']:,}"
    )

    print(
        f"Excluded: {inv['excluded_records']:,}"
    )

    print("\nINVOICE EXCLUSION BREAKDOWN")

    for reason, count in (
        inv["exclusion_breakdown"].items()
    ):
        print(f"{reason}: {count:,}")

    print("\nDUPLICATE REVIEW BREAKDOWN")

    for reason, count in (
        inv["duplicate_review_breakdown"].items()
    ):
        print(f"{reason}: {count:,}")

    print("\nCOMBINED")

    print(
        "Total input records:",
        f"{combined['total_input_records']:,}"
    )

    print(
        "Included records:",
        f"{combined['included_records']:,}"
    )

    print(
        "Total included net spend (ZAR):",
        f"{combined['total_included_net_spend']:,.2f}"
    )


# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

def save_outputs(erp, invoices, summary):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    erp.to_csv(
        ERP_OUTPUT,
        index=False
    )

    invoices.to_csv(
        INVOICE_OUTPUT,
        index=False
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print("\nFILES SAVED SUCCESSFULLY")
    print("-" * 55)

    print(ERP_OUTPUT)
    print(INVOICE_OUTPUT)
    print(SUMMARY_OUTPUT)


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COMPANY A - ELIGIBILITY AND DEDUPLICATION")
    print("=" * 65)

    print("\nLoading normalized datasets...")

    # Preserve PO identifiers as strings.
    erp_raw = pd.read_csv(
        ERP_INPUT,
        low_memory=False,
        dtype={
            "po_number_normalized": "string"
        }
    )

    invoices_raw = pd.read_csv(
        INVOICE_INPUT,
        low_memory=False,
        dtype={
            "po_number_normalized": "string"
        }
    )

    print(
        f"ERP records loaded: {len(erp_raw):,}"
    )

    print(
        f"Invoice records loaded: {len(invoices_raw):,}"
    )

    print("\nPreparing ERP records...")

    erp = prepare_erp(
        erp_raw
    )

    print("Applying invoice eligibility rules...")

    invoices = prepare_invoices(
        invoices_raw
    )

    print(
        "Checking ERP-invoice duplicates "
        "with one-to-one matching..."
    )

    invoices = detect_duplicates(
        erp,
        invoices
    )

    erp = finalize_spend(
        erp,
        "erp_net_spend"
    )

    invoices = finalize_spend(
        invoices,
        "invoice_net_spend"
    )

    validate_outputs(
        erp_raw,
        invoices_raw,
        erp,
        invoices
    )

    summary = build_summary(
        erp,
        invoices
    )

    print_summary(
        summary
    )

    save_outputs(
        erp,
        invoices,
        summary
    )

    print(
        "\nEligibility and deduplication completed."
    )


if __name__ == "__main__":
    main()
