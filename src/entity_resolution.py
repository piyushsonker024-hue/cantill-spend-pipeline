
"""
Company A - Step 4: Supplier and Item Entity Resolution

Run:
    python -m src.entity_resolution

Inputs:
    data/processed/erp_eligible.csv
    data/processed/invoices_eligible.csv

Outputs:
    data/processed/erp_resolved.csv
    data/processed/invoices_resolved.csv
    data/processed/supplier_mapping.csv
    data/processed/item_mapping.csv
    reports/entity_resolution_summary.json

Important:
- Does NOT read labels_A.csv.
- Does NOT use taxonomy_A.csv.
- Preserves all records and Step 3 eligibility decisions.
- Produces deterministic supplier_key and item_key values.
- Does not change spend or exclusion reasons.
"""

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

ERP_INPUT = DATA_DIR / "erp_eligible.csv"
INVOICE_INPUT = DATA_DIR / "invoices_eligible.csv"

ERP_OUTPUT = DATA_DIR / "erp_resolved.csv"
INVOICE_OUTPUT = DATA_DIR / "invoices_resolved.csv"

SUPPLIER_MAPPING_OUTPUT = DATA_DIR / "supplier_mapping.csv"
ITEM_MAPPING_OUTPUT = DATA_DIR / "item_mapping.csv"

SUMMARY_OUTPUT = REPORT_DIR / "entity_resolution_summary.json"


CONFIG = {
    # High threshold for fuzzy supplier matching.
    "supplier_fuzzy_threshold": 95,

    # Minimum length for fuzzy supplier comparisons.
    "minimum_supplier_name_length": 5,

    # Supplier fuzzy matches require a shared PO.
    "require_shared_po_for_fuzzy_supplier": True,

    # Do not automatically merge different item descriptions
    # using fuzzy similarity alone.
    "enable_fuzzy_item_merging": False,

    # Require exact normalized description for transferring
    # ERP material identity to invoice rows.
    "require_exact_item_description": True,
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def clean_text(value):
    """Normalize text for identity comparisons."""

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def stable_key(prefix, value):
    """
    Create a deterministic key from normalized content.

    Example:
        SUP_...
        ITEM_...

    Keys are independent of CSV row order.
    """

    digest = hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()[:20].upper()

    return f"{prefix}_{digest}"


def normalize_po(series):
    """
    Keep PO numbers as strings.

    Handles CSV values already normalized in Step 2.
    """

    return (
        series
        .astype("string")
        .fillna("")
        .str.strip()
    )


def normalize_material_number(value):
    """
    Normalize ERP material numbers.

    Preserve alphanumeric identifiers.
    Avoid converting identifiers to floating-point values.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text.endswith(".0") and re.fullmatch(
        r"\d+\.0",
        text
    ):
        text = text[:-2]

    return text


def normalize_item_description(value):
    """
    Normalize item descriptions.

    Sorting tokens makes word-order variations
    comparable:

        STEEL PIPE 50MM
        50MM STEEL PIPE

    Both produce the same token signature.
    """

    text = clean_text(value)

    if not text:
        return ""

    tokens = sorted(text.split())

    return " ".join(tokens)


# ============================================================
# 3. SUPPLIER NAME NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "PTY",
    "LTD",
    "LIMITED",
    "PROPRIETARY",
    "PVT",
    "PRIVATE",
    "INC",
    "INCORPORATED",
    "LLC",
}


def supplier_core_name(value):
    """
    Remove common legal suffixes.

    Examples:
        KELMONT PTY LTD -> KELMONT
        LORDEN SOLUTIONS PTY LTD -> LORDEN SOLUTIONS

    These are matching features, not authoritative
    legal entity identifiers.
    """

    text = clean_text(value)

    tokens = text.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def supplier_compact_name(value):
    """
    Remove spaces from the supplier core.

    Examples:
        PEL WICK -> PELWICK
        VER RICK -> VERRICK
    """

    return supplier_core_name(value).replace(" ", "")


# ============================================================
# 4. UNION-FIND FOR SUPPLIER GROUPS
# ============================================================

class UnionFind:
    """
    Efficiently merge equivalent supplier names.
    """

    def __init__(self, values):

        self.parent = {
            value: value
            for value in values
        }

    def find(self, value):

        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(parent)

        return self.parent[value]

    def union(self, left, right):

        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        # Deterministic representative.
        representative = min(
            root_left,
            root_right
        )

        other = max(
            root_left,
            root_right
        )

        self.parent[other] = representative


# ============================================================
# 5. SUPPLIER RESOLUTION
# ============================================================

def resolve_suppliers(combined):
    """
    Build supplier identity groups.

    Matching stages:
        1. Exact normalized name
        2. Exact compact core name
        3. High-similarity names sharing a PO

    Fuzzy supplier matches are conservative.
    """

    df = combined.copy()

    df["supplier_name_clean"] = (
        df["supplier_name_normalized"]
        .apply(clean_text)
    )

    df["supplier_core"] = (
        df["supplier_name_clean"]
        .apply(supplier_core_name)
    )

    df["supplier_compact"] = (
        df["supplier_name_clean"]
        .apply(supplier_compact_name)
    )

    df["po_match_key"] = normalize_po(
        df["po_number_normalized"]
    )

    names = sorted(
        name
        for name in df["supplier_name_clean"].unique()
        if name
    )

    uf = UnionFind(names)

    # --------------------------------------------------------
    # Stage A: exact compact-name matching
    # --------------------------------------------------------

    compact_groups = defaultdict(list)

    for name in names:

        compact = supplier_compact_name(name)

        if compact:
            compact_groups[compact].append(name)

    exact_compact_merges = 0

    for group in compact_groups.values():

        anchor = group[0]

        for name in group[1:]:

            if uf.find(anchor) != uf.find(name):

                uf.union(anchor, name)
                exact_compact_merges += 1

    # --------------------------------------------------------
    # Stage B: fuzzy matching within shared PO groups
    # --------------------------------------------------------

    po_supplier_groups = (
        df.loc[
            df["po_match_key"].ne("")
            & df["supplier_name_clean"].ne(""),
            ["po_match_key", "supplier_name_clean"]
        ]
        .drop_duplicates()
        .groupby("po_match_key")[
            "supplier_name_clean"
        ]
        .apply(lambda s: sorted(set(s)))
    )

    candidate_pairs = set()

    for supplier_group in po_supplier_groups:

        for i, left in enumerate(supplier_group):

            for right in supplier_group[i + 1:]:

                candidate_pairs.add(
                    tuple(sorted((left, right)))
                )

    fuzzy_merges = 0

    for left, right in sorted(candidate_pairs):

        if uf.find(left) == uf.find(right):
            continue

        core_left = supplier_compact_name(left)
        core_right = supplier_compact_name(right)

        if (
            len(core_left)
            < CONFIG["minimum_supplier_name_length"]
            or len(core_right)
            < CONFIG["minimum_supplier_name_length"]
        ):
            continue

        similarity = fuzz.ratio(
            core_left,
            core_right
        )

        if similarity >= CONFIG["supplier_fuzzy_threshold"]:

            uf.union(left, right)
            fuzzy_merges += 1

    # --------------------------------------------------------
    # Stage C: assign stable supplier keys
    # --------------------------------------------------------

    groups = defaultdict(list)

    for name in names:
        groups[uf.find(name)].append(name)

    name_to_key = {}

    for group_names in groups.values():

        # Stable canonical representative.
        canonical = min(
            supplier_compact_name(name)
            for name in group_names
        )

        key = stable_key(
            "SUP",
            canonical
        )

        for name in group_names:
            name_to_key[name] = key

    df["supplier_key"] = (
        df["supplier_name_clean"]
        .map(name_to_key)
    )

    # Avoid missing keys for blank names.
    missing = df["supplier_key"].isna()

    if missing.any():

        df.loc[missing, "supplier_key"] = (
            df.loc[missing, "supplier_name_clean"]
            .apply(
                lambda value: stable_key(
                    "SUP",
                    "UNKNOWN:" + value
                )
            )
        )

    supplier_mapping = (
        df[
            [
                "supplier_name_clean",
                "supplier_core",
                "supplier_compact",
                "supplier_key",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["supplier_key", "supplier_name_clean"]
        )
        .reset_index(drop=True)
    )

    stats = {
        "unique_raw_supplier_names": len(names),
        "resolved_supplier_entities": int(
            df["supplier_key"].nunique()
        ),
        "exact_compact_merges": exact_compact_merges,
        "fuzzy_merges": fuzzy_merges,
        "fuzzy_candidate_pairs": len(candidate_pairs),
    }

    return df, supplier_mapping, stats


# ============================================================
# 6. PREPARE ITEM DESCRIPTIONS
# ============================================================

def prepare_item_fields(combined):
    """
    Build a common item description for ERP and invoices.

    ERP:
        material_short_text
        service_short_text

    Invoice:
        line_text
    """

    df = combined.copy()

    erp_mask = df["record_source"].eq("ERP")

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

    invoice_text = (
        df["line_text"]
        .astype("string")
        .fillna("")
    )

    erp_text = (
        material_text + " " + service_text
    ).str.strip()

    df["item_description_raw"] = invoice_text

    df.loc[
        erp_mask,
        "item_description_raw"
    ] = erp_text.loc[erp_mask]

    df["item_description_normalized"] = (
        df["item_description_raw"]
        .apply(normalize_item_description)
    )

    df["material_number_normalized"] = (
        df["material_number"]
        .apply(normalize_material_number)
    )

    return df


# ============================================================
# 7. ITEM RESOLUTION
# ============================================================

def resolve_items(combined):
    """
    Resolve item identities.

    Priority:
        1. ERP material number
        2. Unique material mapping from supplier + description
        3. Supplier-scoped normalized description
        4. Unique record fallback when description is missing

    Material-number identities are kept distinct from
    description-only identities.

    Fuzzy item merging is deliberately disabled.
    """

    df = prepare_item_fields(combined)

    df["item_key"] = ""
    df["item_resolution_method"] = ""

    erp_mask = df["record_source"].eq("ERP")

    has_material = (
        erp_mask
        & df["material_number_normalized"].ne("")
    )

    # --------------------------------------------------------
    # Stage A: material number identity
    # --------------------------------------------------------

    df.loc[
        has_material,
        "item_key"
    ] = df.loc[
        has_material,
        "material_number_normalized"
    ].apply(
        lambda value: stable_key(
            "ITEM",
            "MATERIAL:" + value
        )
    )

    df.loc[
        has_material,
        "item_resolution_method"
    ] = "ERP_MATERIAL_NUMBER"

    # --------------------------------------------------------
    # Stage B: supplier + description -> material mapping
    # --------------------------------------------------------

    material_reference = (
        df.loc[
            has_material
            & df["item_description_normalized"].ne(""),
            [
                "supplier_key",
                "item_description_normalized",
                "item_key",
            ]
        ]
        .drop_duplicates()
    )

    # Only transfer a material identity when the
    # supplier+description maps to exactly one item key.
    reference_counts = (
        material_reference
        .groupby(
            [
                "supplier_key",
                "item_description_normalized",
            ]
        )["item_key"]
        .nunique()
    )

    unique_reference_keys = (
        reference_counts[
            reference_counts.eq(1)
        ]
        .index
    )

    unique_reference = (
        material_reference
        .set_index(
            [
                "supplier_key",
                "item_description_normalized",
            ]
        )
        .loc[unique_reference_keys]
        .reset_index()
    )

    material_lookup = {
        (
            row.supplier_key,
            row.item_description_normalized
        ): row.item_key

        for row in unique_reference.itertuples(
            index=False
        )
    }

    unresolved = df["item_key"].eq("")

    for idx in df.index[unresolved]:

        supplier = df.at[idx, "supplier_key"]
        description = df.at[
            idx,
            "item_description_normalized"
        ]

        if not description:
            continue

        match = material_lookup.get(
            (supplier, description)
        )

        if match is not None:

            df.at[idx, "item_key"] = match

            df.at[
                idx,
                "item_resolution_method"
            ] = "SUPPLIER_DESCRIPTION_TO_MATERIAL"

    # --------------------------------------------------------
    # Stage C: supplier-scoped description identity
    # --------------------------------------------------------

    unresolved = (
        df["item_key"].eq("")
        & df["item_description_normalized"].ne("")
    )

    for idx in df.index[unresolved]:

        supplier = df.at[idx, "supplier_key"]

        description = df.at[
            idx,
            "item_description_normalized"
        ]

        identity = (
            f"DESCRIPTION:{supplier}:{description}"
        )

        df.at[idx, "item_key"] = stable_key(
            "ITEM",
            identity
        )

        df.at[
            idx,
            "item_resolution_method"
        ] = "SUPPLIER_DESCRIPTION"

    # --------------------------------------------------------
    # Stage D: missing-description fallback
    # --------------------------------------------------------

    unresolved = df["item_key"].eq("")

    for idx in df.index[unresolved]:

        source = df.at[idx, "record_source"]

        if source == "ERP":

            record_id = str(
                df.at[idx, "line_id"]
            )

        else:

            record_id = str(
                df.at[idx, "tool_record_id"]
            )

        identity = (
            f"UNKNOWN_ITEM:{source}:{record_id}"
        )

        df.at[idx, "item_key"] = stable_key(
            "ITEM",
            identity
        )

        df.at[
            idx,
            "item_resolution_method"
        ] = "RECORD_FALLBACK"

    item_mapping = (
        df[
            [
                "record_source",
                "supplier_key",
                "material_number_normalized",
                "item_description_normalized",
                "item_key",
                "item_resolution_method",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "item_key",
                "supplier_key",
                "record_source",
            ]
        )
        .reset_index(drop=True)
    )

    stats = {
        "resolved_item_entities": int(
            df["item_key"].nunique()
        ),

        "item_resolution_breakdown": {
            str(key): int(value)

            for key, value in (
                df["item_resolution_method"]
                .value_counts()
                .items()
            )
        },

        "missing_item_keys": int(
            df["item_key"].eq("").sum()
        ),
    }

    return df, item_mapping, stats


# ============================================================
# 8. VALIDATION
# ============================================================

def validate_resolution(
    original,
    resolved
):
    """
    Ensure entity resolution changes only identity fields.

    It must not change:
        record count
        inclusion decisions
        exclusion reasons
        spend
    """

    assert len(original) == len(resolved), (
        "Record count changed during entity resolution."
    )

    assert resolved["supplier_key"].notna().all(), (
        "Missing supplier keys."
    )

    assert resolved["item_key"].notna().all(), (
        "Missing item keys."
    )

    assert resolved["supplier_key"].ne("").all(), (
        "Blank supplier keys."
    )

    assert resolved["item_key"].ne("").all(), (
        "Blank item keys."
    )

    assert (
        original["include"].tolist()
        == resolved["include"].tolist()
    ), "Eligibility decisions changed."

    assert (
        original["spend"].tolist()
        == resolved["spend"].tolist()
    ), "Spend values changed."

    assert (
    original["exclusion_reason"]
    .astype("string")
    .fillna("")
    .tolist()
    ==
    resolved["exclusion_reason"]
    .astype("string")
    .fillna("")
    .tolist()
), "Exclusion reasons changed."

    print("\nVALIDATION PASSED")
    print("-" * 55)
    print("All records preserved.")
    print("All supplier keys populated.")
    print("All item keys populated.")
    print("Eligibility decisions unchanged.")
    print("Spend values unchanged.")
    print("Exclusion reasons unchanged.")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COMPANY A - ENTITY RESOLUTION")
    print("=" * 65)

    print("\nLoading Step 3 outputs...")

    erp = pd.read_csv(
        ERP_INPUT,
        low_memory=False,
        dtype={
            "po_number_normalized": "string",
            "material_number": "string",
        }
    )

    invoices = pd.read_csv(
        INVOICE_INPUT,
        low_memory=False,
        dtype={
            "po_number_normalized": "string",
        }
    )

    print(f"ERP records: {len(erp):,}")
    print(f"Invoice records: {len(invoices):,}")

    # Add missing source-specific columns so both
    # datasets can be concatenated safely.
    required_columns = [
        "material_short_text",
        "service_short_text",
        "material_number",
        "line_text",
        "line_id",
        "tool_record_id",
    ]

    for column in required_columns:

        if column not in erp.columns:
            erp[column] = pd.NA

        if column not in invoices.columns:
            invoices[column] = pd.NA

    erp["record_source"] = "ERP"
    invoices["record_source"] = "INVOICE_TOOL"

    combined = pd.concat(
        [erp, invoices],
        ignore_index=True,
        sort=False
    )

    print("\nResolving supplier identities...")

    combined, supplier_mapping, supplier_stats = (
        resolve_suppliers(combined)
    )

    print(
        "Raw normalized supplier names:",
        f"{supplier_stats['unique_raw_supplier_names']:,}"
    )

    print(
        "Resolved supplier entities:",
        f"{supplier_stats['resolved_supplier_entities']:,}"
    )

    print(
        "Exact compact-name merges:",
        f"{supplier_stats['exact_compact_merges']:,}"
    )

    print(
        "Fuzzy supplier merges:",
        f"{supplier_stats['fuzzy_merges']:,}"
    )

    print("\nResolving item identities...")

    combined, item_mapping, item_stats = (
        resolve_items(combined)
    )

    print(
        "Resolved item entities:",
        f"{item_stats['resolved_item_entities']:,}"
    )

    print("\nITEM RESOLUTION BREAKDOWN")

    for method, count in (
        item_stats["item_resolution_breakdown"].items()
    ):

        print(f"{method}: {count:,}")

    # Split back into original source order.
    erp_resolved = combined.iloc[
        :len(erp)
    ].copy()

    invoice_resolved = combined.iloc[
        len(erp):
    ].copy()

    # Validate each source independently.
    validate_resolution(
        erp,
        erp_resolved
    )

    validate_resolution(
        invoices,
        invoice_resolved
    )

    summary = {
        "total_records": int(len(combined)),
        "supplier_resolution": supplier_stats,
        "item_resolution": item_stats,

        "included_records": int(
            combined["include"].sum()
        ),

        "included_net_spend_zar": float(
            combined["spend"].sum()
        ),

        "labels_used": False,
    }

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    erp_resolved.to_csv(
        ERP_OUTPUT,
        index=False
    )

    invoice_resolved.to_csv(
        INVOICE_OUTPUT,
        index=False
    )

    supplier_mapping.to_csv(
        SUPPLIER_MAPPING_OUTPUT,
        index=False
    )

    item_mapping.to_csv(
        ITEM_MAPPING_OUTPUT,
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
    print(SUPPLIER_MAPPING_OUTPUT)
    print(ITEM_MAPPING_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nENTITY RESOLUTION SUMMARY")
    print("-" * 55)

    print(
        "Total records:",
        f"{summary['total_records']:,}"
    )

    print(
        "Supplier entities:",
        f"{supplier_stats['resolved_supplier_entities']:,}"
    )

    print(
        "Item entities:",
        f"{item_stats['resolved_item_entities']:,}"
    )

    print(
        "Included net spend (ZAR):",
        f"{summary['included_net_spend_zar']:,.2f}"
    )

    print("\nEntity resolution completed successfully.")


if __name__ == "__main__":
    main()
