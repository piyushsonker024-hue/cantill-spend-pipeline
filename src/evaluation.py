
"""
STEP 6 - COMPANY A CLASSIFICATION EVALUATION

IMPORTANT:
    Evaluation only.
    Ground-truth labels are NEVER used for inference.

Run:
    python -m src.evaluation

Inputs:
    data/processed/erp_classified.csv
    data/raw/labels_A.csv

Outputs:
    reports/evaluation_summary.json
    reports/evaluation_predictions.csv
    reports/evaluation_errors.csv
    reports/evaluation_confidence_bands.csv
    reports/evaluation_category_metrics.csv
    reports/confusion_matrix_l1.csv
    reports/confusion_matrix_l2.csv
    reports/confusion_matrix_l3.csv
    reports/evaluation_low_confidence.csv
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_FILE = (
    PROCESSED_DIR / "erp_classified.csv"
)

LABELS_FILE = (
    RAW_DIR / "labels_A.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "evaluation_summary.json"
)

PREDICTIONS_OUTPUT = (
    REPORT_DIR / "evaluation_predictions.csv"
)

ERRORS_OUTPUT = (
    REPORT_DIR / "evaluation_errors.csv"
)

CONFIDENCE_OUTPUT = (
    REPORT_DIR / "evaluation_confidence_bands.csv"
)

CATEGORY_OUTPUT = (
    REPORT_DIR / "evaluation_category_metrics.csv"
)

LOW_CONFIDENCE_OUTPUT = (
    REPORT_DIR / "evaluation_low_confidence.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

CONFIG = {
    "review_threshold": 0.40,
    "low_confidence_examples": 500,
    "error_examples": 1000,
}

CONFIDENCE_BINS = [
    -0.000001,
    0.10,
    0.20,
    0.40,
    0.60,
    0.80,
    1.000001,
]

CONFIDENCE_LABELS = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.40",
    "0.40-0.60",
    "0.60-0.80",
    "0.80-1.00",
]

LEVELS = ["l1", "l2", "l3"]


# ============================================================
# 3. HELPERS
# ============================================================

def clean_column_name(value):
    """Normalize CSV header names."""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower()
    )


def normalize_category(value):
    """
    Normalize category text for comparison.

    Does not change classification outputs.
    """

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    ).casefold()


def find_column(columns, aliases):
    """Find a column using accepted aliases."""

    normalized = {
        clean_column_name(column): column
        for column in columns
    }

    for alias in aliases:
        key = clean_column_name(alias)

        if key in normalized:
            return normalized[key]

    return None


def category_aliases(level):
    """Supported names for L1, L2 and L3."""

    number = level[-1]

    return [
        level,
        f"category_{level}",
        f"category_{number}",
        f"category_level_{number}",
        f"level_{number}",
        f"level{number}",
        f"lvl{number}",
        f"label_{level}",
        f"true_{level}",
        f"gold_{level}",
    ]


def parse_include(series):
    """Convert common CSV boolean representations."""

    normalized = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true", "1", "yes", "y"
    }

    false_values = {
        "false", "0", "no", "n", ""
    }

    unexpected = set(
        normalized.unique().tolist()
    ) - true_values - false_values

    if unexpected:
        raise ValueError(
            "Unexpected values in include column: "
            f"{sorted(unexpected)}"
        )

    return normalized.isin(true_values)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def evaluate_accuracy(df, column, mask=None):
    """Return accuracy for one hierarchy level."""

    if mask is not None:
        df = df.loc[mask]

    if df.empty:
        return None

    return float(
        accuracy_score(
            df[f"true_{column}_normalized"],
            df[f"pred_{column}_normalized"]
        )
    )


def spend_weighted_accuracy(
    df,
    correct_column
):
    """
    Evaluate classification correctness weighted
    by included net spend.

    Excluded records are not counted.
    """

    included = df.loc[
        df["include_normalized"]
    ]

    if included.empty:
        return None

    weights = included[
        "spend_numeric"
    ].to_numpy(dtype=float)

    if np.any(weights < 0):
        # Negative amounts may represent returns/credits.
        # Signed amounts are unsuitable as accuracy weights.
        # Use absolute net spend magnitude.
        weights = np.abs(weights)

    total_weight = float(weights.sum())

    if total_weight == 0:
        return None

    correct = included[
        correct_column
    ].to_numpy(dtype=bool)

    return float(
        weights[correct].sum() / total_weight
    )


# ============================================================
# 4. LOAD LABELS AND PREDICTIONS
# ============================================================

def load_data():

    print("\nLoading classified ERP records...")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing predictions: {PREDICTIONS_FILE}"
        )

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Missing labels: {LABELS_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        low_memory=False,
        dtype={"line_id": "string"}
    )

    print(
        "ERP prediction records:",
        f"{len(predictions):,}"
    )

    print("\nLoading evaluation labels...")

    labels = pd.read_csv(
        LABELS_FILE,
        low_memory=False,
        dtype="string"
    )

    print(
        "Ground-truth label records:",
        f"{len(labels):,}"
    )

    print(
        "Label columns:",
        list(labels.columns)
    )

    if "line_id" not in predictions.columns:
        raise ValueError(
            "Predictions missing line_id."
        )

    label_id_column = find_column(
    labels.columns,
    ["line_id", "lineid", "record_id", "recordid"]
    )

    if label_id_column is None:
        raise ValueError(
            "labels_A.csv must contain line_id."
        )

    labels = labels.rename(
        columns={
            label_id_column: "line_id"
        }
    )

    # Identify ground-truth hierarchy columns.
    rename_labels = {}

    for level in LEVELS:

        label_column = find_column(
            labels.columns,
            category_aliases(level)
        )

        if label_column is None:
            raise ValueError(
                f"Could not find {level.upper()} "
                "in labels_A.csv. "
                f"Available columns: {list(labels.columns)}"
            )

        rename_labels[label_column] = (
            f"true_{level}"
        )

    labels = labels.rename(
        columns=rename_labels
    )

    # Rename predictions to make the distinction
    # between true and predicted labels explicit.
    for level in LEVELS:

        if level not in predictions.columns:
            raise ValueError(
                f"Predictions missing {level}."
            )

    predictions = predictions.rename(
        columns={
            level: f"pred_{level}"
            for level in LEVELS
        }
    )

    # Validate matching identifiers.
    predictions["line_id"] = (
        predictions["line_id"]
        .astype("string")
        .str.strip()
    )

    labels["line_id"] = (
    labels["line_id"]
    .astype("string")
    .str.strip()
    .str.replace(
        r"^ERP:",
        "",
        regex=True
    )
    )

    for name, frame in [
        ("Predictions", predictions),
        ("Labels", labels),
    ]:

        if frame["line_id"].isna().any():
            raise ValueError(
                f"{name} contains missing line_id."
            )

        if frame["line_id"].eq("").any():
            raise ValueError(
                f"{name} contains blank line_id."
            )

        if frame["line_id"].duplicated().any():

            duplicates = (
                frame.loc[
                    frame["line_id"].duplicated(),
                    "line_id"
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"{name} has duplicate line_id values: "
                f"{duplicates}"
            )

    # Use only label ID and true taxonomy columns.
    # Supplier and item keys in the labels file
    # are intentionally ignored.
    label_subset = labels[
        [
            "line_id",
            "true_l1",
            "true_l2",
            "true_l3",
        ]
    ].copy()

    merged = predictions.merge(
        label_subset,
        on="line_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False
    )

    missing_labels = int(
        merged["_merge"].ne("both").sum()
    )

    if missing_labels:
        raise ValueError(
            f"{missing_labels:,} ERP prediction rows "
            "have no matching ground-truth label."
        )

    unmatched_label_ids = set(
        labels["line_id"]
    ) - set(
        predictions["line_id"]
    )

    if unmatched_label_ids:
        raise ValueError(
            f"{len(unmatched_label_ids):,} label rows "
            "have no matching ERP prediction."
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    print("\nLABEL ALIGNMENT PASSED")
    print("-" * 55)

    print(
        "Matched ERP records:",
        f"{len(merged):,}"
    )

    print("All labels matched by line_id.")
    print("No labels used for inference.")

    return merged


# ============================================================
# 5. PREPARE EVALUATION COLUMNS
# ============================================================

def prepare_evaluation(df):

    result = df.copy()

    required = [
        "include",
        "spend",
        "confidence",
    ]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            "Missing required prediction columns: "
            f"{missing}"
        )

    result["include_normalized"] = (
        parse_include(result["include"])
    )

    result["spend_numeric"] = pd.to_numeric(
        result["spend"],
        errors="raise"
    )

    result["confidence_numeric"] = pd.to_numeric(
        result["confidence"],
        errors="raise"
    )

    if result["confidence_numeric"].isna().any():
        raise ValueError(
            "Some predictions have missing confidence."
        )

    if not result[
        "confidence_numeric"
    ].between(0, 1).all():
        raise ValueError(
            "Confidence values must be in [0, 1]."
        )

    for level in LEVELS:

        true_column = f"true_{level}"
        pred_column = f"pred_{level}"

        # Keep raw labels for human-readable reports.
        result[true_column] = (
            result[true_column]
            .astype("string")
            .fillna("")
            .str.strip()
        )

        result[pred_column] = (
            result[pred_column]
            .astype("string")
            .fillna("")
            .str.strip()
        )

        result[
            f"true_{level}_normalized"
        ] = (
            result[true_column]
            .apply(normalize_category)
        )

        result[
            f"pred_{level}_normalized"
        ] = (
            result[pred_column]
            .apply(normalize_category)
        )

        if result[
            f"true_{level}_normalized"
        ].eq("").any():
            raise ValueError(
                f"Ground-truth {level.upper()} "
                "contains blank values."
            )

        if result[
            f"pred_{level}_normalized"
        ].eq("").any():
            raise ValueError(
                f"Predicted {level.upper()} "
                "contains blank values."
            )

        result[
            f"correct_{level}"
        ] = (
            result[
                f"true_{level}_normalized"
            ]
            ==
            result[
                f"pred_{level}_normalized"
            ]
        )

    result["correct_exact_path"] = (
        result["correct_l1"]
        & result["correct_l2"]
        & result["correct_l3"]
    )

    result["confidence_band"] = pd.cut(
        result["confidence_numeric"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        include_lowest=True,
        right=True
    )

    result["needs_review"] = (
        result["confidence_numeric"]
        < CONFIG["review_threshold"]
    )

    if "classification_review" in result.columns:

        supplied_review = (
            parse_include(
                result["classification_review"]
            )
        )

        # Preserve the original Step 5 review flag
        # as a separate evaluation field.
        result["step5_review_flag"] = (
            supplied_review
        )

    else:
        result["step5_review_flag"] = (
            result["needs_review"]
        )

    return result


# ============================================================
# 6. OVERALL ACCURACY
# ============================================================

def calculate_overall_metrics(df):

    included = df.loc[
        df["include_normalized"]
    ]

    metrics = {
        "total_erp_records": int(
            len(df)
        ),

        "included_erp_records": int(
            len(included)
        ),

        "excluded_erp_records": int(
            len(df) - len(included)
        ),

        "average_confidence": float(
            df["confidence_numeric"].mean()
        ),

        "median_confidence": float(
            df["confidence_numeric"].median()
        ),

        "review_flagged_records": int(
            df["step5_review_flag"].sum()
        ),

        "labels_used_only_for_evaluation": True,

        "all_erp_records": {},

        "included_erp_only": {},

        "included_spend_weighted": {},
    }

    for level in LEVELS:

        metrics[
            "all_erp_records"
        ][f"{level}_accuracy"] = (
            evaluate_accuracy(
                df,
                level
            )
        )

        metrics[
            "included_erp_only"
        ][f"{level}_accuracy"] = (
            evaluate_accuracy(
                included,
                level
            )
        )

        metrics[
            "included_spend_weighted"
        ][f"{level}_accuracy"] = (
            spend_weighted_accuracy(
                df,
                f"correct_{level}"
            )
        )

    metrics[
        "all_erp_records"
    ]["exact_path_accuracy"] = float(
        df["correct_exact_path"].mean()
    )

    metrics[
        "included_erp_only"
    ]["exact_path_accuracy"] = (
        float(
            included["correct_exact_path"].mean()
        )
        if not included.empty
        else None
    )

    metrics[
        "included_spend_weighted"
    ]["exact_path_accuracy"] = (
        spend_weighted_accuracy(
            df,
            "correct_exact_path"
        )
    )

    return metrics


# ============================================================
# 7. CONFIDENCE-BAND ANALYSIS
# ============================================================

def calculate_confidence_bands(df):

    rows = []

    for band in CONFIDENCE_LABELS:

        subset = df.loc[
            df["confidence_band"]
            .astype("string")
            .eq(band)
        ]

        included = subset.loc[
            subset["include_normalized"]
        ]

        row = {
            "confidence_band": band,

            "records": int(
                len(subset)
            ),

            "included_records": int(
                len(included)
            ),

            "average_confidence": (
                float(
                    subset[
                        "confidence_numeric"
                    ].mean()
                )
                if not subset.empty
                else None
            ),

            "l1_accuracy": (
                float(
                    subset["correct_l1"].mean()
                )
                if not subset.empty
                else None
            ),

            "l2_accuracy": (
                float(
                    subset["correct_l2"].mean()
                )
                if not subset.empty
                else None
            ),

            "l3_accuracy": (
                float(
                    subset["correct_l3"].mean()
                )
                if not subset.empty
                else None
            ),

            "exact_path_accuracy": (
                float(
                    subset[
                        "correct_exact_path"
                    ].mean()
                )
                if not subset.empty
                else None
            ),

            "included_exact_path_accuracy": (
                float(
                    included[
                        "correct_exact_path"
                    ].mean()
                )
                if not included.empty
                else None
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. CATEGORY METRICS
# ============================================================

def calculate_category_metrics(df):

    all_rows = []

    for level in LEVELS:

        true_col = (
            f"true_{level}_normalized"
        )

        pred_col = (
            f"pred_{level}_normalized"
        )

        y_true = df[
            true_col
        ].to_numpy()

        y_pred = df[
            pred_col
        ].to_numpy()

        categories = sorted(
            set(y_true) | set(y_pred)
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=categories,
                zero_division=0
            )
        )

        predicted_counts = (
            pd.Series(y_pred)
            .value_counts()
        )

        for index, category in enumerate(
            categories
        ):

            all_rows.append({
                "level": level.upper(),

                "category": category,

                "precision": float(
                    precision[index]
                ),

                "recall": float(
                    recall[index]
                ),

                "f1_score": float(
                    f1[index]
                ),

                "true_support": int(
                    support[index]
                ),

                "predicted_count": int(
                    predicted_counts.get(
                        category,
                        0
                    )
                ),
            })

    return pd.DataFrame(
        all_rows
    )


# ============================================================
# 9. CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(df):

    paths = {}

    for level in LEVELS:

        true_col = (
            f"true_{level}_normalized"
        )

        pred_col = (
            f"pred_{level}_normalized"
        )

        y_true = df[
            true_col
        ].to_numpy()

        y_pred = df[
            pred_col
        ].to_numpy()

        categories = sorted(
            set(y_true) | set(y_pred)
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=categories
        )

        output = pd.DataFrame(
            matrix,
            index=categories,
            columns=categories
        )

        output.index.name = (
            "actual_category"
        )

        output.columns.name = (
            "predicted_category"
        )

        path = (
            REPORT_DIR
            / f"confusion_matrix_{level}.csv"
        )

        output.to_csv(path)

        paths[level] = str(path)

    return paths


# ============================================================
# 10. ERROR ANALYSIS
# ============================================================

def save_error_reports(df):

    errors = df.loc[
        ~df["correct_exact_path"]
    ].copy()

    errors = errors.sort_values(
        [
            "confidence_numeric",
            "line_id",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable"
    )

    preferred_columns = [
        "line_id",
        "material_short_text",
        "service_short_text",
        "classification_text",
        "supplier_key",
        "item_key",
        "include",
        "spend",
        "true_l1",
        "pred_l1",
        "true_l2",
        "pred_l2",
        "true_l3",
        "pred_l3",
        "confidence",
        "classification_method",
        "classification_review",
        "correct_l1",
        "correct_l2",
        "correct_l3",
        "correct_exact_path",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in errors.columns
    ]

    errors[
        columns
    ].head(
        CONFIG["error_examples"]
    ).to_csv(
        ERRORS_OUTPUT,
        index=False
    )

    low_confidence = df.sort_values(
        [
            "confidence_numeric",
            "line_id",
        ],
        ascending=[
            True,
            True,
        ],
        kind="stable"
    )

    low_confidence[
        columns
    ].head(
        CONFIG["low_confidence_examples"]
    ).to_csv(
        LOW_CONFIDENCE_OUTPUT,
        index=False
    )

    return int(
        len(errors)
    )


# ============================================================
# 11. PRINT RESULTS
# ============================================================

def print_accuracy_table(metrics):

    print("\nOVERALL CLASSIFICATION ACCURACY")
    print("-" * 65)

    all_records = metrics[
        "all_erp_records"
    ]

    included = metrics[
        "included_erp_only"
    ]

    weighted = metrics[
        "included_spend_weighted"
    ]

    def percentage(value):

        if value is None:
            return "N/A"

        return f"{value * 100:.2f}%"

    print(
        f"{'Metric':<22}"
        f"{'All ERP':>13}"
        f"{'Included ERP':>15}"
        f"{'Spend-weighted':>16}"
    )

    print("-" * 65)

    for level in LEVELS:

        key = f"{level}_accuracy"

        print(
            f"{level.upper() + ' accuracy':<22}"
            f"{percentage(all_records[key]):>13}"
            f"{percentage(included[key]):>15}"
            f"{percentage(weighted[key]):>16}"
        )

    key = "exact_path_accuracy"

    print(
        f"{'Exact path accuracy':<22}"
        f"{percentage(all_records[key]):>13}"
        f"{percentage(included[key]):>15}"
        f"{percentage(weighted[key]):>16}"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("COMPANY A - CLASSIFICATION EVALUATION")
    print("=" * 65)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged = load_data()

    evaluated = prepare_evaluation(
        merged
    )

    metrics = calculate_overall_metrics(
        evaluated
    )

    print_accuracy_table(
        metrics
    )

    confidence_bands = (
        calculate_confidence_bands(
            evaluated
        )
    )

    category_metrics = (
        calculate_category_metrics(
            evaluated
        )
    )

    confusion_paths = (
        save_confusion_matrices(
            evaluated
        )
    )

    error_count = save_error_reports(
        evaluated
    )

    # Complete row-level evaluation.
    evaluated.to_csv(
        PREDICTIONS_OUTPUT,
        index=False
    )

    confidence_bands.to_csv(
        CONFIDENCE_OUTPUT,
        index=False
    )

    category_metrics.to_csv(
        CATEGORY_OUTPUT,
        index=False
    )

    summary = {
        "evaluation_dataset": "Company A ERP",

        "labels_used_only_for_evaluation": True,

        "prediction_source": str(
            PREDICTIONS_FILE
        ),

        "label_source": str(
            LABELS_FILE
        ),

        "metrics": metrics,

        "error_analysis": {
            "incorrect_exact_paths": (
                error_count
            ),

            "correct_exact_paths": int(
                evaluated[
                    "correct_exact_path"
                ].sum()
            ),
        },

        "configuration": CONFIG,

        "outputs": {
            "evaluated_predictions": str(
                PREDICTIONS_OUTPUT
            ),

            "errors": str(
                ERRORS_OUTPUT
            ),

            "low_confidence": str(
                LOW_CONFIDENCE_OUTPUT
            ),

            "confidence_bands": str(
                CONFIDENCE_OUTPUT
            ),

            "category_metrics": str(
                CATEGORY_OUTPUT
            ),

            "confusion_matrices": (
                confusion_paths
            ),
        },
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=False
        )

    print("\nCONFIDENCE BAND ACCURACY")
    print("-" * 65)

    print(
        confidence_bands[
            [
                "confidence_band",
                "records",
                "l1_accuracy",
                "l3_accuracy",
                "exact_path_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nERROR ANALYSIS")
    print("-" * 65)

    print(
        "Incorrect exact paths:",
        f"{error_count:,}"
    )

    print(
        "Correct exact paths:",
        f"{evaluated['correct_exact_path'].sum():,}"
    )

    print("\nFILES SAVED")
    print("-" * 65)

    print(SUMMARY_FILE)
    print(PREDICTIONS_OUTPUT)
    print(ERRORS_OUTPUT)
    print(LOW_CONFIDENCE_OUTPUT)
    print(CONFIDENCE_OUTPUT)
    print(CATEGORY_OUTPUT)

    for path in confusion_paths.values():
        print(path)

    print("\nEVALUATION COMPLETE")
    print("No classification predictions were modified.")
    print("Ground-truth labels were used only here.")


if __name__ == "__main__":
    main()
