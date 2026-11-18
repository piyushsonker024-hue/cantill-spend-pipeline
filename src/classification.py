"""Taxonomy classification v2: inference only; never loads ground-truth labels.

Run: python -m src.classification
Optional configurable paths:
  python -m src.classification --taxonomy data/raw/taxonomy_B.csv \
      --erp data/processed/erp_resolved.csv \
      --invoices data/processed/invoices_resolved.csv \
      --output-dir data/processed --report-dir reports

This is a conservative taxonomy-text baseline, not a calibrated ML model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "review_threshold": 0.40,
    "minimum_similarity": 0.08,
    "consistency_min_similarity": 0.22,
    "consistency_min_margin": 0.035,
    "consistency_min_token_overlap": 0.18,
    "batch_size": 2048,
    "word_weight": 0.45,
    "char_weight": 0.35,
    "keyword_weight": 0.20,
}

# Generic technical vocabulary only; no Company A taxonomy paths or labels.
ABBREVIATIONS = {
    "assy": "assembly", "assm": "assembly", "assmbly": "assembly",
    "brg": "bearing", "brng": "bearing", "hyd": "hydraulic",
    "elec": "electrical", "elect": "electrical", "mech": "mechanical",
    "mtr": "motor", "mot": "motor", "eng": "engine", "pmp": "pump",
    "vlv": "valve", "cyl": "cylinder", "hsg": "housing",
    "gbox": "gearbox", "gbx": "gearbox", "flt": "filter",
    "filt": "filter", "bkt": "bracket", "bsh": "bush",
    "sft": "shaft", "shft": "shaft", "blt": "bolt",
    "scr": "screw", "nut": "nut", "rbr": "rubber",
    "pvc": "pvc", "dia": "diameter", "qty": "quantity",
    "spares": "spare", "repairs": "repair", "bits": "bit",
    "miners": "miner", "mining": "mine", "drilling": "drill",
    "ducting": "duct", "hoses": "hose", "bolts": "bolt",
    "lhd": "load haul dumper", "cm": "continuous miner",
    "rh": "road header", "pn": "", "partno": "",
}

STOPWORDS = {
    "the", "and", "for", "with", "from", "each", "per", "of", "to",
    "a", "an", "in", "on", "or", "supply", "supplied", "new", "old",
    "item", "items", "part", "parts", "number", "no", "unit", "units",
    "set", "sets", "kit", "kits", "type", "size", "mm", "cm", "kg",
    "m", "x", "ea", "pc", "pcs", "misc", "general",
}


def norm_header(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_level(columns, level):
    n = str(level)
    aliases = {
        f"l{n}", f"level{n}", f"categoryl{n}",
        f"category{n}", f"categorylevel{n}", f"lvl{n}",
    }
    for col in columns:
        if norm_header(col) in aliases:
            return col
    return None


def clean_value(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_text(value):
    """Generic normalization, independent of evaluation labels."""
    text = clean_value(value).lower()
    text = re.sub(r"\b(?:p\s*[/.-]?\s*n|part\s*(?:no|number))\b", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = []
    for token in text.split():
        # Serial/part-number tokens are usually poor taxonomy evidence.
        if token.isdigit() or (len(token) >= 6 and any(c.isdigit() for c in token)):
            continue
        expansion = ABBREVIATIONS.get(token, token)
        for word in expansion.split():
            if len(word) > 1 and word not in STOPWORDS:
                tokens.append(word)
    return " ".join(tokens)


def token_set(value):
    return set(normalize_text(value).split())


def load_taxonomy(path):
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {path}")
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    print("Taxonomy columns:", raw.columns.tolist())
    cols = {f"l{i}": find_level(raw.columns, i) for i in range(1, 5)}
    if any(cols[f"l{i}"] is None for i in (1, 2, 3)):
        raise ValueError(f"Taxonomy needs L1, L2, L3 columns. Found: {raw.columns.tolist()}")
    taxonomy = pd.DataFrame({
        level: raw[col].map(clean_value) if col is not None
        else pd.Series([""] * len(raw), dtype="string")
        for level, col in cols.items()
    })
    taxonomy = taxonomy.drop_duplicates().reset_index(drop=True)
    if taxonomy.empty or taxonomy[["l1", "l2", "l3"]].eq("").any().any():
        raise ValueError("Taxonomy contains empty mandatory paths or no rows")
    taxonomy["taxonomy_path_id"] = np.arange(len(taxonomy), dtype=int)
    taxonomy["path_text"] = taxonomy[["l1", "l2", "l3", "l4"]].agg(" ".join, axis=1)
    taxonomy["path_normalized"] = taxonomy["path_text"].map(normalize_text)
    taxonomy["leaf_normalized"] = taxonomy["l3"].map(normalize_text)
    taxonomy["path_tokens"] = taxonomy["path_normalized"].map(lambda x: set(x.split()))
    print(f"Valid taxonomy paths: {len(taxonomy):,}")
    print("L1 / L2 / L3 / L4:", *[taxonomy[f"l{i}"].replace("", pd.NA).nunique() for i in range(1, 5)])
    return taxonomy


def load_records(erp_path, invoice_path):
    frames = []
    for source, path in (("ERP", erp_path), ("INVOICE", invoice_path)):
        if not path.exists():
            raise FileNotFoundError(f"Missing resolved file: {path}")
        df = pd.read_csv(path, low_memory=False, dtype={
            "line_id": "string", "tool_record_id": "string",
            "item_key": "string", "supplier_key": "string",
            "po_number_normalized": "string",
        })
        required = {"include", "spend", "supplier_key", "item_key"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{source} missing required columns: {sorted(missing)}")
        df["_source_group"] = source
        df["_source_order"] = np.arange(len(df))
        frames.append(df)
        print(f"{source} records: {len(df):,}")
    return pd.concat(frames, ignore_index=True, sort=False), [len(f) for f in frames]


def make_description(df):
    def field(name):
        if name not in df:
            return pd.Series([""] * len(df), index=df.index, dtype="string")
        return df[name].astype("string").fillna("").str.strip()
    erp = (field("material_short_text") + " " + field("service_short_text")).str.strip()
    inv = field("line_text")
    raw = erp.where(df["_source_group"].eq("ERP"), inv)
    return raw.fillna("").astype("string")


def build_similarity(taxonomy, texts):
    """Taxonomy-only TF-IDF with a conservative token-overlap component."""
    path_texts = taxonomy["path_normalized"].tolist()
    if not any(path_texts):
        raise ValueError("No usable taxonomy text after normalization")
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    word.fit(path_texts)
    char.fit(path_texts)
    path_word = word.transform(path_texts)
    path_char = char.transform(path_texts)
    path_tokens = taxonomy["path_tokens"].tolist()
    leaf_tokens = [set(x.split()) for x in taxonomy["leaf_normalized"]]
    n = len(texts)
    predictions = []
    for start in range(0, n, CONFIG["batch_size"]):
        batch = texts[start:start + CONFIG["batch_size"]]
        word_sim = linear_kernel(word.transform(batch), path_word).astype(np.float32)
        char_sim = linear_kernel(char.transform(batch), path_char).astype(np.float32)
        for row_idx, desc in enumerate(batch):
            tokens = set(desc.split())
            overlap = np.zeros(len(path_texts), dtype=np.float32)
            if tokens:
                for j, (all_tokens, leaf) in enumerate(zip(path_tokens, leaf_tokens)):
                    # Reward coverage of actual description terms, especially the leaf.
                    shared = len(tokens & all_tokens)
                    leaf_shared = len(tokens & leaf)
                    overlap[j] = min(1.0, 0.7 * shared / len(tokens) +
                                     0.3 * leaf_shared / max(1, len(leaf)))
            scores = (CONFIG["word_weight"] * word_sim[row_idx] +
                      CONFIG["char_weight"] * char_sim[row_idx] +
                      CONFIG["keyword_weight"] * overlap)
            order = np.argsort(-scores, kind="stable")
            best_idx = int(order[0])
            best = float(scores[best_idx])
            second = float(scores[order[1]]) if len(order) > 1 else 0.0
            margin = max(0.0, best - second)
            best_overlap = float(overlap[best_idx])
            # Conservative heuristic, not a probability. Penalize ambiguous paths.
            confidence = float(np.clip(
                0.65 * best + 0.20 * min(1.0, margin * 3.0) +
                0.15 * best_overlap, 0.0, 1.0))
            no_match = not tokens or best < CONFIG["minimum_similarity"]
            if no_match:
                confidence = 0.0
            predictions.append({
                "classification_text": desc,
                "taxonomy_path_id": best_idx,
                "similarity_score": best,
                "second_best_score": second,
                "similarity_margin": margin,
                "keyword_overlap": best_overlap,
                "confidence": confidence,
                "classification_method": "NO_TEXT_MATCH" if no_match else "HYBRID_TAXONOMY",
                "classification_review": bool(no_match or confidence < CONFIG["review_threshold"]),
            })
    return pd.DataFrame(predictions)


def apply_item_consistency(df, taxonomy):
    """Change a path only with independent text evidence and an unambiguous item group.

    Do not copy the representative's confidence. Scores for a changed path are
    reset to zero and the record is always flagged for manual review.
    """
    df = df.copy()
    df["row_similarity_score"] = df["similarity_score"]
    df["row_taxonomy_path_id"] = df["taxonomy_path_id"]
    df["item_consistency_applied"] = False
    keys = df["item_key"].astype("string").fillna("").str.strip()
    df["_item_key_clean"] = keys
    for key, idx in df.loc[keys.ne("")].groupby("_item_key_clean", sort=False).groups.items():
        if len(idx) < 2:
            continue
        group = df.loc[idx]
        # Only consider groups with a clear, independently supported winner.
        strong = group.loc[
            (group["similarity_score"] >= CONFIG["consistency_min_similarity"]) &
            (group["similarity_margin"] >= CONFIG["consistency_min_margin"]) &
            (group["classification_method"] == "HYBRID_TAXONOMY")
        ]
        if strong.empty:
            continue
        # Conflicting strong predictions: do not force consensus.
        if strong["taxonomy_path_id"].nunique() != 1:
            continue
        target = int(strong["taxonomy_path_id"].iloc[0])
        target_tokens = taxonomy.iloc[target]["path_tokens"]
        for row_idx, row in group.iterrows():
            if int(row["taxonomy_path_id"]) == target:
                continue
            words = set(str(row["classification_text"]).split())
            overlap = len(words & target_tokens) / max(1, len(words))
            if overlap < CONFIG["consistency_min_token_overlap"]:
                continue
            df.at[row_idx, "taxonomy_path_id"] = target
            df.at[row_idx, "item_consistency_applied"] = True
            df.at[row_idx, "confidence"] = 0.0
            df.at[row_idx, "similarity_score"] = 0.0
            df.at[row_idx, "similarity_margin"] = 0.0
            df.at[row_idx, "second_best_score"] = 0.0
            df.at[row_idx, "classification_method"] = "ITEM_CONSISTENCY_REVIEW"
            df.at[row_idx, "classification_review"] = True
    return df.drop(columns=["_item_key_clean"])


def include_mask(values):
    values = values.astype("string").fillna("").str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n"}
    bad = set(values.unique()) - allowed
    if bad:
        raise ValueError(f"Unexpected include values: {bad}")
    return values.isin({"true", "1", "yes", "y"})


def validate(before, after, taxonomy):
    if len(before) != len(after):
        raise AssertionError("Record count changed")
    if before["_source_group"].tolist() != after["_source_group"].tolist():
        raise AssertionError("Source ordering changed")
    for col in ("line_id", "tool_record_id", "supplier_key", "item_key", "exclusion_reason"):
        if col in before and col in after:
            a = before[col].astype("string").fillna("").tolist()
            b = after[col].astype("string").fillna("").tolist()
            if a != b:
                raise AssertionError(f"Changed original column: {col}")
    if not np.array_equal(include_mask(before["include"]), include_mask(after["include"])):
        raise AssertionError("Eligibility changed")
    a = pd.to_numeric(before["spend"], errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(after["spend"], errors="raise").to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=0, atol=1e-7, equal_nan=True):
        raise AssertionError("Spend changed")
    if not after["confidence"].between(0, 1).all():
        raise AssertionError("Confidence outside [0,1]")
    valid = set(map(tuple, taxonomy[["l1", "l2", "l3", "l4"]].itertuples(index=False, name=None)))
    observed = set(map(tuple, after[["l1", "l2", "l3", "l4"]].itertuples(index=False, name=None)))
    if not observed.issubset(valid):
        raise AssertionError("Invalid taxonomy hierarchy")
    if after[["l1", "l2", "l3"]].isna().any().any():
        raise AssertionError("Missing mandatory taxonomy prediction")
    print("VALIDATION PASSED: rows, eligibility, spend, keys, hierarchy, confidence")


def main():
    parser = argparse.ArgumentParser(description="Inference-only taxonomy classification")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "data/raw/taxonomy_A.csv")
    parser.add_argument("--erp", type=Path, default=ROOT / "data/processed/erp_resolved.csv")
    parser.add_argument("--invoices", type=Path, default=ROOT / "data/processed/invoices_resolved.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("TAXONOMY CLASSIFICATION V2 - NO LABELS USED")
    print("=" * 65)
    taxonomy = load_taxonomy(args.taxonomy)
    original, sizes = load_records(args.erp, args.invoices)
    work = original.copy()
    work["classification_text_raw"] = make_description(work)
    work["classification_text"] = work["classification_text_raw"].map(normalize_text)
    descriptions = work["classification_text"].drop_duplicates().tolist()
    print(f"Unique normalized descriptions: {len(descriptions):,}")
    predictions = build_similarity(taxonomy, descriptions)
    work = work.merge(predictions, on="classification_text", how="left", sort=False, validate="many_to_one")
    work = work.sort_values(["_source_group", "_source_order"], kind="stable")
    # Restore original concatenation order, independent of alphabetical source labels.
    work = apply_item_consistency(work, taxonomy)
    path_lookup = taxonomy.set_index("taxonomy_path_id")[["l1", "l2", "l3", "l4"]]
    work = work.join(
    path_lookup,
    on="taxonomy_path_id",
    validate="many_to_one",
    suffixes=("", "_taxonomy")
    )
    for level in ("l1", "l2", "l3", "l4"):
        if f"{level}_taxonomy" in work.columns:
            work[level] = work.pop(f"{level}_taxonomy")
    # Order by original source row order, using a numeric source rank.
    work["_rank"] = work["_source_group"].map({"ERP": 0, "INVOICE": 1})
    work = work.sort_values(["_rank", "_source_order"], kind="stable").reset_index(drop=True)
    original = original.reset_index(drop=True)
    validate(original, work, taxonomy)
    included = include_mask(work["include"])
    spend = pd.to_numeric(work["spend"], errors="raise").fillna(0.0)
    summary = {
        "total_records": int(len(work)),
        "included_records": int(included.sum()),
        "review_flagged_records": int(work["classification_review"].sum()),
        "average_confidence": float(work["confidence"].mean()),
        "unique_predicted_taxonomy_paths": int(work["taxonomy_path_id"].nunique()),
        "zero_or_weak_match_records": int(work["classification_method"].eq("NO_TEXT_MATCH").sum()),
        "item_consistency_review_records": int(work["item_consistency_applied"].sum()),
        "included_net_spend_zar": float(spend.loc[included].sum()),
        "labels_used_for_inference": False,
        "configuration": CONFIG,
    }
    audit_cols = ["_source_group", "line_id", "tool_record_id", "item_key", "supplier_key",
                  "classification_text_raw", "classification_text", "l1", "l2", "l3", "l4",
                  "taxonomy_path_id", "row_taxonomy_path_id", "similarity_score", "row_similarity_score",
                  "second_best_score", "similarity_margin", "keyword_overlap", "confidence",
                  "classification_review", "classification_method", "item_consistency_applied", "include", "spend"]
    work[[c for c in audit_cols if c in work.columns]].to_csv(args.output_dir / "classification_audit.csv", index=False)
    clean = work.drop(columns=["_source_group", "_source_order", "_rank"])
    clean.iloc[:sizes[0]].to_csv(args.output_dir / "erp_classified.csv", index=False)
    clean.iloc[sizes[0]:].to_csv(args.output_dir / "invoices_classified.csv", index=False)
    (args.report_dir / "classification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION SUMMARY")
    for key, value in summary.items():
        if key != "configuration":
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    print("\nSaved classified ERP, invoices, audit and JSON summary.")


if __name__ == "__main__":
    main()
