"""Step 7: isolated runner for legacy Cantill spend-pipeline stages.

Run from the project root:
  python pipeline.py --company A --erp data/raw/erp_goods_receipts.csv \
    --invoices data/raw/invoice_tool_export.csv --taxonomy data/raw/taxonomy_A.csv

Does not import labels or execute src.evaluation. Each run is isolated in
runs/<company>/, so legacy hardcoded paths cannot overwrite other runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
STAGES = ("normalization", "eligibility", "entity_resolution")
REQUIRED_ERP = {"line_id", "source_system", "spend_type", "po_number", "posting_date", "value_zar", "supplier_name"}
REQUIRED_INVOICE = {"tool_record_id", "invoice_number", "supplier_name_as_captured", "line_text", "gross_amount", "status"}
OUTPUT_FIELDS = ("record_id", "record_source", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence", "classification_review", "classification_method")


def fail(message):
    raise RuntimeError(message)


def headers(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return set(next(csv.reader(f)))


def check_inputs(erp, invoices, taxonomy):
    for path in (erp, invoices, taxonomy):
        if not path.is_file():
            fail(f"Missing input: {path}")
    for label, path, required in (("ERP", erp, REQUIRED_ERP), ("invoice", invoices, REQUIRED_INVOICE)):
        missing = required - headers(path)
        if missing:
            fail(f"{label} input lacks columns required by legacy stages: {sorted(missing)}. "
                 "Company B needs a schema adapter before running these modules.")
    tax_cols = {c.lower().replace("_", "").replace(" ", "") for c in headers(taxonomy)}
    for n in (1, 2, 3):
        if not ({f"l{n}", f"categoryl{n}", f"level{n}", f"category{n}"} & tax_cols):
            fail(f"Taxonomy is missing level {n} column")


def count_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in f) - 1


def bools(series, name):
    values = series.astype("string").fillna("").str.strip().str.lower()
    yes = {"true", "1", "yes", "y"}
    no = {"false", "0", "no", "n"}
    bad = set(values.unique()) - yes - no
    if bad:
        fail(f"Invalid boolean values in {name}: {sorted(bad)}")
    return values.isin(yes)


def run_stage(workspace, module, extra=()):
    cmd = [sys.executable, "-m", f"src.{module}", *map(str, extra)]
    print("\n" + "=" * 64, flush=True)
    print("RUNNING:", " ".join(cmd), flush=True)
    start = time.monotonic()
    completed = subprocess.run(cmd, cwd=workspace, check=False)
    if completed.returncode:
        fail(f"Stage {module} failed (exit {completed.returncode}); inspect output above. "
             f"Intermediate files are in {workspace}")
    print(f"Stage {module}: {time.monotonic() - start:.1f}s", flush=True)


def build_final(workspace, erp_count, invoice_count):
    processed = workspace / "data" / "processed"
    parts = []
    for source, filename, id_column, expected in (
        ("ERP", "erp_classified.csv", "line_id", erp_count),
        ("INVOICE", "invoices_classified.csv", "tool_record_id", invoice_count),
    ):
        path = processed / filename
        if not path.exists():
            fail(f"Missing classified output: {path}")
        frame = pd.read_csv(path, low_memory=False, dtype={
            "line_id": "string", "tool_record_id": "string",
            "supplier_key": "string", "item_key": "string"})
        if len(frame) != expected:
            fail(f"{source}: expected {expected:,} rows, found {len(frame):,}")
        required = {"include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "confidence", id_column}
        missing = required - set(frame.columns)
        if missing:
            fail(f"{source} classified output missing {sorted(missing)}")
        ids = frame[id_column].astype("string").fillna("").str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            fail(f"{source}: blank or duplicate record identifiers")
        frame["record_id"] = ("ERP:" if source == "ERP" else "INVOICE:") + ids
        frame["record_source"] = source
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    if combined["record_id"].duplicated().any():
        fail("Combined record IDs are not unique")
    included = bools(combined["include"], "include")
    spend = pd.to_numeric(combined["spend"], errors="raise")
    confidence = pd.to_numeric(combined["confidence"], errors="raise")
    if spend.isna().any() or ~pd.Series(confidence.between(0, 1)).all():
        fail("Missing spend or invalid confidence")
    if (spend.loc[~included].abs() > 1e-7).any():
        fail("Excluded records must have zero spend")
    reasons = combined["exclusion_reason"].astype("string").fillna("").str.strip()
    if reasons.loc[~included].eq("").any():
        fail("Excluded records missing exclusion_reason")
    for key in ("supplier_key", "item_key"):
        if combined[key].astype("string").fillna("").str.strip().eq("").any():
            fail(f"Missing {key}")

    # Read the selected taxonomy; never read evaluation labels.
    from_taxonomy = pd.read_csv(workspace / "data/raw/taxonomy_A.csv", dtype="string", keep_default_na=False)
    normalized = {"".join(c.lower().split()).replace("_", ""): c for c in from_taxonomy.columns}
    def level(n):
        for alias in (f"l{n}", f"categoryl{n}", f"level{n}", f"category{n}", f"categorylevel{n}"):
            if alias in normalized:
                return from_taxonomy[normalized[alias]].fillna("").astype("string").str.strip()
        return pd.Series([""] * len(from_taxonomy), dtype="string")
    valid_paths = set(zip(*(level(i).tolist() for i in range(1, 5))))
    for name in ("l1", "l2", "l3", "l4"):
        if name not in combined:
            if name == "l4":
                combined[name] = ""
            else:
                fail(f"Missing predicted {name}")
    predicted_paths = set(zip(*(combined[f"l{i}"].astype("string").fillna("").str.strip().tolist() for i in range(1, 5))))
    if not predicted_paths.issubset(valid_paths):
        fail(f"Invalid taxonomy paths: {list(predicted_paths - valid_paths)[:3]}")

    for optional in ("classification_review", "classification_method"):
        if optional not in combined:
            combined[optional] = False if optional == "classification_review" else ""
    final = combined[list(OUTPUT_FIELDS)].copy()
    final["include"] = included
    final["spend"] = spend
    final["confidence"] = confidence
    out = workspace / "final_spend.csv"
    final.to_csv(out, index=False)
    # Preserve source-specific classified files and audit for traceability.
    summary = {
        "total_records": int(len(final)),
        "erp_records": int(erp_count),
        "invoice_records": int(invoice_count),
        "included_records": int(included.sum()),
        "excluded_records": int((~included).sum()),
        "included_net_spend_zar": float(spend.loc[included].sum()),
        "excluded_spend_zar": float(spend.loc[~included].sum()),
        "taxonomy_paths_used": int(len(predicted_paths)),
        "review_flagged_records": int(bools(final["classification_review"], "classification_review").sum()),
        "labels_used_for_inference": False,
        "output": str(out),
    }
    return out, summary


def main():
    parser = argparse.ArgumentParser(description="Run isolated Cantill spend pipeline; inference only")
    parser.add_argument("--company", required=True, help="Run label, e.g. A or B")
    parser.add_argument("--erp", type=Path)
    parser.add_argument("--invoices", type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--lines", type=Path, help="Company B invoice_lines.csv")
    parser.add_argument("--fx", type=Path, help="Company B fx_rates_usd.csv")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--force", action="store_true", help="Replace an existing company run")
    args = parser.parse_args()
    if not args.company.replace("-", "").replace("_", "").isalnum():
        fail("Company name must contain only letters, numbers, hyphens or underscores")
    if args.company.upper() == "B":
        if not args.lines or not args.fx:
            fail("Company B requires --lines and --fx, plus --taxonomy")
        lines, fx, taxonomy_b = (p.expanduser().resolve() for p in (args.lines, args.fx, args.taxonomy))
        for path in (lines, fx, taxonomy_b):
            if not path.is_file():
                fail(f"Missing Company B input: {path}")
        source = PROJECT / "src"
        for module in ("classification", "company_b_adapter"):
            if not (source / f"{module}.py").is_file():
                fail(f"Missing source module: {source / (module + '.py')}")
        workspace = args.runs_dir.expanduser().resolve() / args.company
        if workspace.exists():
            if not args.force:
                fail(f"Run already exists: {workspace}. Use --force to replace it")
            if workspace == PROJECT or workspace in PROJECT.parents or workspace == source:
                fail("Refusing to delete project source")
            shutil.rmtree(workspace)
        for folder in ("src", "data/raw", "data/processed", "reports"):
            (workspace / folder).mkdir(parents=True, exist_ok=True)
        for module in ("classification", "company_b_adapter"):
            shutil.copy2(source / f"{module}.py", workspace / "src" / f"{module}.py")
        (workspace / "src/__init__.py").touch(exist_ok=True)
        shutil.copy2(lines, workspace / "data/raw/invoice_lines.csv")
        shutil.copy2(fx, workspace / "data/raw/fx_rates_usd.csv")
        shutil.copy2(taxonomy_b, workspace / "data/raw/taxonomy_B.csv")
        print(f"Company B: {count_csv(lines):,} invoice lines", flush=True)
        print(f"Isolated workspace: {workspace}", flush=True)
        start = time.monotonic()
        run_stage(workspace, "company_b_adapter", (
            "--lines", workspace / "data/raw/invoice_lines.csv",
            "--fx", workspace / "data/raw/fx_rates_usd.csv",
            "--taxonomy", workspace / "data/raw/taxonomy_B.csv",
            "--output-dir", workspace / "data/processed",
            "--report-dir", workspace / "reports",
        ))
        summary_path = workspace / "reports/pipeline_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["runtime_seconds"] = round(time.monotonic() - start, 2)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Company B complete: {workspace / 'final_spend.csv'}")
        return
    if not args.erp or not args.invoices:
        fail("Company A requires --erp and --invoices, plus --taxonomy")
    erp, invoices, taxonomy = (p.expanduser().resolve() for p in (args.erp, args.invoices, args.taxonomy))
    check_inputs(erp, invoices, taxonomy)
    source = PROJECT / "src"
    for module in (*STAGES, "classification"):
        if not (source / f"{module}.py").is_file():
            fail(f"Missing stage source: {source / (module + '.py')}")
    # Never copy labels, evaluation.py, or the user's original processed files.
    workspace = args.runs_dir.expanduser().resolve() / args.company
    if workspace.exists():
        if not args.force:
            fail(f"Run already exists: {workspace}. Use --force to replace it")
        if workspace == PROJECT or workspace in PROJECT.parents or workspace == PROJECT / "src" or workspace == source:
            fail("Refusing to delete project source")
        shutil.rmtree(workspace)
    (workspace / "data/raw").mkdir(parents=True)
    (workspace / "data/processed").mkdir(parents=True)
    (workspace / "reports").mkdir(parents=True)
    (workspace / "src").mkdir(parents=True)
    # Copy all Python helpers in src, but not evaluation.py; legacy imports work.
    for file in source.glob("*.py"):
        if file.name not in {"evaluation.py", "duplicate_audit.py"}:
            shutil.copy2(file, workspace / "src" / file.name)
    (workspace / "src/__init__.py").touch(exist_ok=True)
    shutil.copy2(erp, workspace / "data/raw/erp_goods_receipts.csv")
    shutil.copy2(invoices, workspace / "data/raw/invoice_tool_export.csv")
    shutil.copy2(taxonomy, workspace / "data/raw/taxonomy_A.csv")
    erp_count, invoice_count = count_csv(erp), count_csv(invoices)
    print(f"Company {args.company}: ERP={erp_count:,}, invoices={invoice_count:,}", flush=True)
    print(f"Isolated workspace: {workspace}", flush=True)
    start = time.monotonic()
    for module in STAGES:
        run_stage(workspace, module)
    run_stage(workspace, "classification", (
        "--taxonomy", workspace / "data/raw/taxonomy_A.csv",
        "--erp", workspace / "data/processed/erp_resolved.csv",
        "--invoices", workspace / "data/processed/invoices_resolved.csv",
        "--output-dir", workspace / "data/processed",
        "--report-dir", workspace / "reports",
    ))
    output, summary = build_final(workspace, erp_count, invoice_count)
    summary.update({"company": args.company, "runtime_seconds": round(time.monotonic() - start, 2),
                    "input_erp": str(erp), "input_invoices": str(invoices), "input_taxonomy": str(taxonomy)})
    report = workspace / "reports/pipeline_summary.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nVALIDATION PASSED: record counts, IDs, eligibility, spend, keys, hierarchy, confidence")
    print(f"Final CSV: {output}")
    print(f"Report: {report}")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"input_erp", "input_invoices", "input_taxonomy", "output"}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
