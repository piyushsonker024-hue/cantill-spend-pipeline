# Cantill.ai Spend Analytics Pipeline

## Overview

This project implements a reproducible spend analytics pipeline for two companies with different ERP and procurement data structures.

The pipeline covers data normalization, transaction eligibility, supplier and item entity resolution, spend classification, and validation.

Company A and Company B are processed independently, with their outputs stored in separate run directories.

## Project Structure

```text
cantill-spend-pipeline/
├── pipeline.py
├── requirements.txt
├── README.md
├── src/
│   ├── normalization.py
│   ├── eligibility.py
│   ├── entity_resolution.py
│   ├── classification.py
│   ├── company_b_adapter.py
│   └── evaluation.py
├── data/
│   └── raw/
│       ├── erp_goods_receipts.csv
│       ├── invoice_tool_export.csv
│       ├── taxonomy_A.csv
│       ├── labels_A.csv
│       ├── invoice_lines.csv
│       ├── fx_rates_usd.csv
│       ├── taxonomy_B.csv
│       └── seed_labels.csv
└── runs/
    ├── A/
    │   ├── final_spend.csv
    │   └── reports/
    └── B/
        ├── final_spend.csv
        └── reports/
```

## Requirements

Python 3.14 and the dependencies listed in `requirements.txt`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running Company A

Company A combines ERP goods receipts with invoice-tool records.

```bash
python pipeline.py \
  --company A \
  --erp data/raw/erp_goods_receipts.csv \
  --invoices data/raw/invoice_tool_export.csv \
  --taxonomy data/raw/taxonomy_A.csv \
  --force
```

Output:

`runs/A/final_spend.csv`

### Company A processing

1. Normalize ERP and invoice-tool records.
2. Convert invoice gross amounts to net amounts using VAT.
3. Apply eligibility rules and deduplicate overlapping ERP/invoice records.
4. Generate supplier and item entity keys.
5. Classify included spend using the supplied taxonomy.
6. Validate output consistency.

ERP records take precedence over matched invoice-tool records during deduplication.

## Running Company B

Company B contains invoice-line data from eight ERP source systems.

```bash
python pipeline.py \
  --company B \
  --lines data/raw/invoice_lines.csv \
  --fx data/raw/fx_rates_usd.csv \
  --taxonomy data/raw/taxonomy_B.csv \
  --force
```

Output:

`runs/B/final_spend.csv`

### Company B processing

1. Read and normalize invoice-line records.
2. Convert transaction amounts to USD using monthly exchange rates.
3. Identify uniquely matched reversal pairs.
4. Preserve unmatched negative transactions for further analysis.
5. Generate supplier and item keys.
6. Classify transactions using Company B's taxonomy.
7. Validate row counts, currency conversion, eligibility and taxonomy paths.

Currency conversion uses:

`USD amount = local currency amount / units_per_usd`

The posting month determines the FX rate.

Document types are not excluded automatically without a verified business-rule mapping.

## Pipeline Results

### Company A

| Metric | Result |
|---|---:|
| Total records | 65,549 |
| Included records | 52,468 |
| Excluded records | 13,081 |
| Included net spend | ZAR 3,620,518,805.92 |
| Taxonomy paths used | 40 |
| Review-flagged records | 53,983 |

### Company B

| Metric | Result |
|---|---:|
| Total records | 41,028 |
| Included records | 40,879 |
| Excluded records | 149 |
| Included net spend | USD 345,622,273.68 |
| Reversal-pair records | 146 |
| Invalid-amount records | 3 |
| Missing FX records | 0 |
| Ambiguous reversal review records | 182 |
| Taxonomy paths used | 40 |
| Review-flagged records | 36,578 |

## Classification

The classification stage uses taxonomy-driven text matching and item consistency rules.

Predictions are restricted to valid paths in the company-specific taxonomy.

Classification confidence is heuristic rather than statistically calibrated. Low-confidence and unresolved predictions are flagged for review.

## Evaluation

Reference labels are not used during inference.

Company A was evaluated separately against `labels_A.csv`.

| Metric | Result |
|---|---:|
| Level 1 accuracy | 40.38% |
| Level 2 accuracy | 21.19% |
| Level 3 / exact-path accuracy | 15.71% |
| Spend-weighted exact-path accuracy | 23.44% |

These results show that classification accuracy remains a significant limitation.

Company B's `seed_labels.csv` is reserved for evaluation and is not used to generate predictions.

## Validation

The pipeline checks record preservation, transaction identifiers, eligibility consistency, spend treatment, entity keys and taxonomy validity.

Company B additionally validates USD conversion and FX coverage.

Validation passing means the output satisfies the implemented consistency checks; it does not establish that every transaction is correctly classified.

## Limitations and Future Improvements

- Classification accuracy is limited by inconsistent and incomplete transaction descriptions.
- Confidence scores are heuristic and require calibration.
- Company B document-type eligibility rules require business confirmation.
- Reversal matching is conservative; ambiguous cases require manual review.
- Supplier and item resolution can be improved using additional master-data mappings.
- Company B seed-label evaluation remains to be completed.

## Reproducibility

Each company can be run independently using the commands above.

Generated results are stored under `runs/A/` and `runs/B/`, preventing Company A and Company B outputs from overwriting each other.

The supplied reference labels are not used for inference.