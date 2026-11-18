# APPROACH — Spend Analytics Pipeline

## Design and execution

The solution processes Company A and Company B through separate, reproducible entry points in `pipeline.py`. Each run normalizes source records, determines eligible spend, resolves supplier/item identifiers, assigns categories from the supplied taxonomy, validates the output, and writes `runs/<company>/final_spend.csv` plus a pipeline summary. Excluded source records are retained for auditability with zero eligible spend. The two companies have different source schemas, so Company B uses a dedicated adapter rather than applying Company A's invoice rules unchanged.

## Matching and de-duplication

**Company A:** ERP goods receipts and invoice-tool lines may represent the same purchase. Invoice gross values are converted to net values using VAT (15% default where blank). Matching uses purchase-order references, amount tolerance of 0.05, and description similarity of at least 0.85. Matching is one-to-one, with ERP records taking precedence. Cancelled, unapproved, and matched duplicate invoice lines are excluded from eligible spend but remain in the audit output.

**Company B:** Invoice lines come from eight ERP systems. Amounts are converted to USD using the posting-month rate (`USD = line_amount / units_per_usd`). Strict, uniquely matched opposite-sign transactions are treated as reversal pairs; ambiguous candidates are flagged, and unmatched negative credits remain. Invalid amounts are excluded. Document-type exclusions are not inferred without an authoritative business mapping.

## Classification into an unseen taxonomy

The classifier reads the provided taxonomy at runtime, builds candidate paths, and matches normalized transaction text using TF-IDF similarity, fuzzy text similarity, and item-consistency rules. It does not rely on hard-coded category names or read label files during inference. Predictions are restricted to valid taxonomy paths. Weak or uncertain predictions are flagged for review; confidence scores are heuristic, not calibrated probabilities. A new taxonomy file can be supplied, but a truly new source schema may require a normalization adapter.

## Measured accuracy (labels kept out of inference)

Company A predictions were evaluated separately against `labels_A.csv`, after classification. Measured accuracy: **L1 40.38%**, **L2 21.19%**, **L3/exact taxonomy path 15.71%**, and **spend-weighted exact path 23.44%**. These figures show substantial error at fine-grained levels. Company B's `seed_labels.csv` was not used for inference; a separate accuracy result for B has not yet been measured. The upcoming blind-company test is therefore not represented as completed.

## Runtime

Observed local end-to-end runtime: **Company A 15.81 s** (65,549 records) and **Company B 7.7 s** (41,028 records). These are single observed runs, not standardized benchmarks. No external inference API is used.

## Limitations

Category descriptions and transaction text are often sparse, contributing to low exact-path accuracy and high review volumes. Company B document-type semantics and ambiguous reversal candidates need business validation. The pipeline has been run on A and B; unseen-company schema compatibility and blind-test performance remain unverified.
