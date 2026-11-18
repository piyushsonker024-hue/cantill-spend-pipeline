# DECISIONS — Findings and Engineering Choices

This log records observed data characteristics, the corresponding implementation choices, and unresolved assumptions. Figures below are from the reported local pipeline runs, not independently audited business totals.

## Company A

| Finding | Decision and rationale |
|---|---|
| ERP goods receipts and invoice-tool lines overlap | Match across sources, use one-to-one deduplication, and prefer ERP receipts to avoid double counting. |
| ERP values are net; invoice-tool amounts are gross | Convert invoice gross to net using VAT before comparing amounts; use 15% VAT when missing. |
| Invoice records include cancelled and unapproved statuses | Exclude those from eligible spend, but preserve their source rows and exclusion reasons. |
| Similar descriptions can match multiple transactions | Require purchase-order context, amount tolerance (0.05), and description similarity (>=0.85); do not allow unrestricted many-to-one matching. |
| Supplier and item representations vary | Create normalized supplier and item keys for aggregation and consistency checks. |
| Company A taxonomy has 40 valid paths | Classify only into paths present in the supplied taxonomy. |
| Held-out evaluation shows low fine-grained accuracy | Report actual metrics and flag uncertain predictions instead of presenting confidence as calibrated. |

**Observed run:** 65,549 total records; 52,468 included; 13,081 excluded; **ZAR 3,620,518,805.92** included net spend. Company A exact-path classification accuracy was **15.71%** (spend-weighted: **23.44%**).

## Company B

| Finding | Decision and rationale |
|---|---|
| 41,028 invoice lines from eight JDE/SAP ERP systems | Use a separate Company B source adapter, preserving original line IDs. |
| Multiple transaction currencies and monthly `units_per_usd` rates | Convert to USD using posting month and division by the applicable rate; validate rates. |
| All observed currency/posting-month combinations had FX entries | Continue checking FX completeness at runtime rather than assuming future files are complete. |
| 833 negative amounts and 918 zero amounts | Do not treat sign alone as proof of invalidity; preserve unmatched credits and assess reversals explicitly. |
| Opposite-sign transactions sometimes appear to offset | Exclude only strict, uniquely identified reversal pairs; flag ambiguous candidates for review. |
| Three amounts are invalid or missing | Exclude from eligible spend with an auditable reason. |
| ERP-specific document types (`PV`, `RE`, `PD`, etc.) have uncertain semantics | Avoid unverified document-type exclusions. |
| Supplier names are not available in the observed source columns | Use available vendor IDs, scoped by source where appropriate, rather than inventing supplier names. |
| Descriptions are incomplete; material and GL fields provide additional signals | Use available text and material identifiers for item resolution and classification, with review flags for weak evidence. |
| Company B taxonomy has 40 paths across three levels | Read the taxonomy dynamically; do not assume Company A's category names apply. |
| `seed_labels.csv` contains reference annotations | Keep it out of inference; reserve it for a separate evaluation. |

**Observed run:** 41,028 total records; 40,879 included; 149 excluded (146 reversal-pair records, three invalid amounts); **USD 345,622,273.68** included spend; zero missing FX; 182 ambiguous-reversal review records; 36,578 review-flagged records overall.

## Cross-company choices and trade-offs

- **Auditability over silent filtering:** preserve source rows, record eligibility decisions, and set excluded eligible spend to zero.
- **Deterministic processing:** use local code and company-specific output directories (`runs/A`, `runs/B`).
- **No label leakage:** classification uses transactions and supplied taxonomy, not evaluation labels.
- **Conservative exclusion:** where business semantics are uncertain, retain or flag rather than automatically discard.
- **Explicit limitations:** taxonomy validity and arithmetic checks do not prove classification correctness or business-rule completeness.

## Outstanding validation

1. Evaluate Company B predictions against `seed_labels.csv` without using those labels to tune the evaluated run.
2. Test with the genuinely unseen company when blind-test data arrives; verify schema adaptation, one-command execution, totals, and taxonomy validity.
3. Confirm Company B document-type and reversal policies with authoritative definitions.
4. Improve category matching and calibrate confidence using a properly separated development/evaluation protocol.
