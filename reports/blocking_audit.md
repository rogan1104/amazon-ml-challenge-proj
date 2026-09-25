# Blocking Audit Report

**Generated:** 2026-09-25 20:15:54
**Scope:** Amazon ML Challenge 2026 — Business Entity Resolution (training GT)

---

## 1. Existing Blocking Architecture

### Discovery Result

**Teammate production blocking implementation: NOT FOUND**

Exhaustive repository scan performed. Expected locations checked:

- `MISSING: /Users/rohan/Downloads/student_resource/code`
- `MISSING: /Users/rohan/Downloads/student_resource/output/candidate_pairs.tsv`
- `MISSING: /Users/rohan/Downloads/student_resource/src`
- `NOTE: analysis/phase10_blocking.py is forensics simulation, NOT production blocking`
- `NOTE: utils/validate_submission.py validates format only`

**Python files matching blocking keywords:** 0

**candidate_pairs.tsv outputs found:** 0

### What exists in this repository

| Component | Path | Role |
|-----------|------|------|
| Submission validator | `utils/validate_submission.py` | Validates `candidate_pairs.tsv` format; does NOT generate candidates |
| Forensics simulation | `analysis/phase10_blocking.py` | **Audit-only** baseline simulation from dataset forensics; NOT production blocking |
| Challenge spec | `README.md` | Defines expected `candidate_pairs.tsv` output format |

### OBSERVATION

No `code/business_entity_resolution/`, no `output/candidate_pairs.tsv`, and no runnable blocking/retrieval module was found. **There is currently nothing to audit as teammate production blocking.**

To complete a teammate audit when code is available, re-run:
```bash
.venv/bin/python -m analysis.blocking_audit --candidate output/candidate_pairs.tsv
```

---

## 2. Recall Results (Reference Baselines — Combined Simple Blocking)

Since no teammate blocking exists, we report **reference baseline E** (norm_name OR norm_addr OR country+prefix5) as the measured ceiling for simple rules.

| Target | True Pairs Recalled | Recall (of all 7.64M) | Recall (of target pairs) |
|--------|--------------------:|----------------------:|-------------------------:|
| S1→S2 | {s2_row.get('true_matches_recalled', 'N/A'):,} | {s2_row.get('recall', 0)*100:.2f}% | {s2_row.get('recall_per_target_only', 0)*100:.2f}% |
| S1→S3 | {s3_row.get('true_matches_recalled', 'N/A'):,} | {s3_row.get('recall', 0)*100:.2f}% | {s3_row.get('recall_per_target_only', 0)*100:.2f}% |
| **Overall S2+S3 union** | **{combined.get('true_matches_recalled', 'N/A'):,}** | **{combined.get('recall', 0)*100:.2f}%** | — |

**~23% of true match pairs are NOT retrieved** by even the best simple combined rule.

---

## 3. Candidate Volume (Baseline E)

| Metric | S2 | S3 | Combined |
|--------|----|----|----------|
| Total candidates | {s2_row.get('total_candidates', 0):,} | {s3_row.get('total_candidates', 0):,} | {combined.get('total_candidates', 0):,} |
| Avg candidates/S1 | {s2_row.get('avg_candidates_per_s1', 0)} | {s3_row.get('avg_candidates_per_s1', 0)} | {combined.get('avg_candidates_per_s1', 0)} |
| P95 candidates/S1 | {s2_row.get('p95_candidates_per_s1', 0)} | {s3_row.get('p95_candidates_per_s1', 0)} | — |
| Max candidates/S1 | {s2_row.get('max_candidates_per_s1', 0):,} | {s3_row.get('max_candidates_per_s1', 0):,} | — |
| S1 with zero candidates | {s2_row.get('s1_zero_candidates', 0):,} ({s2_row.get('s1_zero_pct', 0)}%) | {s3_row.get('s1_zero_candidates', 0):,} ({s3_row.get('s1_zero_pct', 0)}%) | — |
| Reduction ratio | {s2_row.get('reduction_ratio', 0)} | {s3_row.get('reduction_ratio', 0)} | — |

---

## 4. Runtime

Baseline evaluation runtime (DuckDB key-aggregation, in-memory):

---

## 5. Recall by Source

See baseline comparison table in `blocking_audit.csv`. Key finding: **S3 recall (39.8%) slightly exceeds S2 (37.5%)** under combined simple blocking, despite S3 having noisier addresses in true matches.

---

## 6. Recall by Match Difficulty / Match Count Bucket

- **S2 / 1_match**: 75.48% recall (44,347/58,750 pairs)
- **S2 / 2-3_matches**: 76.82% recall (875,112/1,139,104 pairs)
- **S2 / 4+_matches**: 77.98% recall (1,946,247/2,495,765 pairs)
- **S3 / 1_match**: 77.28% recall (46,680/60,407 pairs)
- **S3 / 2-3_matches**: 77.14% recall (928,588/1,203,843 pairs)
- **S3 / 4+_matches**: 76.96% recall (2,062,873/2,680,496 pairs)

---

## 7. Recall by Country (Combined Baseline)

- **India / S2**: 62.13% (919,911/1,480,545)
- **US / S2**: 87.92% (1,945,795/2,213,074)
- **US / S3**: 85.03% (2,011,277/2,365,448)
- **India / S3**: 65.02% (1,026,864/1,579,298)

---

## 8. Failure Categories (Missed True Matches — Baseline E)

Sample of missed pairs (first 150 extracted):

| Category | Count in Sample |
|----------|----------------|
| transliteration/multilingual | 31 |
| missing_address | 7 |
| abbreviation/legal_suffix | 4 |
| common_name/short_name | 4 |
| semantic_dba_or_other | 2 |
| minor_token_diff/typo | 1 |
| word_order | 1 |

### Representative Missed Examples

| S1 ID | Match ID | Category | S1 Name | Other Name |
|-------|----------|----------|---------|------------|
| S1-627736066 | S3-575071279 | minor_token_diff/typo | Invictus & Co | Co & Invimsmtus |
| S1-144788927 | S3-650657239 | transliteration/multilingual | Southern Consulting Limited | सदर्न कंसल्टिंग लिमिटेड |
| S1-272351912 | S3-65710282 | abbreviation/legal_suffix | DEW Spire Corp | Lumjax |
| S1-869658100 | S3-610588056 | transliteration/multilingual | Laxmi Investments Private Limited | लक्ष्मी इन्वेस्टमेंट्स प्राइवेट लिम |
| S1-50142763 | S2-959466871 | common_name/short_name | Advance & Co | Belojax |
| S1-727424687 | S3-436095307 | transliteration/multilingual | Eastern Seven Technology Private Li | ઈસ્ટર્ન સેવન ટેક્નોલોજી પ્રાઇવેટ લિ |
| S1-796967925 | S3-140339931 | missing_address | Mmk Fashions Pvt Ltd | Mmk Ltd Pvt [Fashions] |
| S1-795047694 | S2-195311814 | semantic_dba_or_other | Elsinore Fisher Total Bce | Zetaarcveo |
| S1-265392356 | S2-644844388 | transliteration/multilingual | Tirupati Impex Pvt Ltd | તિરુપતિ ઇમ્પેક્સ પ્રા. લિ. |
| S1-397955896 | S2-671724890 | transliteration/multilingual | Best Impex Private Limited | बेस्ट इम्पेक्स प्राइवेट लिमिटेड |
| S1-574668405 | S2-28438203 | transliteration/multilingual | Modern Ventures Private Limited | मॉडर्न वेंचर्स प्राइवेट लिमिटेड |
| S1-536583156 | S2-770363267 | transliteration/multilingual | United Enterprises Private Limited | யுனைடெட் எண்டர்பிரைசஸ் பிரைவேட் லிம |
| S1-602087272 | S3-854772376 | transliteration/multilingual | Lakshmi Power Private Limited | लक्ष्मी पावर प्राइवेट लिमिटेड |
| S1-161249692 | S3-653123686 | missing_address | Omega & Sons Limited | Omega + |
| S1-481927230 | S2-47424255 | abbreviation/legal_suffix | Raviraj Club Limited | NEXVEO |

---

## 9. Candidate Quality (Combined Baseline E)

| Metric | Value |
|--------|-------|
| Total candidate pairs (S2+S3) | 12,843,832,657 |
| True positives in candidates | 5,903,847 |
| False positives in candidates | 12,837,928,810 |
| True positive rate | 0.0460% |
| False positive rate | 99.95% |
| Candidates per true match | 1681.49 |

**OBSERVATION:** Matcher receives **millions of mostly irrelevant candidates** (~100.0% false) even after blocking reduction. Precision-heavy F_0.5 requires the matcher to reject >99% of candidates correctly.

---

## 10. Baseline Comparison

| Strategy | Target | Recall | Avg Cand/S1 | P95 | Max | Zero-S1 % |
|----------|--------|--------|-------------|-----|-----|-----------|
| A_exact_norm_name_S2 | S2 | 6.68% | 7.85 | 38.0 | 414 | 57.2398% |
| A_exact_norm_name_S3 | S3 | 6.96% | 8.05 | 38.0 | 425 | 55.5181% |
| name_prefix_5_S2 | S2 | 36.15% | 3447.43 | 12723.0 | 41,215 | 0.7011% |
| name_prefix_5_S3 | S3 | 37.87% | 3811.14 | 14364.0 | 49,404 | 0.704% |
| name_prefix_10_S2 | S2 | 26.18% | 814.1 | 7125.0 | 29,297 | 7.8451% |
| name_prefix_10_S3 | S3 | 26.99% | 835.03 | 7046.0 | 29,498 | 7.8782% |
| D_name_first_token_S2 | S2 | 31.84% | 2964.16 | 9561.0 | 41,035 | 0.2327% |
| D_name_first_token_S3 | S3 | 33.44% | 3313.51 | 10277.0 | 51,992 | 0.2344% |
| country_+_name_prefix5_S2 | S2 | 36.15% | 2800.54 | 10066.0 | 41,204 | 0.8205% |
| country_+_name_prefix5_S3 | S3 | 37.87% | 3067.94 | 10238.0 | 49,393 | 0.8296% |
| country_+_norm_name_S2 | S2 | 6.68% | 7.83 | 38.0 | 414 | 57.2767% |
| country_+_norm_name_S3 | S3 | 6.96% | 8.0 | 38.0 | 425 | 55.5482% |
| B_exact_norm_address_S2 | S2 | 5.41% | 1.47 | 3.0 | 19 | 84.3532% |
| B_exact_norm_address_S3 | S3 | 2.24% | 1.04 | 1.0 | 6 | 91.0569% |
| country_+_norm_addr_S2 | S2 | 5.41% | 1.47 | 3.0 | 19 | 84.3532% |
| country_+_norm_addr_S3 | S3 | 2.24% | 1.04 | 1.0 | 6 | 91.0569% |
| E_combined_name_or_addr_or_cprefix_S2 | S2 | 37.52% | 2777.57 | 0 | 41,204 | 0.7412% |
| E_combined_name_or_addr_or_cprefix_S3 | S3 | 39.77% | 3042.5 | 0 | 49,393 | 0.7623% |

---

## 11. Recall-vs-Candidate Curve (Top-K Block Size Cap)

Uses minimum blocking-key collision count per true pair as proxy for candidate limit K.

| K limit | S2 recall@K | S3 recall@K | S2 p95 block size | S3 p95 block size |
|---------|------------|------------|-------------------|-------------------|
| 100 | 73.6% | 71.8% | 6171.0 | 7053.1 |
| 250 | 79.3% | 77.4% | 6171.0 | 7053.1 |
| 500 | 82.4% | 80.8% | 6171.0 | 7053.1 |
| 1000 | 85.0% | 83.7% | 6171.0 | 7053.1 |
| 2500 | 88.5% | 86.2% | 6171.0 | 7053.1 |
| 5000 | 92.6% | 90.9% | 6171.0 | 7053.1 |
| 10000 | 98.2% | 97.6% | 6171.0 | 7053.1 |

**OBSERVATION:** Aggressive K caps cause steep recall drops for name-prefix blocking because common prefixes (e.g. "sai", "new") create block sizes >>10K.

---

## 12. France / Domain Generalization Risks

- Training GT covers **US + India only**; France is 15% of test S1 (~259K entities)
- Combined baseline uses **country as a blocking key** — will still work for France but with no train-time recall measurement
- **Transliteration/multilingual** failures dominate missed matches — likely worse for France
- Rules assuming US address formats (ZIP, state abbrev) may underperform on French addresses
- **Do not hard-code country filters to {US, India}** per challenge rules

---

## 13. Final Assessment

### BLOCKING STATUS: **REPLACE**

**Justification (measured evidence):**
1. **No production blocking implementation exists** in the repository to keep or improve
2. Reference simple combined blocking achieves only **~77% pair recall** — missing ~1.73M true pairs irrecoverably if used as-is
3. Candidate volume remains **~12.8B total pairs** across S2+S3 under combined blocking — matcher must process enormous false-positive load
4. **57% of S1 entities get zero candidates** under exact-name-only blocking (unacceptable)
5. Hard positives (transliteration, DBA names, abbreviation) systematically missed

---

### CURRENT RECALL (no teammate system — baseline E reference):
- **S2 = {s2_row.get('recall_per_target_only', 0)*100:.2f}%** (of S2 true pairs)
- **S3 = {s3_row.get('recall_per_target_only', 0)*100:.2f}%** (of S3 true pairs)
- **Overall = {combined.get('recall', 0)*100:.2f}%** (of all 7.64M true pairs)

### CANDIDATE COST (baseline E):
- **Average = {combined.get('avg_candidates_per_s1', 0)}** candidates/S1 (S2+S3 combined)
- **P95 = ~{max(s2_row.get('p95_candidates_per_s1', 0), s3_row.get('p95_candidates_per_s1', 0))}**
- **P99 = see blocking_audit.csv**
- **Maximum = {max(s2_row.get('max_candidates_per_s1', 0), s3_row.get('max_candidates_per_s1', 0)):,}**

### BIGGEST FAILURE MODE:
**Transliteration/multilingual name variants and DBA/semantic name differences** — blocking keys on normalized ASCII text cannot join Devanagari/Gujarati/ASCII transliteration variants of the same business.

### RECOMMENDED NEXT STEP:
1. **Obtain/commit teammate blocking code** or `output/candidate_pairs.tsv` on training data
2. Re-run this audit: `.venv/bin/python -m analysis.blocking_audit --candidate <path>`
3. Until then, **design new blocking** targeting >95% recall with:
   - Multi-key union (name trigrams + address tokens + postal digits)
   - Transliteration-normalization for Indic scripts
   - Separate S2 vs S3 tuning (S3 address noise is higher)
   - Candidate cap with fallback secondary retrieval for zero-candidate S1

---

*End of Blocking Audit Report*
