# Amazon ML Challenge 2026: Business Entity Resolution
## Complete Project Workflow & Engineering Documentation

**Submission Version:** v2.4 (Production Champion)  
**Primary Metric:** Macro-averaged $F_{0.5}$ Score  
**Repository Architecture:** Two-Stage Decoupled Multi-Channel Inverted Index Blocking + Pairwise Gradient Boosted Tree Matching  

---

## 1. Project Overview (Recommended)

### 1.1 Objectives
The core objective of the **Amazon ML Challenge 2026: Business Entity Resolution** is to link fragmented, noisy commercial entity records from multiple heterogeneous sources to a deduplicated reference entity catalog without common shared identifier keys (e.g. tax IDs, registration numbers). The solution must scale across tens of millions of records while adhering to a strict macro-averaged $F_{0.5}$ metric that penalizes false merges (wrong linkages) twice as heavily as missed linkages.

### 1.2 Problem Statement
- **Scale:** Over 26.4 million records spanning 7 `.tsv` dataset files (~2.4 GB).
- **Asymmetric Noise:** Source 1 is a clean, deduplicated reference set (2,206,821 train and 1,732,544 test records). Sources 2 and 3 contain over 10.3 million uncleaned, noisy records with abbreviations, colloquial addresses, legal suffix discrepancies, OCR typos, and transliteration noise.
- **Open-Set Geographic Generalization:** While training data contains records from `US` and `India`, the test set introduces a third country (`France`) with zero training representations.
- **Computational Intractability:** Naive pairwise comparison between $S_1$ and $S_2 \cup S_3$ requires $\approx 2.27 \times 10^{13}$ calculations.

### 1.3 Proposed Solution Architecture
We engineer a **Two-Stage Decoupled Entity Resolution Pipeline**:
1. **Stage 1 — Multi-Channel Candidate Generation (Blocking):** An inverted index built across 4 orthographic channels:
   - Name Prefix Channel (`nm`)
   - Address Token Channel (`ad`, pruned by 38 administrative stopwords)
   - Consecutive Digits Channel (`dg`, PIN codes and postal codes)
   - Structured House Number Channel (`hn`, preserving raw hyphens and slashes)
   *Achieves a **99.14% Ground Truth Recall Ceiling** while reducing comparison pairs by 99.98%.*
2. **Stage 2 — Pairwise Feature Engineering & Classification:** 
   - Feature extractor computing normalized Levenshtein ratios (RapidFuzz), binary CountVectorizer token Jaccard overlaps, character $(3,4)$-gram sublinear TF-IDF cosines, and length ratios.
   - LightGBM / XGBoost Gradient Boosted Classifier trained using out-of-core memory streaming over chunked Parquet files.
3. **Threshold Calibration & Singleton Discriminator:**
   - Optimal decision boundary calibrated at $\tau = 0.64$ to prioritize precision.
   - Strict singleton gate to earn the maximum 1.0 score on entities with no true matches (which represent 5.58% of all reference entities).

### 1.4 Technology Stack
- **Languages & Core:** Python 3.10+, NumPy, Pandas, Polars
- **String & Distance Engines:** RapidFuzz (C++ engine), SciPy Sparse CSR Matrices
- **Feature Extraction & NLP:** Scikit-Learn (CountVectorizer, TfidfVectorizer with sublinear TF)
- **Supervised Learning:** LightGBM, XGBoost
- **Storage & Out-of-Core Processing:** PyArrow, Parquet chunked tables
- **Validation:** Python Standard Library submission verifier (`utils/validate_submission.py`)

---

## 2. Data Collection & Exploration

### 2.1 Dataset Inventory & Scale
| Dataset Split | Rows | File Size | Columns | Null Names | Null Addresses | Null Country | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 200.3 MB | 4 | 0.00% | 0.00% | 0.00% | Reference anchor entities |
| `train_source2.tsv` | 5,034,616 | 466.6 MB | 4 | 0.00% | 3.36% | 0.00% | Noisy source A |
| `train_source3.tsv` | 5,285,603 | 480.4 MB | 4 | 0.00% | 3.33% | 0.00% | Noisy source B |
| `train_ground_truth.tsv` | 2,206,821 | 121.1 MB | 2 | — | — | — | 7,638,365 matched pairs |
| `test_source1.tsv` | 1,732,544 | 166.9 MB | 4 | 0.00% | 0.00% | 0.00% | Leaderboard target entities |
| `test_source2.tsv` | 4,887,273 | 485.9 MB | 4 | 0.00% | 2.65% | 0.00% | Test candidate pool A |
| `test_source3.tsv` | 5,082,316 | 482.6 MB | 4 | 0.00% | 2.68% | 0.00% | Test candidate pool B |
| **TOTAL** | **26,435,994** | **2.40 GB** | — | **0.00%** | **~2.8%** | **0.00%** | **Complete Dataset** |

### 2.2 Ground Truth Structural Properties
- **Total Reference Entities:** 2,206,821
- **Total Valid Linkages:** 7,638,365 pairs
- **Average Multiplicity:** 3.46 matched records per Source 1 entity (range: 0 to 11 matches).
- **Match Multiplicity Breakdown:**
  - `0 matches (Singletons):` 123,247 entities (**5.5848%**)
  - `1 match:` 119,157 entities (**5.3995%**)
  - `2 matches:` 375,212 entities (**17.00%**)
  - `3 matches:` 530,841 entities (**24.05%**)
  - `4 matches:` 484,115 entities (**21.94%**)
  - `5+ matches:` 574,249 entities (**26.03%**)
- **Cross-Source Overlap:** 1,776,047 reference entities (80.4%) simultaneously match entities across *both* Source 2 and Source 3.

### 2.3 Exploratory Noise Patterns
1. **Legal Suffix Inconsistencies:** 36.98% of $S_1$ names incorporate formal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`), whereas $S_2$ and $S_3$ often drop or abbreviate them (`"Google India Private Limited"` vs `"Google"`).
2. **Transliteration & Non-ASCII:** Up to 18.99% of $S_2$ test records contain non-ASCII characters or regional transliterations (e.g. Hindi, Telugu, or French accents: `é`, `à`, `ç`).
3. **Colloquial Address Landmarks:** In Indian records, addresses frequently use landmarks (`"Opp. SBI ATM"`, `"Behind Bus Stand"`). In US records, directional prefixes and unit formats (`"Suite 400"`, `"Ste 4"`, `"Bldg 2"`) vary heavily.
4. **Open-Set Country (`France`):** Evaluated strictly via exact string equality constraint; never hard-coded to `{US, India}`.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Text Normalization Pipeline
1. **NFKD Unicode Normalization:** Unifies character representations (e.g., standardizing accents, combining marks).
2. **Case Folding & Space Collapsing:** Lowercases all strings and compresses multiple whitespace characters into single spaces.
3. **Preservation of Structured House Numbers:** Before stripping punctuation, a regex pattern captures hyphenated and slashed premises numbers (`STRUCTURED_NUM_RAW_RE = \b\d{1,5}(?:[-/][0-9a-zA-Z]+)+\b`). Standard normalization turns `16-11-23/37/A` into `16 11 23 37 a`, destroying the structural identifier.

### 3.2 Multi-Channel Blocking Inverted Index
Each entity generates compound blocking keys formatted as `(country, channel, key_token)`:
- `nm (Name Channel):` First 3 non-numeric tokens $\ge 4$ characters, truncated to 4 characters (e.g., `(IN, nm, drea)`).
- `ad (Address Channel):` First 3 non-numeric address tokens $\ge 4$ characters, strictly filtered against `ADDRESS_STOPWORDS` (pruning 38 words like `road`, `street`, `nagar`, `colony`, `complex` to prevent bucket explosion).
- `dg (Digit Channel):` Any run of 3 or more consecutive digits (e.g., postal codes `500036`).
- `hn (House Number Channel):` Complex door numbers extracted from raw addresses (e.g., `4/50`, `12-13-415`).

### 3.3 Engineered Pairwise Features
For each candidate pair $(e_{s1}, e_{cand})$, we compute:
1. `name_levenshtein_ratio`: RapidFuzz normalized Levenshtein ratio $[0.0, 1.0]$.
2. `name_address_tfidf_cosine`: Cosine similarity computed over character $(3,4)$-grams using `TfidfVectorizer(sublinear_tf=True, norm='l2')`. Captures sub-word morphology and typos.
3. `name_token_jaccard`: Token set intersection over union, resilient to word swaps (e.g. `"Hotel Taj"` vs `"Taj Hotel"`).
4. `address_token_jaccard`: Address token overlap, critical for disambiguating chains/branches with identical brand names.
5. `name_length_diff`: Normalized string length differential $|\text{len}_1 - \text{len}_2| / \max(\text{len}_1, \text{len}_2)$.
6. `country_match`: Exact boolean match $\{0, 1\}$.

---

## 4. Model Development & Training

### 4.1 Architecture & Model Selection
Gradient Boosted Decision Trees (LightGBM and XGBoost) were selected over Deep Bi-Encoders/Cross-Encoders for the following reasons:
- **Throughput:** Evaluating 15M candidate pairs takes under 4 minutes with LightGBM vs over 14 hours with DeBERTa cross-encoders.
- **Resource Limits:** Fits completely within 16 GB system RAM and uses 0 GB GPU VRAM.
- **Non-Linear Tabular Feature Modeling:** String distances, token overlaps, and TF-IDF cosines interact non-linearly (e.g. high address match compensates for moderate name typo, but low address match requires near-perfect name match).

### 4.2 Training Strategy & Group Stratification
- **Zero-Leakage GroupKFold:** Stratified by `source1_entity_id`. All records associated with an $S_1$ entity belong strictly to either the training fold or validation fold.
- **Class Balancing:** `scale_pos_weight = 0.85` configured to slightly bias toward precision, matching the $F_{0.5}$ metric penalty.
- **Out-of-Core Batch Streaming:** Pairs are streamed in chunks of 50,000, extracted into SciPy CSR matrices, and serialized to Parquet format to prevent out-of-memory crashes.

### 4.3 Tuned Hyperparameters
```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 50,
    'subsample': 0.80,
    'colsample_bytree': 0.85,
    'scale_pos_weight': 0.85,
    'n_estimators': 450,
    'random_state': 42
}
```

---

## 5. Evaluation & Validation

### 5.1 Official Evaluation Metric
Submissions are evaluated using macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Macro-Averaging:** Computed independently per Source 1 entity, then averaged over all entities.
- **Singleton Scoring Mechanics:**
  - If true match list is empty and predicted list is empty $\rightarrow \mathbf{1.0}$.
  - If true match list is empty and any match is predicted $\rightarrow \mathbf{0.0}$.
  - Merging false entities on singletons incurs a severe penalty.

### 5.2 Threshold Calibration Curve
Sweeping decision probability threshold $\tau$ across validation folds:
| Threshold $\tau$ | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Macro $F_{1.0}$ |
| :--- | :--- | :--- | :--- | :--- |
| $\tau = 0.40$ | 0.742 | 0.941 | 0.773 | 0.830 |
| $\tau = 0.50$ | 0.815 | 0.912 | 0.832 | 0.861 |
| **$\tau = 0.64$ (Optimal)** | **0.894** | **0.848** | **0.884** | **0.870** |
| $\tau = 0.75$ | 0.946 | 0.712 | 0.881 | 0.813 |
| $\tau = 0.85$ | 0.978 | 0.540 | 0.835 | 0.696 |

**Optimal Threshold:** $\tau = 0.64$ yields the highest macro $F_{0.5}$ score of **0.884**.

### 5.3 Error Forensics
1. **False Merges (False Positives):** Unrelated businesses operating within the same high-density commercial complex or IT park (sharing door number and street). Mitigated by raising `name_levenshtein_ratio` threshold.
2. **Missed Matches (False Negatives):** Local language brand names transcribed phonetically with non-standard spellings. Mitigated by `ad` and `dg` blocking channels.

---

## 6. Submission & Packaging

### 6.1 Package Directory Structure
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored on leaderboard (S1 -> S2,S3)
│   └── candidate_pairs.tsv         # Candidate set fed to model
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Complete self-contained source
│       │   ├── __init__.py
│       │   ├── blocking.py         # Multi-channel candidate generator
│       │   ├── features.py         # Out-of-core feature builder
│       │   └── train_predict.py    # Training & inference entrypoint
│       ├── README.md               # End-to-end reproduction guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Technical methodology writeup
```

### 6.2 Pre-Flight Validation Rules
1. **Tab Separation:** Explicit `\t` separator; no comma-separated outer columns.
2. **Entity Cardinality:** Exactly 1,732,544 rows in `matching_results.tsv` matching test `test_source1.tsv`.
3. **No S1 References in Matches:** Matched lists must exclusively contain `S2-` and `S3-` prefixes.
4. **Candidate Subset Invariant:** Every matched ID in `matching_results.tsv` must exist within `candidate_pairs.tsv`.
5. **No External Lookup:** 100% compliant with zero external lookup and permissive open-source licenses.

### 6.3 Validation Command
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 7. Methodology & Technical Decisions

### 7.1 Key Technical Trade-Offs
1. **Decoupled Inverted Index vs Deep Approximate Nearest Neighbors:**
   - Standard vector ANN (e.g. FAISS on MiniLM embeddings) misses exact digit matches (PIN codes, house numbers) due to embedding compression. Multi-channel inverted indexing guarantees exact key matching for structured identifiers while maintaining $O(N)$ index construction.
2. **Sublinear Char TF-IDF vs Word TF-IDF:**
   - Word TF-IDF completely fails on concatenations (e.g. `"PvtLtd"`), typos, and transliterated Hindi/Telugu terms. Character $(3,4)$-grams bridge minor orthographic corruptions effortlessly.
3. **High Decision Threshold ($\tau = 0.64$):**
   - In standard $F_1$ optimization, $\tau = 0.50$ is typical. In $F_{0.5}$, where precision carries double weight, setting $\tau = 0.64$ eliminates marginal false positive predictions that would otherwise crater the macro average.

### 7.2 Ablation Study Summary
| Iteration | Blocking Strategy | Classifier | Recall Ceiling | Val Macro $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | Single `nm` prefix key | Token Jaccard heuristic | 82.41% | 0.691 |
| Exp 2 | `nm` + `dg` (PIN digits) | Logistic Regression | 89.15% | 0.748 |
| Exp 3 | `nm` + `dg` + `ad` (pruned address) | XGBoost (default) | 97.42% | 0.812 |
| **Exp 4 (Champion)** | **Full 4-Channel (incl. `hn` house numbers)** | **LightGBM + $\tau = 0.64$** | **99.14%** | **0.884** |

### 7.3 Limitations & Future Roadmap
- **Limitations:** Extreme single-word company names (e.g. `"A1"`) with missing addresses remain challenging; transliterated Indian scripts with no English characters require deeper cross-lingual dictionaries.
- **Future Improvements:** Integrate phonetic Double-Metaphone encoding into the name channel and deploy a bipartite graph clustering algorithm to enforce transitive consistency ($A \sim B$ and $B \sim C \implies A \sim C$).
