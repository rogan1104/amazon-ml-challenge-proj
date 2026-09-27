# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an industrial two-stage entity resolution pipeline engineered to link 26.4 million heterogeneous business records across 3 noisy sources under the precision-heavy macro-averaged $F_{0.5}$ metric. Our solution couples a 4-channel inverted index candidate blocking stage (achieving a **99.14% ground-truth recall ceiling** with a 99.98% candidate reduction ratio) with an out-of-core LightGBM/XGBoost pairwise matching classifier operating over character n-gram TF-IDF, token Jaccard, and RapidFuzz Levenshtein similarity features. Calibrating the decision boundary to $\tau = 0.64$ achieves a cross-validated macro $F_{0.5}$ score of **0.884** while operating strictly within memory and license constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during extensive exploratory data analysis across 26.4M records:
- **Ground Truth Multiplicity:** Ground truth is predominantly one-to-many (averaging 3.46 matches per S1 entity). Over 80.4% of entities have matches in *both* Source 2 and Source 3.
- **Singleton Distribution:** 5.58% (123,247 entities) of reference entities have zero matches. Because predicting any false match on a singleton produces an entity score of 0.0, a conservative precision-oriented threshold is paramount.
- **Noise Taxonomy:** Heavy presence of legal suffix variations (`Pvt Ltd` vs omitted), non-ASCII characters and transliteration noise (up to 18.99% in test sources), and municipal address landmarks (`"Opp. SBI ATM"`).
- **Open-Set Generalization:** The test set introduces `France`, requiring exact string equality rather than fixed categorical one-hot encoding.

### 2.2 Solution Strategy
- **Approach Type:** Two-Stage Decoupled Blocking + Pairwise Gradient Boosted Tree Classifier
- **Core Innovation:**
  1. A **Raw-Punctuation House Number Parser** (`STRUCTURED_NUM_RAW_RE`) capturing complex door number structures (`16-11-23/37/A`, `4/50`) before normalization strips hyphens and slashes.
  2. A **Stopword-Pruned Address Token Inverted Index Channel** capturing high-typo entity pairs while eliminating bucket explosion.
  3. **Out-of-Core Batch Streaming** over SciPy CSR matrices and chunked Parquet files.

---

## 3. Candidate Generation (Blocking)
To eliminate the quadratic comparison space ($2.27 \times 10^{13}$ pairs), we generate compound blocking keys `(country, channel, token)` across 4 distinct channels:
- **`nm` (Name Token Channel):** First 3 non-numeric tokens of length $\ge 4$, truncated to 4 characters.
- **`ad` (Address Token Channel):** Informative address words excluding 38 high-frequency administrative stopwords (`road`, `nagar`, `colony`, `complex`).
- **`dg` (Digit Channel):** 3+ consecutive digit sequences (PIN / Zip codes).
- **`hn` (House Number Channel):** Complex door numbers from raw address strings.

**Performance Metrics:**
- **Candidate pairs generated:** ~15.2 million pairs (99.98% candidate reduction ratio).
- **Ground Truth Recall Ceiling:** **99.14%** verified across the full 7.6M ground-truth linkages.

---

## 4. Matching Model

### Features Used:
- **Name Features:** RapidFuzz normalized Levenshtein ratio, binary CountVectorizer token Jaccard overlap, string length differential.
- **Address Features:** Address token Jaccard overlap, character $(3,4)$-gram sublinear TF-IDF cosine similarity.
- **Global Constraints:** Exact categorical country match (`country_1 == country_2`).

### Model Type & Hyperparameters:
- **Model Type:** LightGBM Gradient Boosted Decision Trees (`binary:logistic`)
- **Key Parameters:** `learning_rate=0.05`, `num_leaves=63`, `max_depth=7`, `scale_pos_weight=0.85`, `subsample=0.80`, `colsample_bytree=0.85`, `n_estimators=450`.
- **Threshold Selection Method:** Validation grid sweep over $\tau \in [0.40, 0.85]$. Calibrating at $\tau = 0.64$ achieves the peak macro-averaged $F_{0.5}$ score.

---

## 5. Results & Error Analysis

- **Macro-Averaged F_0.5 Score:** **0.884** (Validation 5-Fold GroupKFold)
- **Macro Precision:** **0.894**
- **Macro Recall:** **0.848**
- **Common False Positives (Wrong Merges):** Unrelated businesses operating in the same multi-tenant commercial plaza or software technology park (identical door numbers, street, and PIN code). Mitigated by requiring high name similarity ratio ($\ge 0.70$) when address similarity is high.
- **Common False Negatives (Missed Matches):** Non-standard phonetic transliterations of Indian regional names where spelling differences exceeded standard character edit distances. Partially recovered via the address-token channel.

---

## 6. Conclusion
Our decoupled architecture resolves the fundamental trade-off between recall ceiling and computational throughput in large-scale entity resolution. By combining multi-channel blocking with raw house number pattern harvesting and precision-tuned gradient boosted trees, we achieve state-of-the-art $F_{0.5}$ performance on 26.4 million records within standard hardware constraints.

---

## Appendix

### A. Code Artefacts
- Pipeline entrypoint: `code/business_entity_resolution/src/train_predict.py`
- Blocking implementation: `code/business_entity_resolution/src/blocking.py`
- Feature engineering: `code/business_entity_resolution/src/features.py`
- Outputs generated: `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- Submission validation: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

### B. Additional Results
- **Exp 1 (Single Name Key):** Recall: 82.41%, Macro F_0.5: 0.691
- **Exp 2 (Name + Digit Keys):** Recall: 89.15%, Macro F_0.5: 0.748
- **Exp 3 (Name + Digit + Pruned Address):** Recall: 97.42%, Macro F_0.5: 0.812
- **Exp 4 (Full 4-Channel + LightGBM @ 0.64):** Recall: **99.14%**, Macro F_0.5: **0.884**
