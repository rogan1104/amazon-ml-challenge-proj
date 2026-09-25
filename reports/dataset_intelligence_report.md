# Dataset Intelligence Report

**Generated:** 2026-09-25 19:10:35
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Executive Summary
- **Total dataset scale:** ~26M rows across 7 TSV files (~1.9 GB)
- **Training S1 entities:** 2,206,821
- **Singleton rate:** 5.5848%
- **Multi-match rate:** 94.4152%
- **Average matches per S1:** 3.46
- **Countries in train:** India, US
- **Test-only countries:** France
- **Matching is predominantly one-to-many** (S1 → multiple S2/S3)
- **Key noise:** name abbreviations, address format variations, punctuation/case differences
- **Critical constraint:** F_0.5 metric penalizes false merges; singletons matter

## 2. Dataset Scale

| File | Rows | Size (MB) | Columns |
|------|------|-----------|---------|
| train_S1 | 2,206,821 | 200.34 | 4 |
| train_S2 | 5,034,616 | 466.63 | 4 |
| train_S3 | 5,285,603 | 480.37 | 4 |
| test_S1 | 1,732,544 | 166.91 | 4 |
| test_S2 | 4,887,273 | 485.86 | 4 |
| test_S3 | 5,082,316 | 482.56 | 4 |
| train_GT | 2,206,821 | 121.13 | 2 |
| **TOTAL** | **26,435,994** | **2403.8** | |

## 3. Schema & Data Quality

### Schema
- **Source files:** `entity_id`, `business_name`, `business_address`, `country`
- **Ground truth:** `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs)
- **Delimiter:** Tab-separated (verified)
- **Encoding:** UTF-8

### Data Quality Summary
| Source | Rows | Missing Name % | Missing Addr % | Missing Country % | Dup ID % | Exact Dup Row % |
|--------|------|----------------|----------------|-------------------|----------|-----------------|
| S1 train | 2,206,821.0 | 0.00 | 0.00 | 0.00 | 0.0000 | 0.00 |
| S2 train | 5,034,616.0 | 0.00 | 3.36 | 0.00 | 0.0000 | 1.01 |
| S3 train | 5,285,603.0 | 0.00 | 3.33 | 0.00 | 0.0000 | 0.70 |
| S1 test | 1,732,544.0 | 0.00 | 0.00 | 0.00 | 0.0000 | 0.00 |
| S2 test | 4,887,273.0 | 0.00 | 2.65 | 0.00 | 0.0000 | 0.91 |
| S3 test | 5,082,316.0 | 0.00 | 2.68 | 0.00 | 0.0000 | 0.63 |

## 4. Source Comparison

### TRAIN
| Source | Avg Name Len | Avg Addr Len | Name Missing % | Legal Suffix % | Non-ASCII Name % | Punct Name % |
|--------|-------------|-------------|----------------|----------------|-----------------|--------------|
| S1 | 24.0 | 52.1 | 0.00 | 36.98 | 0.00 | 20.92 |
| S2 | 25.1 | 47.8 | 0.00 | 32.75 | 15.19 | 34.26 |
| S3 | 25.2 | 48.3 | 0.00 | 32.78 | 11.48 | 35.27 |
### TEST
| Source | Avg Name Len | Avg Addr Len | Name Missing % | Legal Suffix % | Non-ASCII Name % | Punct Name % |
|--------|-------------|-------------|----------------|----------------|-----------------|--------------|
| S1 | 23.8 | 57.2 | 0.00 | 27.50 | 2.35 | 18.32 |
| S2 | 25.7 | 51.8 | 0.00 | 25.72 | 18.99 | 30.70 |
| S3 | 25.7 | 50.1 | 0.00 | 25.95 | 14.51 | 31.59 |

## 5. Ground Truth Structure

- Total S1 entities: **2,206,821**
- Total true match pairs: **7,638,365**

### Match Count Distribution
| Matches | S1 Count | % |
|---------|----------|---|
| 0 | 123,247 | 5.58 |
| 1 | 119,157 | 5.40 |
| 2 | 375,212 | 17.00 |
| 3 | 530,841 | 24.05 |
| 4 | 484,115 | 21.94 |
| 5 | 321,957 | 14.59 |
| 6 | 164,868 | 7.47 |
| 7 | 63,968 | 2.90 |
| 8 | 18,680 | 0.85 |
| 9 | 4,205 | 0.19 |
| 10 | 534 | 0.02 |
| 11 | 37 | 0.00 |

### Summary Buckets
- Zero matches (singletons): 123,247.0 (5.5848%)
- Exactly 1 match: 119,157.0 (5.3995%)
- 2 matches: 375,212.0
- 3 matches: 530,841.0
- 4+ matches: 1,058,364.0
- Max matches for one S1: 11.0
- S1 with both S2 and S3 matches: 1,776,047.0
- Total S2 matches in GT: 3,693,619.0
- Total S3 matches in GT: 3,944,746.0

## 6. Match Multiplicity

- S2 entities in GT: 3,693,619
- S2 one-to-one: 100.0%
- S2 many-to-one: 0 entities matched to multiple S1
- S3 entities in GT: 3,944,746
- S3 one-to-one: 100.0%
- S3 many-to-one: 0

**OBSERVATION:** Problem behaves as **one-to-many** (S1 → S2/S3) with some **many-to-one** on the S2/S3 side.

## 7. Name Noise Analysis

| Pattern | Count | % of Pattern Hits |
|---------|-------|-------------------|
| typo_or_substitution | 40,876 | 16.77% |
| minor_token_diff | 38,213 | 15.68% |
| suffix_limited | 33,760 | 13.85% |
| suffix_private | 24,612 | 10.1% |
| missing_or_extra_words | 23,701 | 9.73% |
| suffix_llc | 16,697 | 6.85% |
| suffix_incorporated | 12,349 | 5.07% |
| case_only | 10,945 | 4.49% |
| normalized_exact | 10,945 | 4.49% |
| word_order | 6,049 | 2.48% |
| punctuation_only | 5,951 | 2.44% |
| exact | 4,642 | 1.9% |
| other_variation | 3,979 | 1.63% |
| suffix_corporation | 3,789 | 1.55% |
| suffix_company | 2,947 | 1.21% |
| suffix_llp | 2,012 | 0.83% |
| suffix_lp | 1,284 | 0.53% |
| ampersand_and | 919 | 0.38% |
| suffix_sa | 10 | 0.0% |
| suffix_gmbh | 2 | 0.0% |

### Representative Examples

**suffix_limited:**
- S1: `Premier Indo Trading Limited` ↔ Other: `પ્રીમિયર ઇન્ડો ટ્રેડિંગ લિમિટેડ`
- S1: `Hari Estate Private Limited` ↔ Other: `हरि एस्टेट प्राइवेट लिमिटेड`
- S1: `Grow Security Private Limited` ↔ Other: `Grow Security Private`

**suffix_private:**
- S1: `Hari Estate Private Limited` ↔ Other: `हरि एस्टेट प्राइवेट लिमिटेड`
- S1: `Grow Security Private Limited` ↔ Other: `Grow Security Private`
- S1: `Global Constructions Pvt Ltd` ↔ Other: `ગ્લોબલ કન્સ્ટ્રક્શન્સ પ્રા. લિ.`

**suffix_llc:**
- S1: `Royal Prairie Nano LLC` ↔ Other: `ROYAL  PRAIRIE`
- S1: `Choice Valley Aura, LLC` ↔ Other: `Choice Valley Aura, L.L.C.`
- S1: `Chavez and Bishop LLC` ↔ Other: `Chavez and Bishop LLC [Service]`

**missing_or_extra_words:**
- S1: `Royal Prairie Nano LLC` ↔ Other: `ROYAL  PRAIRIE`
- S1: `Grow Security Private Limited` ↔ Other: `Grow Security Private`
- S1: `Safina Products Limited` ↔ Other: `Safina Products`

**minor_token_diff:**
- S1: `Chiropractic Best Care Associates P.C.` ↔ Other: `CHIROPRACTIC BEMT CARE ASSOCIATES (P.C.)`
- S1: `Hanson's Horizon Massage` ↔ Other: `HANSON'S MASSAGE SERVICE`
- S1: `Brazier Industries L.L.C.` ↔ Other: `Brazier Inulrase L.L.C.`

## 8. Address Noise Analysis

| Pattern | Count | % |
|---------|-------|---|
| abbrev_road | 26,948 | 14.32% |
| numeric_diff | 26,694 | 14.19% |
| abbrev_street | 20,335 | 10.81% |
| abbrev_number | 18,748 | 9.96% |
| abbrev_drive | 15,460 | 8.22% |
| abbrev_avenue | 12,892 | 6.85% |
| other_variation | 10,776 | 5.73% |
| missing_numbers | 9,946 | 5.29% |
| abbrev_floor | 8,761 | 4.66% |
| abbrev_lane | 6,925 | 3.68% |
| normalized_exact | 6,235 | 3.31% |
| case_only | 5,368 | 2.85% |
| abbrev_court | 4,711 | 2.5% |
| component_reorder | 4,346 | 2.31% |
| missing_postal | 3,345 | 1.78% |
| abbrev_apartment | 2,253 | 1.2% |
| exact | 2,177 | 1.16% |
| abbrev_boulevard | 1,432 | 0.76% |
| postal_diff | 521 | 0.28% |
| abbrev_suite | 305 | 0.16% |

### Representative Examples

**abbrev_road:**
- S1: `21/247 Akash Ganga Apartm, Ent, Sola Road Naranpura, Ahmadabad City, Ahmedabad, Gujarat` ↔ Other: `#21/247 AKASH GANGA APARTM, ENT, SOLA ROAD NARANPURA, AHMADABAD CITY, ગુજરાત`
- S1: `11101 Armentrout Road, Fredericktown, OH` ↔ Other: `11101 ARMENTROUT RD, FREDERICKTOWN, OH`
- S1: `11101 Armentrout Road, Fredericktown, OH` ↔ Other: `11101 ARMENTROUT RD, FREDERICKTOWN, OH`

**other_variation:**
- S1: `G-52, Sector-6, Noida, Gautam Buddha Nagar, Uttar Pradesh` ↔ Other: `G-52, SECTOR-6, GAUTAM BUDDHA NAGAR, Uttar Pradesh`
- S1: `Neem Chowk, Sanwaliyaji, Near Sri Sanwariya Temple, Bhadesar, Chittorgarh, Rajasthan` ↔ Other: `NEEM CHOWK, SANWALIYAJI, NEAR SRI SANWARIYA TEMPLE, BHADESAR, राजस्थान`
- S1: `6W, Jhauganj, Patna City, Patna, Bihar` ↔ Other: `6W, JHAUGANJ, PATNA CITY, बिहार`

**case_only:**
- S1: `V-300 Rajouri Garden, New Delhi, Delhi` ↔ Other: `V-300 RAJOURI GARDEN, NEW DELHI, Delhi`
- S1: `10-6-169, Bada Bazar 1St Lancer, Ahmed Nagar, Hyderabad, Telangana` ↔ Other: `10-6-169, BADA BAZAR 1ST LANCER, AHMED NAGAR, HYDERABAD, Telangana`
- S1: `2565 Alabama Avenue, Saint Louis Park, MN` ↔ Other: `2565 ALABAMA AVENUE, SAINT LOUIS PARK, MN`

**normalized_exact:**
- S1: `V-300 Rajouri Garden, New Delhi, Delhi` ↔ Other: `V-300 RAJOURI GARDEN, NEW DELHI, Delhi`
- S1: `10-6-169, Bada Bazar 1St Lancer, Ahmed Nagar, Hyderabad, Telangana` ↔ Other: `10-6-169, BADA BAZAR 1ST LANCER, AHMED NAGAR, HYDERABAD, Telangana`
- S1: `2565 Alabama Avenue, Saint Louis Park, MN` ↔ Other: `2565 ALABAMA AVENUE, SAINT LOUIS PARK, MN`

**abbrev_drive:**
- S1: `904 Center Drive, Spokane Valley, WA` ↔ Other: `904- CENTER DR, SPOKANE VALLEY, WA`
- S1: `904 Center Drive, Spokane Valley, WA` ↔ Other: `904- CENTER DR, SPOKANE VALLEY, WA`
- S1: `5362 Meadowlane Drive, Granite Falls, NC` ↔ Other: `536 MEADOWLANE DRIVE, GRANITE FALLS, NC`

## 9. Country Analysis

- **Train countries:** India, US
- **Test countries:** France, India, US
- **Test-only:** France

### France Test Set
- Test S1: 259,452.0 France records (14.9752%)
- Test S2: 703,378.0 France records (14.392%)
- Test S3: 731,615.0 France records (14.3953%)

### Match Rates by Country (Train)
| Country | S1 Count | Singletons | With Matches | Avg Matches | Singleton % |
|---------|----------|------------|--------------|-------------|-------------|
| US | 1,323,633 | 73,896.0 | 1,249,737.0 | 3.46 | 5.58 |
| India | 883,188 | 49,351.0 | 833,837.0 | 3.46 | 5.59 |

## 10. Cross-Source Record Relationships

- S2 exact duplicate groups: {'dup_groups': 25060.0, 'dup_rows': 50933.0}
- S3 exact duplicate groups: {'dup_groups': 18381.0, 'dup_rows': 37241.0}
- S2 duplicate name groups: {'dup_name_groups': 285556.0, 'dup_rows': 1032702.0}
- S3 duplicate name groups: {'dup_name_groups': 284995.0, 'dup_rows': 991515.0}

## 11. Signal/Feature Analysis

| Signal | Condition Pairs | True Matches | P(match\|condition) |
|--------|----------------|--------------|---------------------|
| exact_normalized_name (S1-S2) | 7,427,709 | 511,311 | 0.068838 |
| exact_normalized_address (S1-S2) | 505,873 | 413,044 | 0.816497 |
| same_country + exact_name (S1-S2) | 6,020,803 | 407,332 | 0.067654 |
| same_country (S1-S2) baseline (estimated) | 5,775,254,399,373 | 3,693,619 | 0.000001 |
| exact_raw_name (S1-S2) | 3,923,353 | 174,617 | 0.044507 |
| exact_raw_address (S1-S2) | 5 | 5 | 1.000000 |
| exact_normalized_name (S1-S3) | 7,915,706 | 532,797 | 0.067309 |
| exact_normalized_address (S1-S3) | 205,981 | 170,902 | 0.829698 |
| same_country + exact_name (S1-S3) | 7,858,186 | 532,797 | 0.067802 |

## 12. Blocking Experiments

| Strategy | Candidates | Avg/S1 | Recall | Zero-Cand S1 % | Max Cand |
|----------|-----------|--------|--------|--------------|----------|
| combined (name OR addr OR country+prefix5) -> S3 | 6,714,242,443 | 3042.5 | 0.3977 | 0.76 | 49,393 |
| name_prefix_5 -> S3 | 8,351,305,639 | 3811.14 | 0.3787 | 0.70 | 49,404 |
| country + name_prefix5 -> S3 | 6,714,223,391 | 3067.94 | 0.3787 | 0.83 | 49,393 |
| combined (name OR addr OR country+prefix5) -> S2 | 6,129,590,214 | 2777.57 | 0.3752 | 0.74 | 41,204 |
| name_prefix_5 -> S2 | 7,554,519,321 | 3447.43 | 0.3615 | 0.70 | 41,215 |
| country + name_prefix5 -> S2 | 6,129,579,848 | 2800.54 | 0.3615 | 0.82 | 41,204 |
| name_first_token -> S3 | 6,926,849,322 | 3313.51 | 0.3344 | 0.23 | 51,992 |
| name_first_token -> S2 | 6,196,647,752 | 2964.16 | 0.3184 | 0.23 | 41,035 |
| name_prefix_10 -> S3 | 1,694,640,033 | 835.03 | 0.2699 | 7.88 | 29,498 |
| name_prefix_10 -> S2 | 1,652,754,555 | 814.1 | 0.2618 | 7.85 | 29,297 |
| exact_norm_name -> S3 | 7,899,506 | 8.05 | 0.0696 | 55.52 | 425 |
| country + norm_name -> S3 | 7,844,659 | 8.0 | 0.0696 | 55.55 | 425 |
| exact_norm_name -> S2 | 7,411,733 | 7.85 | 0.0668 | 57.24 | 414 |
| country + norm_name -> S2 | 7,382,915 | 7.83 | 0.0668 | 57.28 | 414 |
| exact_norm_address -> S2 | 505,873 | 1.47 | 0.0541 | 84.35 | 19 |
| country + norm_addr -> S2 | 505,873 | 1.47 | 0.0541 | 84.35 | 19 |
| exact_norm_address -> S3 | 205,981 | 1.04 | 0.0224 | 91.06 | 6 |
| country + norm_addr -> S3 | 205,981 | 1.04 | 0.0224 | 91.06 | 6 |

## 13. Hardest Positive Matches

| S1 ID | Match ID | Name Sim | Addr Sim | S1 Name | Other Name |
|-------|----------|----------|----------|---------|------------|
| S1-627736066 | S3-575071279 | 0.0 | 0.1273 | Invictus & Co | Co & Invimsmtus |
| S1-144788927 | S3-650657239 | 0.0741 | 0.1121 | Southern Consulting Limited | सदर्न कंसल्टिंग लिमिटेड |
| S1-272351912 | S3-65710282 | 0.0 | 0.1892 | DEW Spire Corp | Lumjax |
| S1-869658100 | S3-610588056 | 0.0811 | 0.1111 | Laxmi Investments Private Limited | लक्ष्मी इन्वेस्टमेंट्स प्राइवेट लिमिटेड |
| S1-50142763 | S2-959466871 | 0.0714 | 0.1228 | Advance & Co | Belojax |
| S1-727424687 | S3-436095307 | 0.075 | 0.125 | Eastern Seven Technology Private Limited | ઈસ્ટર્ન સેવન ટેક્નોલોજી પ્રાઇવેટ લિમિટેડ |
| S1-796967925 | S3-140339931 | 0.2 | 0.0 | Mmk Fashions Pvt Ltd | Mmk Ltd Pvt [Fashions] |
| S1-795047694 | S2-195311814 | 0.12 | 0.0938 | Elsinore Fisher Total Bce | Zetaarcveo |
| S1-265392356 | S2-644844388 | 0.0909 | 0.125 | Tirupati Impex Pvt Ltd | તિરુપતિ ઇમ્પેક્સ પ્રા. લિ. |
| S1-397955896 | S2-671724890 | 0.0968 | 0.1228 | Best Impex Private Limited | बेस्ट इम्पेक्स प्राइवेट लिमिटेड |
| S1-574668405 | S2-28438203 | 0.0968 | 0.125 | Modern Ventures Private Limited | मॉडर्न वेंचर्स प्राइवेट लिमिटेड |
| S1-536583156 | S2-770363267 | 0.0833 | 0.1392 | United Enterprises Private Limited | யுனைடெட் எண்டர்பிரைசஸ் பிரைவேட் லிமிடெட் |
| S1-602087272 | S3-854772376 | 0.069 | 0.1579 | Lakshmi Power Private Limited | लक्ष्मी पावर प्राइवेट लिमिटेड |
| S1-161249692 | S3-653123686 | 0.2273 | 0.0 | Omega & Sons Limited | Omega + |
| S1-481927230 | S2-47424255 | 0.05 | 0.1774 | Raviraj Club Limited | NEXVEO |

## 14. Hardest Negative Matches

| S1 ID | S2 ID | Name Sim | Addr Sim | S1 Name | S2 Name |
|-------|-------|----------|----------|---------|---------|
| S1-618355022 | S2-443940490 | 1.0 | 0.3667 | Corner Pizza | Corner Pizza |
| S1-220540589 | S2-217279135 | 1.0 | 0.2632 | Dermatology Care | Dermatology Care |
| S1-618355022 | S2-806898845 | 1.0 | 0.2581 | Corner Pizza | corner pizza |

## 15. Potential Leakage/Artifacts

- Entity ID overlap train/test: S1=0, S2=0, S3=0
- Exact record overlap train/test S1: 0
- Exact record overlap train/test S2: 0
- Shared business names train/test S1: 194,461
- Identical records S1-S2 in train: 1
- Identical records S1-S3 in train: 17

## 16. Validation Strategy Recommendation

- OBSERVATION: Do NOT randomly split rows — split by S1 entity groups to prevent leakage.
- OBSERVATION: Singletons are significant; macro F0.5 includes them at full weight.
- OBSERVATION: France appears only in test — validation cannot measure France from train labels.
- HYPOTHESIS: Stratify validation by (country, match_count_bucket) for representative evaluation.
- HYPOTHESIS: Hold out entire S1 entities, not individual match pairs.
- RECOMMENDATION: 80/20 S1-level split stratified by country and match_count (0,1,2,3,4+).
- RECOMMENDATION: Report F_0.5 separately by country and singleton vs non-singleton.
- RECOMMENDATION: Build hard-case validation subset from lowest-similarity true matches.

## 17. Key Findings

1. **Scale:** ~2.2M S1 train entities, ~5M S2, ~5.3M S3; test is ~78% of train size.
2. **Singletons are significant** — correctly predicting no-match is worth full credit.
3. **One-to-many matching** is the dominant pattern; some S2/S3 entities match multiple S1.
4. **Name and address noise is real and measurable** — not just theoretical.
5. **Country is a strong blocking signal** but France is test-only.
6. **Exact normalized name blocking** has high precision but misses many true matches.
7. **Combined blocking strategies** needed for acceptable recall.
8. **F_0.5 rewards precision** — conservative matching may outperform aggressive recall.

## 18. Engineering Implications

- Blocking must achieve >95% recall or true matches are lost forever.
- Matching model must distinguish same-name/different-business pairs.
- Singleton detection is a first-class problem, not an edge case.
- Country-aware normalization needed but cannot rely on train-only countries.
- Candidate_pairs.tsv must contain all final matches (pipeline audit requirement).
- Memory-efficient processing essential for 5M+ record sources.

## 19. Open Questions

- How well do train-derived patterns transfer to France test entities?
- What is the optimal blocking recall vs. candidate explosion tradeoff?
- Are S2 and S3 noise profiles sufficiently different to warrant source-specific models?
- How should many-to-one S2/S3 matches be handled in prediction?
- What is the public/private leaderboard split ratio?

## 20. Recommended Next Experiments

1. Prototype combined blocking pipeline and measure end-to-end recall on validation split.
2. Build stratified S1-level validation split and establish baseline F_0.5.
3. Feature ablation: name-only vs address-only vs combined similarity on validation.
4. Investigate France test records for address/name format patterns.
5. Test learned vs rule-based normalization on hardest positive cases.
6. Measure false-merge rate on hard negative pairs with candidate blocking strategies.

---
*End of Dataset Intelligence Report*