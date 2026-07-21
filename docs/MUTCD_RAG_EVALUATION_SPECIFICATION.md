---
title: "MUTCD-150 Multimodal RAG Evaluation Specification"
version: "1.0"
benchmark_id: "MUTCD-150-v1.0"
annotation_revision: "MUTCD-150-annotations-v1.1-MSDI"
msdi_version: "M-SDI-v1.0"
question_count: 150
answerable_questions: 120
unanswerable_questions: 30
evaluated_model_configurations: 12
question_only_sha256: "3a04b1d620a80704eefac34c565449a0cb8814e781dd6d73b8afb77318b954b2"
status: "provisional; repeatability and second-annotator validation pending"
---

# 1. Purpose

This document is the canonical, self-contained description of the evaluation
used for the MUTCD multimodal retrieval-augmented generation project.

It is intended for:

- researchers reviewing the experimental design;
- other RAG systems that need reliable context about the benchmark;
- agents producing tables, figures, methods sections, or results narratives;
- future evaluators checking whether new runs are directly comparable.

A downstream system should use this document as the interpretation guide and
use the final model-level JSON/CSV artifacts as the numerical source of truth.

# 2. Scope of the evaluation

The evaluation measures an end-to-end multimodal RAG pipeline that answers
questions from the Manual on Uniform Traffic Control Devices (MUTCD).

The evaluated pipeline can use:

1. text chunks;
2. source-page images;
3. cropped figures and tables;
4. knowledge-graph cross-references;
5. a vision-language model for figure filtering;
6. a vision-language model for final answering.

For the final Claude and Gemini runs, the selected answer model also performed
figure filtering because the filter override was set to `None`.

The evaluation is not limited to answer correctness. It separately measures:

- retrieval quality;
- visual-evidence retrieval;
- normative and factual answer quality;
- completeness;
- evidence faithfulness;
- citation/evidence localization;
- multimodal reasoning;
- unanswerable-question handling;
- pipeline confidence calibration;
- latency;
- structural difficulty.

# 3. Benchmark composition

## 3.1 Question distribution

| Dimension | Distribution |
|---|---:|
| Total questions | 150 |
| Answerable | 120 |
| Unanswerable | 30 |
| Text-primary | 60 |
| Table-primary | 30 |
| Figure-primary | 30 |
| Mixed-primary | 30 |
| Development/calibration split | 30 |
| Test split | 120 |

Question ID prefixes:

- `T###`: text-primary;
- `TB###`: table-primary;
- `F###`: figure-primary;
- `M###`: mixed evidence.

Eighty-five questions contain an explicit required table or figure annotation
and are included in the visual-retrieval analysis.

## 3.2 Locked evidence annotations

Each benchmark item can contain the following gold fields:

- relevant MUTCD section identifiers;
- source PDF page numbers;
- required table identifiers;
- required figure identifiers;
- gold answer;
- answerability;
- answerability subtype;
- primary modality;
- normative type;
- original difficulty;
- M-SDI component scores;
- revised M-SDI difficulty;
- split.

The question-only runtime file does not expose gold answers or evaluator
metadata to the answering system.

# 4. Evaluation units and source-of-truth hierarchy

For each model, the preferred artifact hierarchy is:

1. `*_evaluation_summary.json` — final model-level metrics;
2. `*_item_scores_msdi.csv` — final 150-item audit table;
3. final canonical or merged answers JSONL;
4. final canonical or merged retrieval JSONL;
5. merge provenance CSV, when retries were used;
6. raw primary and retry archives for audit only;
7. derived cross-model tables and heatmaps.

A downstream system must not replace a valid primary answer with a later answer
merely because the later answer is better.

# 5. Runtime validity rules

A record is operationally valid only when:

- the run status is successful;
- the saved answer is non-empty;
- `serialized_return.answer` is non-empty;
- the answer is not a notebook heading or evidence caption;
- the provider stop reason does not indicate truncation;
- no unresolved provider/runtime error remains;
- required retrieval logging is present.

Confirmed invalidity includes:

- quota or provider failure with no final answer;
- empty provider response;
- notebook heading saved as an answer;
- visibly truncated response;
- `finish_reason="length"`;
- `stop_reason="max_tokens"`;
- missing question record;
- corrupted JSONL;
- materially incomplete retrieval trace.

# 6. Retry and merge policy

Retries are allowed only for operational failures:

- provider or quota failure;
- empty response;
- confirmed truncation;
- missing record;
- corrupted output;
- interrupted execution;
- materially incomplete retrieval logging.

A substantively wrong answer is not retried.

Merge rules:

1. keep every valid primary result;
2. replace only unresolved invalid primary records;
3. use the first clean recovery run unless that run is itself invalid;
4. retain raw error histories for audit;
5. write a clean canonical errors file after reconciliation;
6. document the selected source for every replaced item.

This policy prevents post-hoc answer resampling and optimistic bias.

# 7. Retrieval evaluation

## 7.1 Chunk relevance rule

A retrieved text chunk is treated as relevant when at least one of the following
is true:

- its PDF page is a required source page;
- its section identifier is a required section;
- its table references intersect the required tables;
- its figure references intersect the required figures.

## 7.2 Retrieval metrics

The retrieval component is worth 25 points.

| Metric | Meaning | Weight |
|---|---|---:|
| Required-evidence Recall@5 | At least one relevant chunk in the first five retrieved chunks | 10 |
| MRR | Reciprocal rank of the first relevant chunk | Part of 5 |
| nDCG@6 | Rank-sensitive relevance over the six reranked chunks | Part of 5 |
| Context precision@6 | Relevant retrieved chunks divided by retrieved chunks | 5 |
| Modality routing | Whether the figure router's decision matches the gold modality requirement | 3 |
| Evidence sufficiency | Whether the retrieved text/page/visual package contains enough required evidence | 2 |

The implemented retrieval formula is:

```text
Retrieval points
= 10 × Recall@5
+ 5 × (MRR + nDCG@6) / 2
+ 5 × ContextPrecision@6
+ 3 × ModalityRouting
+ 2 × EvidenceSufficiency
```

# 8. Visual-retrieval evaluation

Visual evaluation is performed on the 85 items containing an explicit required
table or figure.

Metrics:

- **Exact required visual-crop hit:** at least one retrieved crop has the
  canonical identifier of a required table or figure.
- **Required page hit@1:** a required source page is ranked first.
- **Required page hit@3:** a required source page appears within the first
  three page images.
- **Required page hit@4:** a required source page appears within the first
  four page images.
- **Visual-crop precision:** required matching crops divided by all retrieved
  crops.
- **Visual recall:** unique required visuals recovered divided by the number of
  required visuals; retained in item-level files even when not shown in the
  headline table.

The present framework does not use a generic visual Top-5 metric as the primary
visual result. It distinguishes page retrieval from exact visual-entity
retrieval and penalizes irrelevant extra crops.

# 9. Answer-generation evaluation

The answer-generation component is worth 60 points.

| Dimension | Weight |
|---|---:|
| Factual and normative correctness | 22 |
| Completeness | 10 |
| Faithfulness to retrieved evidence | 12 |
| Citation and evidence localization | 6 |
| Visual/table/layout reasoning | 7 |
| Relevance and concision | 3 |

The implemented formula is:

```text
Generation points
= 22 × Correctness
+ 10 × Completeness
+ 12 × Faithfulness
+ 6 × CitationLocalization
+ 7 × VisualReasoning
+ 3 × RelevanceConcision
```

## 9.1 Item-level scale

Correctness, completeness, faithfulness, and citation localization are audited
on a 0–1 scale against the locked gold annotation.

Typical interpretation:

- `1.00`: fully satisfies the dimension;
- `0.75`: mostly correct or complete with a limited omission;
- `0.50`: partially correct or incomplete;
- `0.25`: minimal useful content;
- `0.00`: absent, incorrect, or unjustified.

Intermediate values can be used when the omission or error lies between these
anchors.

## 9.2 Normative correctness

Normative correctness considers:

- MUTCD modal force: `shall`, `should`, `may`;
- Standard, Guidance, Option, and Support distinctions;
- numerical thresholds;
- exceptions;
- required quantities and placements;
- conditions governing applicability;
- exact sign, marking, table, or figure interpretation.

## 9.3 Strict full-credit rate

An item receives strict full credit only when all four core audit dimensions
equal `1.0`:

```text
Correctness = 1
Completeness = 1
Faithfulness = 1
Citation localization = 1
```

The strict full-credit indicator does not directly include the separate visual
reasoning or relevance/concision dimensions.

## 9.4 Visual reasoning implementation

The reported model-level visual reasoning rate is the mean correctness score
over non-text-primary questions:

- table;
- figure;
- mixed.

It is not a separate second manual score at item level.

## 9.5 Relevance/concision implementation

In the completed model evaluations, relevance/concision was held at the same
conservative value, `0.696`, for every model.

Therefore, it does not contribute to cross-model differences. This should be
reported transparently and should be replaced by item-level relevance/concision
scoring in a future evaluator revision.

# 10. Reliability evaluation

The intended reliability component is worth 15 points:

| Dimension | Weight |
|---|---:|
| Unanswerable-question handling | 6 |
| Confidence calibration | 6 |
| Repeatability | 3 |

Repeatability is pending. Current model comparisons therefore use 12 measurable
reliability points.

## 10.1 Unanswerable handling

The benchmark includes 30 deliberately unanswerable questions.

Unanswerable success is the strict full-credit rate on those 30 items. A good
response should abstain accurately and explain that the requested information
is absent, unsupported, out of scope, contradicted, or falsely presupposed.

## 10.2 Calibration

The current calibration implementation uses:

- Brier score;
- 10-bin expected calibration error (ECE).

The available calibration points are:

```text
Calibration points = 6 × [1 - (Brier + ECE) / 2]
```

Important limitation:

The confidence value used in the completed item tables is the pipeline's
`figure_router.confidence`, not a direct answer-confidence probability supplied
by the final VLM.

Therefore, the current Brier/ECE values should be interpreted as pipeline/router
calibration against strict full-credit outcomes, not as pure LLM answer
confidence calibration.

# 11. Composite score

Currently measurable maximum:

```text
25 retrieval + 60 generation + 12 reliability = 97 points
```

Current total:

```text
Current total = Retrieval + Generation + Available reliability
```

Normalized provisional score:

```text
Normalized score = Current total / 97 × 100
```

The final 100-point score will require the pending 3-point repeatability
component.

All current rankings are provisional.

# 12. M-SDI structural difficulty

M-SDI is a project-defined, model-independent structural difficulty index.

Each item receives five component scores from 0 to 2:

| Symbol | Component |
|---|---|
| L | Evidence localization burden |
| M | Modality integration burden |
| R | Reasoning operations burden |
| A | Answer composition burden |
| N | Normative precision burden |

```text
M-SDI total = L + M + R + A + N
```

Difficulty bands:

- Easy: 0–3;
- Medium: 4–6;
- Hard: 7–10.

Revised distribution:

| Difficulty | Count |
|---|---:|
| Easy | 66 |
| Medium | 62 |
| Hard | 22 |

Answerability remains separate from difficulty.

Unanswerable distribution:

- 12 easy;
- 16 medium;
- 2 hard.

# 13. Answerability subtypes

Unanswerable items may be classified as:

- `absent_information`;
- `false_presupposition`;
- `unsupported_precision`;
- `out_of_scope`;
- `contradicted_premise`.

A downstream analysis should report answerability separately from difficulty
and modality.

# 14. Metrics not used as primary measures

The reported evaluation does not use the following as primary answer-quality
metrics:

- exact match;
- token-level F1;
- LLM-as-a-judge;
- BLEU, ROUGE, or generic semantic similarity.

Reason:

MUTCD answers frequently require semantically equivalent wording, modal-force
distinctions, multiple thresholds, exceptions, and evidence citations. Exact
match and token overlap can mis-score valid paraphrases and incomplete
normative answers.

These metrics may be added later for constrained subsets such as:

- sign codes;
- dimensions;
- colors;
- yes/no answers;
- single numerical thresholds.

# 15. Final cross-model results

## 15.1 Core performance

|   Rank | Model                  | Family   | Tier       |   Overall score (/100) |   Retrieval (/25) |   Generation (/60) |   Reliability (/12) | Correctness   | Completeness   | Faithfulness   | Citation localization   | Visual reasoning   | Full-credit rate   | Unanswerable success   |   Median latency (s) |
|-------:|:-----------------------|:---------|:-----------|-----------------------:|------------------:|-------------------:|--------------------:|:--------------|:---------------|:---------------|:------------------------|:-------------------|:-------------------|:-----------------------|---------------------:|
|      1 | Claude Fable 5         | Claude   | Frontier   |                  88.08 |             19.35 |              55.18 |               10.9  | 92.0%         | 90.2%          | 98.4%          | 94.5%                   | 90.9%              | 80.0%              | 96.7%                  |                28.41 |
|      2 | Claude Sonnet 5        | Claude   | Balanced   |                  87.81 |             19.35 |              55.18 |               10.66 | 92.4%         | 89.9%          | 98.1%          | 94.2%                   | 90.8%              | 74.7%              | 96.7%                  |                10.97 |
|      3 | Qwen3.7 Max            | Qwen     | Frontier   |                  87.35 |             19.63 |              54.33 |               10.78 | 90.0%         | 88.1%          | 97.3%          | 93.7%                   | 90.3%              | 76.7%              | 96.7%                  |                68.07 |
|      4 | Gemini 3.1 Pro Preview | Gemini   | Frontier   |                  87.25 |             19.35 |              54.34 |               10.94 | 89.9%         | 87.8%          | 98.7%          | 92.8%                   | 89.8%              | 76.0%              | 100.0%                 |                21.71 |
|      5 | Qwen3-VL 235B          | Qwen     | Flagship   |                  87.02 |             19.63 |              54.11 |               10.67 | 93.0%         | 89.9%          | 92.3%          | 86.2%                   | 90.3%              | 74.0%              | 96.7%                  |                17.31 |
|      6 | Qwen3.6 Flash          | Qwen     | Fast       |                  86.96 |             19.35 |              54.38 |               10.63 | 90.4%         | 87.8%          | 97.2%          | 94.2%                   | 90.2%              | 74.0%              | 96.7%                  |                32.72 |
|      7 | Qwen3.5 Omni Plus      | Qwen     | Challenger |                  86.95 |             19.63 |              54.45 |               10.27 | 92.0%         | 88.1%          | 96.2%          | 90.8%                   | 90.2%              | 71.3%              | 90.0%                  |                10.67 |
|      8 | Gemini 3.1 Flash-Lite  | Gemini   | Fast       |                  86.92 |             19.35 |              54.21 |               10.76 | 89.9%         | 87.8%          | 97.7%          | 93.5%                   | 89.0%              | 76.7%              | 96.7%                  |                 9.21 |
|      9 | Qwen3.7 Plus           | Qwen     | Balanced   |                  86.89 |             19.35 |              54.16 |               10.78 | 89.9%         | 87.5%          | 97.8%          | 92.3%                   | 89.7%              | 76.7%              | 96.7%                  |                72.64 |
|     10 | Gemini 3.5 Flash       | Gemini   | Balanced   |                  86.6  |             19.35 |              53.92 |               10.74 | 89.1%         | 87.4%          | 98.1%          | 91.7%                   | 88.8%              | 76.0%              | 96.7%                  |                22.53 |
|     11 | Claude Haiku 4.5       | Claude   | Fast       |                  86.47 |             19.35 |              53.64 |               10.89 | 88.9%         | 87.0%          | 97.6%          | 92.8%                   | 85.7%              | 75.3%              | 100.0%                 |                10.43 |
|     12 | Qwen3-VL Flash         | Qwen     | Fast       |                  85.08 |             19.63 |              53    |                9.9  | 88.5%         | 87.6%          | 92.0%          | 91.7%                   | 87.8%              | 68.7%              | 86.7%                  |                16.29 |

## 15.2 Retrieval, visual retrieval, and calibration

| Model                  | Recall@5   | MRR   | nDCG@6   | Context precision@6   | Modality routing   | Evidence sufficiency   | Exact visual crop hit   | Visual crop precision   |   Brier score |   ECE |
|:-----------------------|:-----------|:------|:---------|:----------------------|:-------------------|:-----------------------|:------------------------|:------------------------|--------------:|------:|
| Claude Fable 5         | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 76.5%                   | 72.5%                   |         0.169 | 0.13  |
| Claude Sonnet 5        | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 76.5%                   | 72.5%                   |         0.211 | 0.17  |
| Qwen3.7 Max            | 88.0%      | 83.6% | 82.4%    | 49.4%                 | 78.0%              | 93.3%                  | 83.5%                   | 79.8%                   |         0.191 | 0.15  |
| Gemini 3.1 Pro Preview | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 78.8%                   | 73.8%                   |         0.198 | 0.156 |
| Qwen3-VL 235B          | N/A        | N/A   | N/A      | N/A                   | N/A                | N/A                    | N/A                     | N/A                     |         0.214 | 0.163 |
| Qwen3.6 Flash          | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 75.3%                   | 68.9%                   |         0.215 | 0.176 |
| Qwen3.5 Omni Plus      | 88.0%      | 83.6% | 82.4%    | 49.4%                 | 78.0%              | 93.3%                  | 83.5%                   | 79.8%                   |         0.22  | 0.157 |
| Gemini 3.1 Flash-Lite  | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 77.6%                   | 70.1%                   |         0.198 | 0.15  |
| Qwen3.7 Plus           | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 78.8%                   | 71.0%                   |         0.191 | 0.15  |
| Gemini 3.5 Flash       | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 78.8%                   | 67.1%                   |         0.198 | 0.156 |
| Claude Haiku 4.5       | 86.7%      | 82.6% | 81.3%    | 47.6%                 | 78.0%              | 93.3%                  | 77.6%                   | 71.2%                   |         0.208 | 0.163 |
| Qwen3-VL Flash         | 88.0%      | 83.6% | 82.4%    | 49.4%                 | 78.0%              | 93.3%                  | 82.4%                   | 68.0%                   |         0.25  | 0.184 |

Notes:

- `N/A` means the detailed retrieval submetrics were not persisted in the final
  Flagship artifact.
- The Flagship retrieval component was reconstructed from the final normalized
  score and the persisted generation and reliability components.
- The reconstructed Flagship retrieval component is 19.63/25 at two-decimal
  precision.
- Comparison tables should preserve this provenance note.

# 16. Highest observed provisional model

The highest observed provisional score is **Claude Fable 5**:

| Metric | Value |
|---|---:|
| Overall normalized provisional score | 88.08/100 |
| Retrieval | 19.35/25 |
| Answer generation | 55.18/60 |
| Available reliability | 10.90/12 |
| Correctness | 92.0% |
| Completeness | 90.2% |
| Faithfulness | 98.4% |
| Citation localization | 94.5% |
| Visual reasoning | 90.9% |
| Strict full-credit rate | 80.0% |
| Unanswerable success | 96.7% |
| Median latency | 28.41 s |
| Brier score | 0.169 |
| ECE | 0.130 |

The observed margin over **Claude Sonnet 5** is only
**0.26 normalized points**.

This should be described as the highest observed provisional result, not as
statistically proven superiority.

# 17. Claude Fable 5 performance by modality

| primary_modality   |   n | correctness   | completeness   | faithfulness   | citation   | full_credit   |   median_latency_s | recall5   |
|:-------------------|----:|:--------------|:---------------|:---------------|:-----------|:--------------|-------------------:|:----------|
| figure             |  30 | 89.2%         | 89.2%          | 97.5%          | 91.7%      | 83.3%         |              29.69 | 90.0%     |
| mixed              |  30 | 91.2%         | 89.5%          | 98.3%          | 96.7%      | 76.7%         |              31.61 | 90.0%     |
| table              |  30 | 92.5%         | 93.3%          | 97.5%          | 96.7%      | 90.0%         |              30.31 | 83.3%     |
| text               |  60 | 93.5%         | 89.5%          | 99.3%          | 93.8%      | 75.0%         |              15.49 | 85.0%     |

Interpretation:

- Table-primary questions produced the highest strict full-credit rate: 90.0%.
- Figure-primary questions produced 83.3% strict full credit.
- Mixed questions had high correctness but lower strict full credit because
  multi-part omissions and evidence-localization requirements remained common.
- Text questions had the lowest latency.

# 18. Claude Fable 5 performance by M-SDI and answerability

| answerable   | revised_difficulty   |   n | correctness   | completeness   | faithfulness   | citation   | full_credit   | recall5   |
|:-------------|:---------------------|----:|:--------------|:---------------|:---------------|:-----------|:--------------|:----------|
| False        | easy                 |  12 | 100.0%        | 100.0%         | 97.9%          | 100.0%     | 91.7%         | 66.7%     |
| False        | hard                 |   2 | 100.0%        | 100.0%         | 100.0%         | 100.0%     | 100.0%        | 100.0%    |
| False        | medium               |  16 | 100.0%        | 100.0%         | 100.0%         | 100.0%     | 100.0%        | 56.2%     |
| True         | easy                 |  54 | 88.2%         | 85.9%          | 97.9%          | 90.7%      | 75.9%         | 88.9%     |
| True         | hard                 |  20 | 89.2%         | 85.9%          | 97.5%          | 95.0%      | 70.0%         | 95.0%     |
| True         | medium               |  46 | 92.3%         | 90.7%          | 98.9%          | 95.1%      | 78.3%         | 95.7%     |

Interpretation:

- On answerable items, medium questions had the highest correctness and strict
  full-credit rate.
- Hard questions retained high correctness but lower full-credit performance,
  indicating omissions of qualifications, quantities, or evidence rather than
  only wholly incorrect conclusions.
- Unanswerable handling was nearly perfect, but retrieval recall is not expected
  to be high for every unanswerable item because the correct behavior can depend
  on recognizing that required evidence is absent.

# 19. Claude Fable 5 principal residual failures

| question_id   | primary_modality   | revised_difficulty   | answerable   | correctness   | completeness   | faithfulness   | citation   | notes                                                                                                                           |
|:--------------|:-------------------|:---------------------|:-------------|:--------------|:---------------|:---------------|:-----------|:--------------------------------------------------------------------------------------------------------------------------------|
| M008          | mixed              | hard                 | True         | 10.0%         | 0.0%           | 60.0%          | 0.0%       | Reverses the 35-mph sign-size exception and omits the required 36 x 36-inch STOP size.                                          |
| F006          | figure             | medium               | True         | 0.0%          | 0.0%           | 100.0%         | 0.0%       | Abstains instead of identifying the W2-1 Cross Road sign.                                                                       |
| F009          | figure             | easy                 | True         | 0.0%          | 0.0%           | 100.0%         | 0.0%       | Abstains instead of identifying DO NOT ENTER above WRONG WAY.                                                                   |
| T030          | text               | easy                 | True         | 0.0%          | 0.0%           | 100.0%         | 0.0%       | Abstains instead of listing yellow, white, red, blue, and purple.                                                               |
| T034          | text               | easy                 | True         | 0.0%          | 0.0%           | 100.0%         | 0.0%       | Abstains instead of stating that a roundabout yield line is optional.                                                           |
| TB008         | table              | medium               | True         | 0.0%          | 0.0%           | 100.0%         | 0.0%       | Abstains and omits all three STOP-sign sizes.                                                                                   |
| TB011         | table              | easy                 | True         | 0.0%          | 0.0%           | 50.0%          | 100.0%     | Returns 275 feet instead of the required 200 feet.                                                                              |
| F005          | figure             | easy                 | True         | 25.0%         | 25.0%          | 75.0%          | 50.0%      | Misreads Figure 2A-3 and does not return the 2-foot island-nose distance.                                                       |
| T024          | text               | easy                 | True         | 25.0%         | 25.0%          | 100.0%         | 50.0%      | Provides the 70-percent high-speed adjustment but omits the 300/200 thresholds and same-eight-hours basis.                      |
| T017          | text               | easy                 | True         | 60.0%         | 50.0%          | 70.0%          | 50.0%      | Correct left-to-right orientation but substitutes a community-wayfinding casing rule for the general sign-message casing rule.  |
| F013          | figure             | medium               | True         | 50.0%         | 50.0%          | 75.0%          | 100.0%     | Identifies optional diagonal markings but substitutes the optional yellow edge line for the requested wide dotted extension.    |
| M013          | mixed              | hard                 | True         | 50.0%         | 50.0%          | 90.0%          | 100.0%     | Identifies the additional condition categories but omits the 5/3 crash thresholds and 80-percent criterion.                     |
| T026          | text               | easy                 | True         | 75.0%         | 50.0%          | 90.0%          | 75.0%      | Names the three sign types but omits minimum quantities and exact required locations.                                           |
| M012          | mixed              | hard                 | True         | 50.0%         | 50.0%          | 100.0%         | 100.0%     | Correct time basis and non-compulsion rule; omits the 600/150-vph thresholds.                                                   |
| M014          | mixed              | medium               | True         | 50.0%         | 67.0%          | 100.0%         | 100.0%     | Correct warning-sign treatment but omits the required 460-foot distance.                                                        |
| T006          | text               | easy                 | True         | 80.0%         | 75.0%          | 100.0%         | 75.0%      | The documentation distinction is inferred correctly but is not stated as explicitly as the locked definitions.                  |
| T046          | text               | medium               | True         | 80.0%         | 57.0%          | 100.0%         | 100.0%     | Correct coordination, flaggers, temporary signals, and self-regulation; omits flag transfer, pilot car, and stop/yield control. |
| T037          | text               | medium               | True         | 90.0%         | 75.0%          | 100.0%         | 75.0%      | Correct function, engineering basis, and duration guidance; omits the prohibition on pre-yellow displays.                       |
| M011          | mixed              | hard                 | True         | 75.0%         | 67.0%          | 100.0%         | 100.0%     | Correct marking trigger and 400-foot connection rule; omits the 900-foot threshold at 55 mph.                                   |
| TB006         | table              | medium               | True         | 75.0%         | 100.0%         | 75.0%          | 100.0%     | Correct temporary-traffic-control and ETC colors but incorrectly includes orange as an incident-management background.          |

Dominant residual failure modes:

1. abstention despite relevant MUTCD evidence being present;
2. failure to retrieve a specific figure or table row;
3. omission of numerical thresholds or exceptions;
4. partial normative answers;
5. incorrect interpretation of a figure dimension;
6. correct governing rule with incomplete quantities or placements.

# 20. Statistical and methodological limitations

The following limitations must accompany paper claims:

1. Repeatability is pending; current totals are normalized from 97 points.
2. Small model-score differences have not been shown statistically significant.
3. The item-level audit should be independently replicated by a second
   MUTCD-informed annotator.
4. Weighted Cohen's kappa or ICC should be reported for independent annotation.
5. Disagreements should be adjudicated before final publication.
6. Calibration presently uses router confidence rather than VLM answer
   confidence.
7. Relevance/concision was held constant across models.
8. Preview endpoints can change; exact model IDs and run dates must be retained.
9. Latency reflects this specific retrieval pipeline, hardware, API conditions,
   image count, and provider load.
10. Provider quotas caused some selective retries; only operationally failed
    records were replaced.

# 21. Rules for downstream RAG systems

When answering questions from this document:

1. Do not invent metrics absent from the tables.
2. Do not describe the ranking as statistically significant.
3. Call scores provisional until repeatability is added.
4. Do not claim LLM-as-a-judge, exact match, or F1 were primary metrics.
5. Do not call the manual audit expert consensus until independent annotation
   is completed.
6. Preserve the distinction between:
   - correctness;
   - completeness;
   - faithfulness;
   - citation localization;
   - visual reasoning;
   - strict full credit.
7. Preserve answerability as separate from difficulty.
8. Do not interpret router Brier/ECE as direct VLM self-confidence.
9. Do not substitute later retry answers for valid primary answers.
10. Use final canonical/merged files, not raw failed archives, for model scoring.
11. Mention the Flagship retrieval-submetric provenance when presenting detailed
    retrieval tables.
12. Report model IDs, not only friendly aliases.

# 22. Recommended artifact names

Per-model final artifacts:

- `*_evaluation_summary.json`
- `*_item_scores_msdi.csv`
- `*_summary_by_modality.csv`
- `*_summary_by_answerability_difficulty.csv`
- `*_visual_retrieval_summary.csv`
- canonical or merged answers JSONL
- canonical or merged retrieval JSONL
- merge provenance CSV, if applicable

Cross-model paper artifacts:

- full metrics CSV;
- compact paper comparison CSV;
- LaTeX comparison table;
- all-model heatmap;
- score–latency plot;
- best-model modality table and heatmap;
- best-model difficulty table and heatmap;
- residual-failure table.

# 23. Minimal canonical summary

```text
Benchmark: MUTCD-150-v1.0
Questions: 150
Answerable/unanswerable: 120/30
Modalities: 60 text, 30 table, 30 figure, 30 mixed
Difficulty: 66 easy, 62 medium, 22 hard
Score: 25 retrieval + 60 generation + 15 reliability
Currently measurable: 97 points
Normalized provisional score: current_total / 97 × 100
Best observed provisional model: Claude Fable 5
Best observed score: 88.08/100
Best strict full-credit rate: 80.0%
Repeatability: pending
Independent second annotation: pending
```
