# SHARP: Socratic Hypothesis-Adaptive Reasoning for Pattern Induction over Scattered Evidence

This repository contains the code and dataset for **SHARP**, an autonomous reasoning framework that enables LLMs to adaptively guide their own reasoning through a **frame-hypothesize-verify** loop. Unlike prior hypothesis-driven reasoning methods that rely on human-prescribed, query-specific prompts, SHARP autonomously formulates clarifying sub-questions (Socratic Questioning) and iteratively generates and verifies hypotheses (Adaptive Hypothesis Generation) to induce latent patterns from scattered, multimodal evidence — without knowing the target query in advance.

## Key Idea

- **Socratic Questioning**: decomposes an open-ended query into structured clarifying sub-questions, collapsing an otherwise exponential hypothesis search space into a focused domain.
- **Adaptive Hypothesis Generation**: an iterative generate-then-verify loop that produces hypotheses which stay consistent and robust across fragmented episodes.
- Together, these form a hierarchical **frame → hypothesize → verify** loop that lets the model handle previously unseen target tasks, rather than only the fixed task a static prompt was designed for.

## Dataset

We introduce a fully synthetic, multimodal benchmark for **Query-Conditioned Experience Transfer**, designed to isolate query-conditioned pattern induction while avoiding pretraining-data contamination. Each episode is an image of a single object (e.g., a colored shape) with structured multi-label annotations — anomaly/normal, heavy/light, hard/soft, and price — governed by hidden compositional rules over shape, color, and attributes that are never revealed to the model.

The benchmark has three graded complexity levels:

| Level | Description | #Train | #Test |
|---|---|---|---|
| **L1 — Single-label** | Single attribute (anomaly); all queries target this one attribute. | 8 | 150 |
| **L2 — Multi-label Individual** | Multiple attributes present (anomaly, weight, hardness, price), but each query targets exactly one. | 8 | 600 |
| **L3 — Multi-label Compositional** | Multiple attributes present; each query requires reasoning over several attributes simultaneously (e.g., filtering + aggregation). | 8 | 600 |

Each episode combines Shape (circle, square) × Color (red, green, blue, yellow) × Attributes (anomaly, weight, hardness, price), governed by hidden compositional rules never revealed to the model. Evaluation uses exact-match QA accuracy (%), averaged over 3 runs.



<!-- TODO: fill in with your actual dataset location/format, e.g.:
Data lives under `data/`, split by complexity level:
data/
├── L1_single_label/
├── L2_multi_label_individual/
└── L3_multi_label_compositional/
Each folder contains episode images plus a JSON/CSV file of queries and ground-truth answers.
-->







## Usage

<!-- TODO: replace with your actual entry-point commands -->
Run SHARP:
```bash
python SHARP_Code/ours_multi_label.py
```





