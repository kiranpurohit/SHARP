# SHARP: Socratic Hypothesis-Adaptive Reasoning for Pattern Induction over Scattered Evidence

This repository contains the code and dataset for **SHARP**, an autonomous reasoning framework that enables LLMs to adaptively guide their own reasoning through a **frame-hypothesize-verify** loop. Unlike prior hypothesis-driven reasoning methods that rely on human-prescribed, query-specific prompts, SHARP autonomously formulates clarifying sub-questions (Socratic Questioning) and iteratively generates and verifies hypotheses (Adaptive Hypothesis Generation) to induce latent patterns from scattered, multimodal evidence — without knowing the target query in advance.

> This repository is anonymized for double-blind review. No author, institution, or identifying information is included.

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

## Baselines

- **Oracle** — uses ground-truth rules as semantic memory.
- **Episodes Only** — reasons directly over raw episodes without abstraction.
- **HDR (Query-Agnostic Semanticization)** — hypotheses generated once from episodes, used as static semantic memory.
- **Direct Query Semanticization** — semantic memory generated in a single step, conditioned on the query.
- **SHARP (Ours)** — decomposes the query into atomic sub-questions and iteratively constructs query-conditioned semantic memory.

## Implementation Details (from paper)

- Models: GPT-5.1 and Gemini 2.5 Flash (via API), used for question framing, hypothesis generation, verification, and final reasoning.
- Generate-verify loop steps: K = 3
- Max reasoning turns: T = 3
- Temperature = 0, fixed seed (deterministic decoding); each experiment repeated 3× and mean accuracy reported.
- Compute: single A100 (80GB).

<!-- TODO: fill in with your actual dataset location/format, e.g.:
Data lives under `data/`, split by complexity level:
data/
├── L1_single_label/
├── L2_multi_label_individual/
└── L3_multi_label_compositional/
Each folder contains episode images plus a JSON/CSV file of queries and ground-truth answers.
-->

## Repository Structure

<!-- TODO: replace with your actual folder layout -->
```
SHARP/
├── data/               # Benchmark dataset (L1/L2/L3)
├── src/                # Core SHARP framework (Socratic Questioning + Hypothesis Generation/Verification)
├── baselines/          # Baseline methods (Oracle, HDR, Socratic-only, etc.)
├── configs/            # Experiment configs
├── scripts/            # Run/eval scripts
├── requirements.txt
└── README.md
```

## Installation

<!-- TODO: confirm/adjust Python version and dependencies -->
```bash
git clone <this-repo-url>
cd SHARP

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

<!-- TODO: add any API key / model access setup, e.g.:
Set your LLM API key as an environment variable:
export OPENAI_API_KEY="your-key-here"
-->

## Usage

<!-- TODO: replace with your actual entry-point commands -->
Run SHARP on a given complexity level:
```bash
python scripts/run_sharp.py --level L3 --model gpt-5.1
```

Run a baseline for comparison:
```bash
python scripts/run_baseline.py --method HDR --level L3 --model gpt-5.1
```

Evaluate results:
```bash
python scripts/evaluate.py --results_dir outputs/L3
```

## Prompt Templates

SHARP uses three core prompt templates (full text in the paper's Appendix A.2):
1. **Sub-question generation** — produces the minimal set of atomic sub-questions needed to recover missing concepts for the target query.
2. **Answer generation after semantic rule extraction** — answers the target question using the induced semantic rules as context.
3. **Answer generation from episodic evidence only** — baseline prompt that answers directly from raw episodes without semanticization.

<!-- TODO: point to the actual prompt files/templates in your code, e.g. `src/prompts/` -->

## Citation

This work is currently under double-blind review. Citation details will be added upon publication.

## License

<!-- TODO: add your license, e.g. MIT, Apache-2.0, or "Released for review purposes only" -->
