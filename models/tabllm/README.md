# TabLLM serialization and evaluation

This package converts the hepatitis, heart-disease, and diabetes tables into
text representations for zero-shot LLM classification. It supports three
representations:

- `text_template`: one sentence per feature
- `json`: one JSON object per row
- `llm`: a natural-language representation, using either the deterministic
  fallback or an optional Hugging Face model

## Leakage-safe data flow

Each dataset preprocessing notebook performs the train/test split first. All
learned preprocessing statistics (imputation values and normalization
parameters) are fitted on training data only.

The notebooks produce two versions of the same split:

```text
data/pre-processed/serialization/<dataset>_train.csv  # before normalization
data/pre-processed/serialization/<dataset>_test.csv
data/pre-processed/<dataset>_train.csv                # after normalization
data/pre-processed/<dataset>_test.csv
```

TabLLM consumes the non-normalized split. Serialization preserves the split and
writes:

```text
data/serialized/<dataset>_train_serialized.csv
data/serialized/<dataset>_test_serialized.csv
```

Model evaluation reads only the serialized test file. The train file remains
available for future few-shot or supervised experiments. For diabetes, the
split is grouped by patient, and `patient_id` and `day` are removed from model
features.

Hepatitis uses `0 = live` and `1 = die`, so `die` is the positive class. Heart
disease and diabetes retain `1` as their adverse-event class.

## Configuration

Dataset configs live beside the code:

```text
models/tabllm/hepatitis_config.json
models/tabllm/heart_config.json
models/tabllm/diabetes_config.json
models/tabllm/setup_config.py
```

All repository paths are resolved from the repository root, so commands work
from either the root or the `models/tabllm` directory.

## Serialize datasets

From the repository root:

```bash
python3 -m models.tabllm.serialization
```

Select datasets or representations:

```bash
python3 -m models.tabllm.serialization \
  --datasets heart diabetes \
  --serializations text_template json
```

Generate the `llm` representation with a Hugging Face model:

```bash
python3 -m models.tabllm.serialization \
  --serializations llm \
  --llm-model mistralai/Mistral-7B-Instruct-v0.2
```

Without `--llm-model`, the `serialized_llm` column uses the deterministic local
fallback.

## Run model evaluation

```bash
python3 -m models.tabllm.test_models \
  --models mistral \
  --datasets hepatitis heart diabetes \
  --serializations text_template json llm
```

A small smoke run can be requested with:

```bash
python3 -m models.tabllm.test_models --test
```

Available model keys are defined in `setup_config.py`: `llama3`, `mistral`,
`meditron`, and `biomistral`.

Outputs are stored under `results/tabllm/`. Raw responses are organized by
dataset and serialization style, and the global summaries are written as
`all_results_summary.csv` or `all_results_test_summary.csv`.

## Metrics

Compute metrics from saved raw responses with:

```bash
python3 -m models.tabllm.metrics
```

The evaluator computes accuracy, positive-class precision and recall, macro F1,
the percentage of invalid answers, and confusion matrices. AUC is not computed
because the model produces hard text labels rather than probability scores.
