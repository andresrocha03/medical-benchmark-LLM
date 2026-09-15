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
data/pre-processed/not_normalized/<dataset>_train.csv  # before normalization
data/pre-processed/not_normalized/<dataset>_test.csv
data/pre-processed/<dataset>_train.csv                # after normalization
data/pre-processed/<dataset>_test.csv
```

TabLLM consumes the non-normalized split. Serialization preserves the split and
writes:

```text
data/pre-processed/serialized/<dataset>_train_serialized.csv
data/pre-processed/serialized/<dataset>_test_serialized.csv
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
models/tabllm/config/hepatitis_config.json
models/tabllm/config/heart_config.json
models/tabllm/config/diabetes_config.json
models/tabllm/config/setup_config.py
```

All repository paths are resolved from the repository root, so commands work
from either the root or the `models/tabllm` directory.

## Serialize datasets

From the repository root:

```bash
python3 -m data.processing.tabllm.serialization
```

Select datasets or representations:

```bash
python3 -m data.processing.tabllm.serialization \
  --datasets heart diabetes \
  --serializations text_template json
```

Generate the `llm` representation with a Hugging Face model:

```bash
python3 -m data.processing.tabllm.serialization \
  --serializations llm \
  --llm-model mistralai/Mistral-7B-Instruct-v0.2
```

Without `--llm-model`, the `serialized_llm` column uses the deterministic local
fallback.

## Run model evaluation

From the repository root:

```bash
python3 -m models.tabllm.run_test \
  --models mistral \
  --datasets hepatitis heart diabetes \
  --serializations text_template json llm
```

Or, from the `models/tabllm` directory, run the file directly:

```bash
python3 run_test.py \
  --models mistral \
  --datasets hepatitis heart diabetes \
  --serializations text_template json llm
```

Available model keys are defined in `setup_config.py`: `llama3`, `mistral`,
`meditron`, and `biomistral`.

Outputs are stored under `results/tabllm/`. Raw responses and confusion
matrices are organized by dataset and serialization style. Every completed
`(model, serialization, dataset)` trio contributes one row to
`tabllm_results.csv`.

## Metrics

Metrics are computed automatically after every trio. To regenerate them from
saved raw responses, run:

```bash
python3 -m models.tabllm.compute_metrics
```

The evaluator saves the same test-metric columns used by the other benchmark
models: accuracy, positive-class precision and recall, macro F1, and AUC. AUC
is recorded as unavailable (`NaN`) because TabLLM produces hard text labels
rather than probability scores. Invalid-answer counts and percentages are also
saved, and invalid answers appear explicitly in the confusion matrices.

Every prompt maps the dataset's adverse outcome to `positive` and the other
outcome to `negative`. A response containing exactly one of those labels is
parsed as that prediction. Responses containing neither label or both labels
are invalid and count as incorrect predictions.
