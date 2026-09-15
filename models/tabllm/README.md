# TabLLM serialization and evaluation

This package converts the hepatitis, heart-disease, and diabetes tables into
text representations for zero-shot LLM classification. It supports three
representations:

- `text_template`: one sentence per feature
- `json`: one JSON object per row
- `llm`: a natural-language representation, using either the deterministic
  fallback or an optional Hugging Face model

## Configuration

Dataset configs live beside the code:

```text
models/tabllm/config/hepatitis_config.json
models/tabllm/config/heart_config.json
models/tabllm/config/diabetes_config.json
models/tabllm/config/setup_config.py
```

Repository paths are resolved by the configuration, while the TabLLM evaluation
commands assume the current directory is `models/tabllm`.

## Run model evaluation

From the `models/tabllm` directory:

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
python3 compute_metrics.py
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
