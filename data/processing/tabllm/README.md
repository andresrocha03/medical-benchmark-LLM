# TabLLM serialization

The serialization script converts the non-normalized medical train/test splits
into textual representations that can be included in TabLLM prompts. The target
column is kept separately and is never included in the serialized patient data.

## Serialization formats

- `text_template`: writes one short sentence per feature, such as
  `The age is 45.`
- `json`: writes the features as a one-line JSON object.
- `llm`: writes one natural-language patient description. By default this is
  deterministic; an optional Hugging Face model can generate the description.

Missing feature values are represented as `unknown`. Dataset-specific category
labels and numerical columns are defined in `models/tabllm/config/*.json`.

## Inputs and outputs

Inputs are read from `data/pre-processed/not_normalized/`. Train and test splits
remain separate throughout serialization.

Outputs are written to `data/pre-processed/serialized/` and contain the selected
serialization columns plus `target_col`:

- `serialized_text`
- `serialized_json`
- `serialized_llm`

## Run

Run all commands from `data/processing/tabllm`:

```bash
python3 serialization.py
```

Select datasets or formats:

```bash
python3 serialization.py \
  --datasets heart diabetes \
  --serializations text_template json
```

To generate the `llm` representation with a model:

```bash
python3 serialization.py \
  --serializations llm \
  --llm-model mistralai/Mistral-7B-Instruct-v0.2
```
