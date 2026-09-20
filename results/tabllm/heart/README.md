# Heart dataset: TabLLM response distribution

The heart test set contains 58 records: 26 records with heart disease
(44.83%) and 32 records without heart disease (55.17%). Each model was tested
with JSON, LLM, and text-template serializations.

## Prompt example

The following is one prompt from the Mistral text-template run:

```text
You are performing a binary medical classification task.

Input:
The age is 49.0. The sex is male. The cp is non-anginal pain. The trestbps is 120.0. The chol is 188.0. The fbs is false. The restecg is normal. The thalach is 139.0. The exang is no. The oldpeak is 2.0. The slope is flat. The ca is 3 major vessels. The thal is reversible defect.

Task:
Predict whether the patient has heart disease.

Label mapping:
- Answer "positive" when the predicted outcome is "heart disease".
- Answer "negative" when the predicted outcome is "no heart disease".

Rules:
- Choose exactly one of these labels: "positive" or "negative".
- Do not explain your reasoning.
- Do not output any text other than the label.
- If uncertain, choose the most likely label.

Answer:
```

The table below reports the parsed responses from the corresponding
`*_raw_responses.csv` files. Percentages use all 58 test records as the
denominator.

| Model | Serialization | Positive | Negative | Invalid |
|---|---|---:|---:|---:|
| BioMistral | JSON | 58 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| BioMistral | LLM | 58 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| BioMistral | Text template | 53 (91.38%) | 5 (8.62%) | 0 (0.00%) |
| Llama 3 | JSON | 0 (0.00%) | 58 (100.00%) | 0 (0.00%) |
| Llama 3 | LLM | 17 (29.31%) | 41 (70.69%) | 0 (0.00%) |
| Llama 3 | Text template | 1 (1.72%) | 57 (98.28%) | 0 (0.00%) |
| Meditron | JSON | 58 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| Meditron | LLM | 2 (3.45%) | 0 (0.00%) | 56 (96.55%) |
| Meditron | Text template | 19 (32.76%) | 0 (0.00%) | 39 (67.24%) |
| Mistral | JSON | 8 (13.79%) | 50 (86.21%) | 0 (0.00%) |
| Mistral | LLM | 5 (8.62%) | 53 (91.38%) | 0 (0.00%) |
| Mistral | Text template | 26 (44.83%) | 32 (55.17%) | 0 (0.00%) |

## Observed response behavior

- BioMistral returned `positive` for every record with the JSON and LLM
  serializations. With the text-template serialization, it returned 53
  positive and 5 negative responses.
- Llama 3 returned `negative` for every record with the JSON serialization.
  Its LLM and text-template runs contained both labels, with negative responses
  accounting for 70.69% and 98.28% of their outputs, respectively.
- Meditron's JSON run produced 57 responses in the form
  `{"positive": true}` and one response in the form
  `{"answer": "positive"}`. Because each response contains only the recognized
  label `positive`, all 58 are parsed as positive. Its LLM and text-template
  runs produced 56 and 39 invalid responses, respectively.
- Mistral produced both positive and negative responses with every
  serialization. Its text-template run returned 26 positive and 32 negative
  responses.

## Parsing rule

The evaluator searches each model output for the labels `positive` and
`negative`. An output containing exactly one of these labels is assigned that
label. An output containing neither label, or containing both labels, is
classified as invalid. In the heart results, all invalid responses contain
neither label; none contain both labels.
