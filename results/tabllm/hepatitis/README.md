# Hepatitis dataset: TabLLM response distribution

The hepatitis test set contains 31 records: 6 deaths (19.35%) and 25 survivals
(80.65%). The positive label represents `die`, and the negative label
represents `live`. Each model was tested with JSON, LLM, and text-template
serializations.

## Prompt example

The following is one prompt from the Mistral text-template run:

```text
You are performing a binary medical classification task.

Input:
The Age is 31.0. The Sex is male. The Steroid is yes. The Antivirals is yes. The Fatigue is yes. The Malaise is yes. The Anorexia is yes. The Liver Big is yes. The Liver Firm is yes. The Spleen Palpable is yes. The Spiders is yes. The Ascites is yes. The Varices is yes. The Bilirubin is 1.0. The Alk Phosphate is 85.0. The Sgot is 20.0. The Albumin is 4.0. The Protime is 100.0. The Histology is no.

Task:
Predict the hepatitis outcome.

Label mapping:
- Answer "positive" when the predicted outcome is "die".
- Answer "negative" when the predicted outcome is "live".

Rules:
- Choose exactly one of these labels: "positive" or "negative".
- Do not explain your reasoning.
- Do not output any text other than the label.
- If uncertain, choose the most likely label.

Answer:
```

## Parsed response distribution

The table reports responses parsed from the corresponding
`*_raw_responses.csv` files. Percentages use all 31 test records as the
denominator.

| Model | Serialization | Positive | Negative | Invalid |
|---|---|---:|---:|---:|
| BioMistral | JSON | 18 (58.06%) | 13 (41.94%) | 0 (0.00%) |
| BioMistral | LLM | 31 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| BioMistral | Text template | 26 (83.87%) | 5 (16.13%) | 0 (0.00%) |
| Llama 3 | JSON | 0 (0.00%) | 31 (100.00%) | 0 (0.00%) |
| Llama 3 | LLM | 0 (0.00%) | 31 (100.00%) | 0 (0.00%) |
| Llama 3 | Text template | 0 (0.00%) | 31 (100.00%) | 0 (0.00%) |
| Meditron | JSON | 31 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| Meditron | LLM | 0 (0.00%) | 0 (0.00%) | 31 (100.00%) |
| Meditron | Text template | 31 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| Mistral | JSON | 7 (22.58%) | 24 (77.42%) | 0 (0.00%) |
| Mistral | LLM | 7 (22.58%) | 24 (77.42%) | 0 (0.00%) |
| Mistral | Text template | 12 (38.71%) | 19 (61.29%) | 0 (0.00%) |

## Observed response behavior

- BioMistral returned both labels with JSON and text-template serialization.
  Its LLM run returned `positive` for all 31 records.
- Llama 3 returned `negative` for all 31 records with every serialization.
- Meditron's JSON run returned `{"positive"}` for all 31 records. Its
  text-template responses also all contained only the recognized label
  `positive`, although the responses included additional text. Both runs were
  therefore parsed as 31 positive predictions.
- Meditron's LLM run produced 31 responses containing neither `positive` nor
  `negative`; all were classified as invalid.
- Mistral returned both labels with every serialization. Its JSON and LLM runs
  had identical parsed distributions: 7 positive and 24 negative responses.
  Its text-template run had 12 positive and 19 negative responses.

## Parsing rule

The evaluator searches each output for the labels `positive` and `negative`.
An output containing exactly one of these labels is assigned that label. An
output containing neither label, or containing both labels, is classified as
invalid. All 31 invalid hepatitis responses contain neither label; no response
contains both labels.
