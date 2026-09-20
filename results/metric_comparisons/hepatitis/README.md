# Hepatitis: model performance and response behavior

## Class distribution

The target is the hepatitis outcome. Death is the positive class, and survival
is the negative class.

| Split | Negative (live) | Positive (die) | Negative:positive ratio |
|---|---:|---:|---:|
| Training | 98 (79.03%) | 26 (20.97%) | 3.77:1 |
| Test | 25 (80.65%) | 6 (19.35%) | 4.17:1 |

On the test set, an always-negative classifier obtains 80.65% accuracy, 0%
positive recall, and a macro F1 of 0.446. An always-positive classifier obtains
19.35% accuracy, 100% positive recall, and a macro F1 of 0.162.

## Tabular-model results

All five tabular models produced both positive and negative predictions.

| Model | Predicted positive | TP | FP | TN | FN | Accuracy | Precision | Recall | Macro F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 7 | 4 | 3 | 22 | 2 | 83.87% | 57.14% | 66.67% | 0.757 | 0.793 |
| XGBoost | 6 | 3 | 3 | 22 | 3 | 80.65% | 50.00% | 50.00% | 0.690 | 0.880 |
| TARTE + XGBoost | 8 | 5 | 3 | 22 | 1 | 87.10% | 62.50% | 83.33% | 0.815 | 0.960 |
| TabICL v2 | 6 | 4 | 2 | 23 | 2 | 87.10% | 66.67% | 66.67% | 0.793 | 0.940 |
| TabPFN | 5 | 3 | 2 | 23 | 3 | 83.87% | 60.00% | 50.00% | 0.724 | 0.920 |

TARTE + XGBoost has the highest tabular recall (83.33%), macro F1 (0.815), and
AUC (0.960). It shares the highest accuracy (87.10%) with TabICL v2. TabICL v2
has the highest precision (66.67%).

XGBoost has the lowest tabular accuracy (80.65%), precision (50.00%), and macro
F1 (0.690). XGBoost and TabPFN share the lowest recall (50.00%). LightGBM has
the lowest AUC (0.793).

## TabLLM results

TabLLM was evaluated zero-shot and has no training phase in these runs. Invalid
outputs are retained as incorrect predictions. AUC is unavailable because the
runs produce text labels rather than continuous positive-class scores.

| Model | Serialization | Positive | Negative | Invalid | Accuracy | Precision | Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BioMistral | JSON | 18 | 13 | 0 | 41.94% | 16.67% | 50.00% | 0.388 |
| BioMistral | LLM | 31 | 0 | 0 | 19.35% | 19.35% | 100.00% | 0.162 |
| BioMistral | Text template | 26 | 5 | 0 | 35.48% | 23.08% | 100.00% | 0.354 |
| Llama 3 | JSON | 0 | 31 | 0 | 80.65% | 0.00% | 0.00% | 0.446 |
| Llama 3 | LLM | 0 | 31 | 0 | 80.65% | 0.00% | 0.00% | 0.446 |
| Llama 3 | Text template | 0 | 31 | 0 | 80.65% | 0.00% | 0.00% | 0.446 |
| Meditron | JSON | 31 | 0 | 0 | 19.35% | 19.35% | 100.00% | 0.162 |
| Meditron | LLM | 0 | 0 | 31 | 0.00% | 0.00% | 0.00% | 0.000 |
| Meditron | Text template | 31 | 0 | 0 | 19.35% | 19.35% | 100.00% | 0.162 |
| Mistral | JSON | 7 | 24 | 0 | 58.06% | 0.00% | 0.00% | 0.367 |
| Mistral | LLM | 7 | 24 | 0 | 58.06% | 0.00% | 0.00% | 0.367 |
| Mistral | Text template | 12 | 19 | 0 | 48.39% | 8.33% | 16.67% | 0.374 |

The three Llama 3 configurations have the highest TabLLM accuracy (80.65%) and
macro F1 (0.446). Each is an always-negative classifier and detects none of the
six positive cases. BioMistral with text-template serialization has the highest
TabLLM precision (23.08%) and 100% recall, with 6 true positives and 20 false
positives.

Meditron with LLM serialization has the lowest accuracy and macro F1, both
zero, because all 31 outputs are invalid. The always-positive configurations
have 100% recall, 19.35% precision, 19.35% accuracy, and a macro F1 of 0.162.
Mistral with JSON and LLM serialization produces seven positive predictions in
each run, all of which are false positives.

## Recorded execution times

The following values are the training and prediction times stored by the
benchmark. Total time is their sum. TabLLM training time is zero because its
evaluation is zero-shot.

### Tabular models

| Model | Training time | Prediction time | Total time |
|---|---:|---:|---:|
| LightGBM | 0.173 s | 0.001 s | 0.174 s |
| XGBoost | 0.360 s | 0.003 s | 0.363 s |
| TabICL v2 | 0.689 s | 0.045 s | 0.734 s |
| TabPFN | 2.078 s | 0.291 s | 2.370 s |
| TARTE + XGBoost | 4.398 s | 0.243 s | 4.642 s |

LightGBM has the shortest tabular training, prediction, and total times.
TARTE + XGBoost has the longest tabular training and total times. TabPFN has
the longest tabular prediction time.

### TabLLM configurations

| Model | Serialization | Training time | Prediction time | Total time |
|---|---|---:|---:|---:|
| BioMistral | JSON | 0 s | 2.734 s | 2.734 s |
| BioMistral | LLM | 0 s | 2.343 s | 2.343 s |
| BioMistral | Text template | 0 s | 2.690 s | 2.690 s |
| Llama 3 | JSON | 0 s | 2.437 s | 2.437 s |
| Llama 3 | LLM | 0 s | 2.159 s | 2.159 s |
| Llama 3 | Text template | 0 s | 9.415 s | 9.415 s |
| Meditron | JSON | 0 s | 6.990 s | 6.990 s |
| Meditron | LLM | 0 s | 6.649 s | 6.649 s |
| Meditron | Text template | 0 s | 6.952 s | 6.952 s |
| Mistral | JSON | 0 s | 2.901 s | 2.901 s |
| Mistral | LLM | 0 s | 2.670 s | 2.670 s |
| Mistral | Text template | 0 s | 2.844 s | 2.844 s |

Llama 3 with LLM serialization has the shortest TabLLM prediction and total
time (2.159 s). Llama 3 with text-template serialization has the longest
(9.415 s). Across all reported hepatitis runs, LightGBM has the shortest total
time (0.174 s), and Llama 3 with text-template serialization has the longest
total time (9.415 s).

## Overall comparison

- TARTE + XGBoost has the highest macro F1, positive recall, and AUC across the
  configurations for which AUC is available.
- TARTE + XGBoost and TabICL v2 share the highest accuracy. TabICL v2 has the
  highest positive precision.
- Every tabular model detects at least three of the six positive cases. The
  lowest tabular macro F1 (0.690) is higher than the highest TabLLM macro F1
  (0.446).
- The highest TabLLM accuracy and macro F1 come from the three always-negative
  Llama 3 runs, which have zero positive recall.
- Meditron with LLM serialization has the lowest accuracy and macro F1 across
  all runs and an invalid-response rate of 100%.
- Tabular total times range from 0.174 s to 4.642 s. TabLLM prediction times
  range from 2.159 s to 9.415 s, with zero recorded training time.
