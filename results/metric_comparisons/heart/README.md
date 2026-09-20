# Heart disease: model performance and response behavior

## Class distribution

The target indicates whether heart disease is present. The positive class is
heart disease and the negative class is no heart disease.

| Split | Negative | Positive | Negative:positive ratio |
|---|---:|---:|---:|
| Training | 125 (54.82%) | 103 (45.18%) | 1.21:1 |
| Test | 32 (55.17%) | 26 (44.83%) | 1.23:1 |

The splits have similar class proportions. On the test set, an always-negative
classifier obtains 55.17% accuracy, 0% positive recall, and a macro F1 of
0.356. An always-positive classifier obtains 44.83% accuracy, 100% positive
recall, and a macro F1 of 0.310.

## Tabular-model results

All five tabular models produced both positive and negative predictions.

| Model | Predicted positive | TP | FP | TN | FN | Accuracy | Precision | Recall | Macro F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 23 | 19 | 4 | 28 | 7 | 81.03% | 82.61% | 73.08% | 0.806 | 0.882 |
| XGBoost | 25 | 20 | 5 | 27 | 6 | 81.03% | 80.00% | 76.92% | 0.808 | 0.869 |
| TARTE + XGBoost | 27 | 20 | 7 | 25 | 6 | 77.59% | 74.07% | 76.92% | 0.774 | 0.897 |
| TabICL v2 | 25 | 21 | 4 | 28 | 5 | 84.48% | 84.00% | 80.77% | 0.843 | 0.910 |
| TabPFN | 25 | 20 | 5 | 27 | 6 | 81.03% | 80.00% | 76.92% | 0.808 | 0.913 |

TabICL v2 has the highest tabular accuracy (84.48%), precision (84.00%),
recall (80.77%), and macro F1 (0.843). TabPFN has the highest AUC (0.913),
followed by TabICL v2 (0.910). TARTE + XGBoost has the lowest tabular accuracy
(77.59%), precision (74.07%), and macro F1 (0.774). XGBoost has the lowest AUC
(0.869). LightGBM has the lowest positive recall (73.08%).

XGBoost and TabPFN produced the same hard-label confusion counts and therefore
the same accuracy, precision, recall, and macro F1. Their AUC values differ
because AUC uses the models' continuous scores rather than their thresholded
class predictions.

## TabLLM results

TabLLM was evaluated zero-shot and has no training phase in these runs. The
table reports the parsed output counts and predictive metrics for each model
and serialization. Invalid outputs are retained as incorrect predictions.

| Model | Serialization | Positive | Negative | Invalid | Accuracy | Precision | Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BioMistral | JSON | 58 | 0 | 0 | 44.83% | 44.83% | 100.00% | 0.310 |
| BioMistral | LLM | 58 | 0 | 0 | 44.83% | 44.83% | 100.00% | 0.310 |
| BioMistral | Text template | 53 | 5 | 0 | 53.45% | 49.06% | 100.00% | 0.464 |
| Llama 3 | JSON | 0 | 58 | 0 | 55.17% | 0.00% | 0.00% | 0.356 |
| Llama 3 | LLM | 17 | 41 | 0 | 60.34% | 58.82% | 38.46% | 0.575 |
| Llama 3 | Text template | 1 | 57 | 0 | 56.90% | 100.00% | 3.85% | 0.397 |
| Meditron | JSON | 58 | 0 | 0 | 44.83% | 44.83% | 100.00% | 0.310 |
| Meditron | LLM | 2 | 0 | 56 | 0.00% | 0.00% | 0.00% | 0.000 |
| Meditron | Text template | 19 | 0 | 39 | 5.17% | 15.79% | 11.54% | 0.067 |
| Mistral | JSON | 8 | 50 | 0 | 62.07% | 75.00% | 23.08% | 0.542 |
| Mistral | LLM | 5 | 53 | 0 | 56.90% | 60.00% | 11.54% | 0.450 |
| Mistral | Text template | 26 | 32 | 0 | 68.97% | 65.38% | 65.38% | 0.686 |

Mistral with text-template serialization has the highest TabLLM accuracy
(68.97%) and macro F1 (0.686). Its 26 positive and 32 negative output counts
equal the test-set class counts, although its confusion matrix contains 17
true positives, 9 false positives, 23 true negatives, and 9 false negatives.

Llama 3 with text-template serialization has the highest precision (100.00%),
based on one positive prediction, and its positive recall is 3.85%. The three
always-positive runs have 100% recall and 44.83% precision. Llama 3 with JSON
serialization is always negative and has 0% positive recall.

Meditron with LLM serialization has the lowest accuracy and macro F1, both
zero, with 56 invalid outputs and two false-positive outputs. Meditron with
text-template serialization has the next-lowest accuracy (5.17%) and macro F1
(0.067), with 39 invalid outputs. No AUC is reported for TabLLM because these
runs produce text labels rather than continuous positive-class scores.

## Recorded execution times

The following values are the training and prediction times stored by the
benchmark. Total time is their sum. TabLLM training time is zero because its
evaluation is zero-shot.

### Tabular models

| Model | Training time | Prediction time | Total time |
|---|---:|---:|---:|
| LightGBM | 0.194 s | 0.001 s | 0.195 s |
| XGBoost | 0.339 s | 0.002 s | 0.340 s |
| TabICL v2 | 0.556 s | 0.042 s | 0.598 s |
| TabPFN | 1.436 s | 0.266 s | 1.702 s |
| TARTE + XGBoost | 5.868 s | 0.251 s | 6.119 s |

LightGBM has the shortest tabular training, prediction, and total times.
TARTE + XGBoost has the longest tabular training and total times. TabPFN has
the longest tabular prediction time.

### TabLLM configurations

| Model | Serialization | Training time | Prediction time | Total time |
|---|---|---:|---:|---:|
| BioMistral | JSON | 0 s | 4.131 s | 4.131 s |
| BioMistral | LLM | 0 s | 3.695 s | 3.695 s |
| BioMistral | Text template | 0 s | 4.068 s | 4.068 s |
| Llama 3 | JSON | 0 s | 4.020 s | 4.020 s |
| Llama 3 | LLM | 0 s | 3.525 s | 3.525 s |
| Llama 3 | Text template | 0 s | 3.662 s | 3.662 s |
| Meditron | JSON | 0 s | 12.716 s | 12.716 s |
| Meditron | LLM | 0 s | 12.442 s | 12.442 s |
| Meditron | Text template | 0 s | 12.560 s | 12.560 s |
| Mistral | JSON | 0 s | 4.420 s | 4.420 s |
| Mistral | LLM | 0 s | 4.005 s | 4.005 s |
| Mistral | Text template | 0 s | 4.385 s | 4.385 s |

Llama 3 with LLM serialization has the shortest TabLLM prediction time
(3.525 s). Meditron with JSON serialization has the longest (12.716 s).
Across all reported runs, LightGBM has the shortest total time (0.195 s), and
Meditron with JSON serialization has the longest total time (12.716 s).

## Overall comparison

- TabICL v2 has the highest macro F1 and accuracy across all tabular and
  TabLLM configurations.
- TabPFN has the highest available AUC. AUC is unavailable for TabLLM.
- Mistral with text-template serialization is the highest-performing TabLLM
  configuration by accuracy and macro F1, but both values are below those of
  every evaluated tabular model.
- Meditron with LLM serialization has the lowest accuracy and macro F1 across
  all configurations; 96.55% of its outputs are invalid.
- The five tabular models have total times from 0.195 s to 6.119 s. The TabLLM
  configurations have prediction times from 3.525 s to 12.716 s and zero
  recorded training time.
