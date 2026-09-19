# Diabetes: class imbalance and model behavior

## Class distribution

The target is whether a patient will have a hypoglycemic event the following
day. The positive class is substantially less frequent than the negative
class.

| Split | Negative | Positive | Negative:positive ratio |
|---|---:|---:|---:|
| Training | 2,801 (93.71%) | 188 (6.29%) | 14.90:1 |
| Test | 720 (87.70%) | 101 (12.30%) | 7.13:1 |

The preprocessing uses a patient-disjoint group split, so the training and
test proportions are not identical. On the test set, a classifier that always
predicts `negative` obtains **87.70% accuracy**, despite having **0% positive
recall** and a **macro F1 of 0.467**. Accuracy must therefore be read together
with positive-class recall, macro F1, and the confusion counts.

## Tabular-model behavior

None of the evaluated tabular-model wrappers applies class weights, positive
class weighting, over-sampling, or under-sampling. All consequently learn from
the imbalanced training data as provided. LightGBM and XGBoost use stratified
cross-validation and select hyperparameters by macro F1, which reduces the
influence of majority-class accuracy during model selection but does not
rebalance training. TabPFN and TabICL v2 receive the same imbalanced training
context with their default classifiers. TARTE uses a frozen table encoder
followed by an unweighted XGBoost classifier.

All five models are conservative about predicting the positive class:

| Model | Predicted positive | TP | FP | TN | FN | Accuracy | Positive recall | Macro F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 31 | 25 | 6 | 714 | 76 | 90.01% | 24.75% | 0.662 | 0.657 |
| XGBoost | 31 | 23 | 8 | 712 | 78 | 89.52% | 22.77% | 0.646 | 0.676 |
| TARTE + XGBoost | 30 | 24 | 6 | 714 | 77 | 89.89% | 23.76% | 0.656 | 0.676 |
| TabICL v2 | 26 | 23 | 3 | 717 | 78 | 90.13% | 22.77% | 0.654 | 0.678 |
| TabPFN | 22 | 20 | 2 | 718 | 81 | 89.89% | 19.80% | 0.635 | 0.685 |

These models perform better than the always-negative baseline in macro F1 and
identify some positive cases. However, every model misses at least 76 of the
101 positive instances. Their approximately 90% accuracy is driven largely by
the 720 negative test instances. LightGBM has the highest positive recall in
this comparison, but it still detects only about one quarter of positive
cases.

## TabLLM answer patterns

TabLLM is evaluated zero-shot: it does not receive the training split or its
class proportions. It is instructed to return either `negative` or `positive`.
The results show response collapse rather than instance-level discrimination
for most model and serialization combinations.

| Model | Serialization | Parsed predictions | Invalid responses | Consequence |
|---|---|---|---:|---|
| BioMistral | JSON, LLM, text template | 821 negative | 0 | Always-negative baseline behavior |
| Llama 3 | JSON, LLM, text template | 821 negative | 0 | Always-negative baseline behavior |
| Mistral | JSON | 821 negative | 0 | Always-negative baseline behavior |
| Mistral | Text template | 821 negative | 0 | Always-negative baseline behavior |
| Mistral | LLM | 820 negative, 1 positive | 0 | The one positive prediction is a false positive; positive recall remains 0% |
| Meditron | JSON | 821 positive | 0 | Detects all positives but misclassifies all 720 negatives |
| Meditron | LLM | 16 positive, 805 invalid | 805 (98.05%) | All 16 positive predictions are false positives; accuracy is 0% |
| Meditron | Text template | 13 positive, 808 invalid | 808 (98.42%) | One true positive, 12 false positives, and 0.12% accuracy |

Eight of the twelve TabLLM configurations return the same parsed `negative`
label for every test instance. Meditron with JSON serialization collapses in
the opposite direction, returning responses that always parse as `positive`.
Mistral with the LLM serialization is nearly constant, with 820 negative
predictions and one positive prediction.

The invalid-response problem is concentrated in Meditron with the LLM and
text-template serializations. These runs produce prompt or patient-record
echoes that contain neither valid label for 805 and 808 instances,
respectively. Invalid responses are retained as incorrect predictions rather
than silently removed from evaluation. Meditron's JSON responses are also not
direct label-only answers, but they contain `positive` and are therefore parsed
as positive.

## Interpretation

- The imbalance favors models that predict the negative class when accuracy is
  used in isolation.
- LightGBM and XGBoost partially address evaluation bias by selecting their
  hyperparameters with macro F1, but no evaluated tabular model explicitly
  corrects the training imbalance.
- The tabular models learn some minority-class signal, but their positive
  recall remains between 19.80% and 24.75%.
- Most TabLLM configurations do not demonstrate meaningful class separation:
  they produce a constant or nearly constant label. Meditron's two high-invalid
  runs additionally fail to follow the required output format.