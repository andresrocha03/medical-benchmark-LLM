# medical-benchmark-LLM
The main objective is to build a benchmark for comparing the performance of LLMs, tabular foundarion models and classical tabular methods in diverse medical datasets.


Corrections from original project:

- Rerun the traditional model results using AUC, F1 (Macro), Precision, Recall, Training time and Prediction time as metrics.

- Defining the same evaluation protocol for all models.

- TabPFN was being tuned in the test set, leading do test leakage.

- 