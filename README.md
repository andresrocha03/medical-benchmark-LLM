# medical-benchmark-LLM

The main objective is to build a benchmark for comparing the performance of LLMs, tabular foundation models and classical tabular methods in diverse medical datasets.

Obs:

The original project was done for a course in Telecom Paris, but due to personal interest, I decided to improve and correct some aspects of it.

Corrections from original project:

- Defining the same evaluation protocol for all models.

- TabPFN was being tuned in the test set, leading do test leakage.

- Replaced the custom TARTE reconstruction with small, cluster-runnable modules
  around the authors' `tarte-ai` frozen featurizer.

- Removed the threshold tuning from ICLv2, that was also leading to test leakage.

- Introduced utils.py inside models/ to reuse functions accross model evaluations.

- 