# medical-benchmark-LLM
The main objective is to build a benchmark for comparing the performance of LLMs, tabular foundation models and classical tabular methods in diverse medical datasets.



Corrections from original project:

- Defining the same evaluation protocol for all models.

- TabPFN was being tuned in the test set, leading do test leakage.

- Converted Tarte code from a notebook to distinct modules in order to execute them all on a cluster.

- Removed the threshold tuning from ICLv2, that was also leading to test leakage.