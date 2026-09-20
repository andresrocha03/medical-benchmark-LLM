# medical-benchmark-LLM

The main objective is to build a benchmark for comparing the performance of LLMs, tabular foundation models and classical tabular methods in diverse medical datasets.

The original project was done for a course in Telecom Paris, but due to personal interest, I decided to improve and correct some aspects of it.

Corrections from original project:

- Defining the same evaluation protocol for all models (F1-macro, precision, recall, accuracy, AUC, training time and prediction time).

- TabPFN was being tuned in the test set, leading do test leakage.

- Removed the threshold tuning from ICLv2, that was also leading to test leakage.

- Replaced the custom TARTE reconstruction with small, cluster-runnable modules
  around the authors' `tarte-ai` frozen featurizer.

- Introduced utils.py inside models/ to reuse functions accross model evaluations.

- Reorganized the TabLLM to integrate with the project structure.

- Included the demand for receiving standard binary answers ("positive" or "negative") when evaluating the zero-shot with LLMs.

- Documented functions.

- All models were executed with the same computational resources.

### Project structure

- **Data**: Processing data in order to make it adequate for the models.

- **Models**: Implementing, training and evaluating models.

- **Results**: Saving metrics and producing aggregated plots to compare models results.