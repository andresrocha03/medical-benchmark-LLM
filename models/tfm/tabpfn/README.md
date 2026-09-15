# TabPFN Evaluation Pipeline


This directory contains the wrapper and evaluation scripts to test TabPFN on preprocessed medical tabular datasets: Heart Disease, Diabetes, and Hepatitis. This pipeline is designed to integrate into the existing cluster evaluation framework to ensure a fair comparison against traditional methods such as XGBoost and LightGBM.

## Pipeline Components

1. **TabPFN Wrapper (`wrapper.py`)**
   - Implements `TabPFNEvaluator` to encapsulate TabPFN's initialization and fitting logic.
   - Utilizes `fit_with_cache` mode, in order to obtain faster predict calls.
   - Includes batched prediction logic to slice data points into chunks, preventing Out-Of-Memory (OOM) errors during inference.
   - Remember to insert your token in the wrapper file or set it in your environment.

2. **Testing Script (`run_tests.py`)**
   - A standalone script that loads the Diabetes, Hepatitis, and Cleveland Heart Disease datasets.
   - Executes the full TabPFN evaluation wrapper against all three datasets.

