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


### Performance Analysis: Why TabPFN Performs Better on Heart Disease

TabPFN performs better on the **Heart Disease** dataset compared to **Diabetes** and **Hepatitis**. This performance gap can be attributed to several structural and clinical characteristics of the datasets:

1. **Class Balance**:
   - **Heart Disease**: Highly balanced distribution (54.8% negative vs. 45.2% positive). This allows TabPFN to learn distinct, unbiased decision boundaries for both classes.
   - **Diabetes & Hepatitis**: Highly imbalanced (Diabetes: 7.4% minority class; Hepatitis: 20.2% minority class). Severe class imbalance makes maximizing the Macro F1 score much more challenging, as predicting the majority class dominates the default model output.

2. **Feature Predictability & Medical Directness**:
   - **Heart Disease**: Contains direct clinical and physiological measurements (e.g., chest pain type, max heart rate, ST depression) that are strongly and directly correlated with coronary artery disease.
   - **Diabetes**: The target is predicting a complex *next-day* physiological event (hypoglycemic event) based on aggregated daily physiological features. This temporal prediction task is inherently noisier and less deterministic.
   - **Hepatitis**: Primarily comprised of binary clinical symptoms (e.g., fatigue, spiders, malaise) which are weak individual predictors and highly subjective compared to numerical lab tests.

3. **Data Quality and Missing Information**:
   - **Heart Disease**: Clean, standard numerical and categorical features with complete columns.
   - **Hepatitis**: Extremely small sample size (124 train samples) and sparse features, which limits TabPFN's ability to recognize generalizable patterns out of the box.