# TabPFN Evaluation Pipeline

*Yahia SBAI (yahia.sbai@telecom-paris.fr)*

This directory contains the wrapper and evaluation scripts to test TabPFN-2.5 on preprocessed medical tabular datasets: Heart Disease, Diabetes, and Hepatitis. This pipeline is designed to integrate into the existing cluster evaluation framework to ensure a fair comparison against traditional methods such as XGBoost and LightGBM.

## Pipeline Components

1. **TabPFN Wrapper (`tabpfn_wrapper.py`)**
   - Implements `TabPFNEvaluator` to encapsulate TabPFN's initialization and fitting logic.
   - Utilizes `fit_with_cache` mode, optimized for small medical tabular datasets.
   - Includes batched prediction logic to slice data points into chunks, preventing Out-Of-Memory (OOM) errors during inference.
   - Designed to securely handle the `TABPFN_TOKEN` environment variable for authentication on non-interactive clusters (remember to insert your token in the wrapper file or set it in your environment).
   - Optimizes the decision threshold by maximizing the macro F1-score across candidate thresholds, ensuring robust performance even on highly imbalanced medical datasets.

2. **Testing Script (`run_tabpfn_tests.py`)**
   - A standalone script that loads the Diabetes, Hepatitis, and Cleveland Heart Disease datasets.
   - Executes the full TabPFN evaluation wrapper against all three datasets.

## Results and Analysis

Testing the wrapper on the full preprocessed datasets yields the following out-of-the-box performance (without explicit hyperparameter tuning):

| Dataset       | Macro F1 Score | Interpretation |
|---------------|----------------|----------------|
| **Heart Disease** | 0.8410         | **Excellent.** TabPFN shows very strong out-of-the-box performance on the preprocessed Heart Disease dataset. Its ability to implicitly model complex interactions without hyperparameter tuning makes it highly suitable for this clinical data. |
| **Diabetes**      | 0.6997         | **Good.** Given the highly engineered and complex physiological target (predicting next-day hypoglycemic events), a ~0.70 macro F1 score is a solid baseline. TabPFN effectively captures the patterns in the aggregated daily features. |
| **Hepatitis**     | 0.7163         | **Strong.** The Hepatitis dataset is notoriously imbalanced and small. By explicitly optimizing the decision threshold for the macro average F1-score, TabPFN successfully mitigates class imbalance degradation, achieving robust performance. |

### Performance Analysis: Why TabPFN Performs Better on Heart Disease

TabPFN performs exceptionally well on the **Heart Disease** dataset compared to **Diabetes** and **Hepatitis**. This performance gap can be attributed to several structural and clinical characteristics of the datasets:

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

## How to Install and Run

It is highly recommended to run this inside a Python virtual environment to avoid conflicts.

1. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate
```

2. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the evaluation script:**
```bash
python run_tabpfn_tests.py
```

## Appendix

[1] Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

[2] Kahn, M. (1994). Diabetes [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5T59G.

[3] Gong, G. (1988). Hepatitis [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5Q59J.
