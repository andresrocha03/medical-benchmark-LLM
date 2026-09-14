### Hepatitis preprocessing

[Data](https://archive.ics.uci.edu/dataset/46/hepatitis)

The script transforms the Hepatitis dataset into an ML-ready dataset. The steps are:

1. Load raw data
Reads the 155 raw patient records and assigns names to the 19 predictive attributes plus the raw `Class` label.

2. Basic cleaning
Replaces the raw missing-value marker (`?`) with `NaN`. Missing records are retained because the dataset is small and the missing feature values can be imputed.

3. Define the features
The data contains 19 predictive features: 13 categorical and 6 numerical.

| # | Feature | Description | Type |
|---|---|---|---|
| 1 | `Age` | Age in years | Numerical |
| 2 | `Sex` | Sex | Categorical |
| 3 | `Steroid` | Steroid treatment indicator | Categorical |
| 4 | `Antivirals` | Antiviral treatment indicator | Categorical |
| 5 | `Fatigue` | Fatigue indicator | Categorical |
| 6 | `Malaise` | Malaise indicator | Categorical |
| 7 | `Anorexia` | Anorexia indicator | Categorical |
| 8 | `Liver Big` | Enlarged liver indicator | Categorical |
| 9 | `Liver Firm` | Firm liver indicator | Categorical |
| 10 | `Spleen Palpable` | Palpable spleen indicator | Categorical |
| 11 | `Spiders` | Spider angiomas indicator | Categorical |
| 12 | `Ascites` | Ascites indicator | Categorical |
| 13 | `Varices` | Varices indicator | Categorical |
| 14 | `Bilirubin` | Bilirubin measurement | Numerical |
| 15 | `Alk Phosphate` | Alkaline phosphatase measurement | Numerical |
| 16 | `Sgot` | Serum glutamic-oxaloacetic transaminase measurement | Numerical |
| 17 | `Albumin` | Albumin measurement | Numerical |
| 18 | `Protime` | Prothrombin time | Numerical |
| 19 | `Histology` | Histology indicator | Categorical |

4. Build the target variable
The raw `Class` label uses `1` for death and `2` for survival. It is converted to the binary `target` column: **death from hepatitis (1 = yes; 0 = survival)**. Thus, `1` consistently represents the bad event.

5. Train/test split
Splits the data into 80% training and 20% test sets, stratified on `target` so both sets preserve the class balance.

6. Handle missing values
Categorical features are filled with the most frequent training value. Numerical features are converted to numeric and filled with the training mean. The test set uses only values learned from the training set.

7. Normalize
Applies `StandardScaler` to the six numerical features, fitting the scaler on the training data and applying it to both splits.

8. Export
Saves `hepatitis_train.csv` and `hepatitis_test.csv` in `data/pre-processed`. Each file contains the 19 features and the `target` column.
