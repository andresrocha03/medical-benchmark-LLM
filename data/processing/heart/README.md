### Heart preprocessing

[Data](http://archive.ics.uci.edu/dataset/45/heart+disease)

The script transforms the Cleveland heart-disease records into an ML-ready dataset. The steps are:

1. Load raw data
Reads the raw Cleveland file as a sequence of 76 attributes per patient and extracts the 14 attributes used by this project. This produces 297 usable rows before cleaning.

2. Basic cleaning
Replaces the raw missing-value marker (`-9`) with `NaN` and converts every selected field to numeric. Rows whose diagnosis is outside the valid 0–4 range are dropped, leaving 286 rows.

3. Define the features
The data contains 13 predictive features: 8 categorical and 5 numerical.

| # | Feature | Description | Type |
|---|---|---|---|
| 1 | `age` | Age in years | Numerical |
| 2 | `sex` | Sex (1 = male, 0 = female) | Categorical |
| 3 | `cp` | Chest-pain type | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| 5 | `chol` | Serum cholesterol (mg/dl) | Numerical |
| 6 | `fbs` | Fasting blood sugar above 120 mg/dl | Categorical |
| 7 | `restecg` | Resting ECG result | Categorical |
| 8 | `thalach` | Maximum heart rate achieved | Numerical |
| 9 | `exang` | Exercise-induced angina | Categorical |
| 10 | `oldpeak` | ST depression induced by exercise | Numerical |
| 11 | `slope` | Slope of the peak exercise ST segment | Categorical |
| 12 | `ca` | Number of major vessels coloured by fluoroscopy | Categorical |
| 13 | `thal` | Thallium stress-test result | Categorical |

4. Build the target variable
The raw diagnosis ranges from 0 to 4. It is converted to the binary `target` column: **heart disease present (1 = yes; 0 = no)**. Any raw diagnosis from 1 to 4 is treated as disease present.

5. Train/test split
Splits the data into 80% training and 20% test sets, stratified on `target` so both sets preserve the class balance.

6. Handle missing values
Categorical features are filled with the most frequent training value. Numerical features are filled with the training mean. The test set uses only values learned from the training set.

7. Normalize
Applies `StandardScaler` to the five numerical features, fitting the scaler on the training data and applying it to both splits.

8. Export
Saves `heart_train.csv` and `heart_test.csv` in `data/pre-processed`. Each file contains the 13 features and the `target` column.
