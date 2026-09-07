### Diabetes preprocessing

The script transforms 70 raw diabetes patient logs into ML-ready dataset. The steps are:

1. Load raw data
Reads all 70 patient files (data-01 to data-70), each being a tab-separated event log with 4 columns: date, time, code, value. Concatenates them into a single DataFrame and adds a patient_id column to track which patient each row belongs to.

2. Basic cleaning
Drops rows with unrecognized codes (stray values like 3A, 22, 21 found in the raw files)
Parses date + time into a proper datetime column
Converts value to numeric, dropping anything that can't be parsed

3. Pivot to one row per (patient, day)
This step aggregates all events within a day for each patient into a single feature vector of 14 features (insulin totals, glucose statistics, hypoglycemic event count, meal pattern, exercise pattern).

The `daily_features` function computes exactly 14 quantities per day:

| # | Feature | Description | Codes|
|---|---------|-------------|------|
| 1 | `insulin_regular_total` | Sum of regular insulin doses | 33, 34, 35
| 2 | `insulin_regular_doses` | Number of regular insulin injections | 33, 34, 35
| 3 | `insulin_nph_total` | Sum of NPH insulin doses | 33, 34, 35
| 4 | `insulin_nph_doses` | Number of NPH injections | 33, 34, 35
| 5 | `insulin_ultralente_total` | Sum of UltraLente insulin doses | 33, 34, 35
| 6 | `insulin_ultralente_doses` | Number of UltraLente injections | 33, 34, 35
| 7 | `glucose_mean` | Mean of all glucose readings |48, 57–64
| 8 | `glucose_min` | Minimum glucose reading | 48, 57–64
| 9 | `glucose_max` | Maximum glucose reading | 48, 57–64
| 10 | `glucose_std` | Standard deviation of glucose readings | 48, 57–64
| 11 | `glucose_count` | Number of glucose measurements | 48, 57–64
| 12 | `hypoglycemic_events` | Count of hypoglycemic symptom events | 65
| 13 | `meal_pattern` | Average meal intensity (0=less, 1=typical, 2=more) | 66, 67, 68
| 14 | `exercise_pattern` | Average exercise intensity (0=less, 1=typical, 2=more) |69, 70, 71

4. Build the target variable
Since there's no pre-existing label, it creates a binary target: **will the patient have a hypoglycemic event tomorrow? (1 = yes, 0 = no).** The last day of each patient is dropped since it has no "next day".

5. Train/test split
Splits into 80% train and 20% test, stratified on the target so both sets have the same proportion of positive cases.

6. Handle missing values
Fills in NaNs that appeared during feature engineering (e.g. days with no glucose measurement). Count/sum columns get 0, statistical columns get the training mean.

7. Normalize
Applies StandardScaler (fit on train only, applied to both) to bring all features to the same scale.

8. Export
Saves train_data.csv and test_data.csv.