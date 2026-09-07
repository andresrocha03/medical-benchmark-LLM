# Preprocessing of Hepatitis Dataset

- The dataset has 155 instances and 19 features. 12 features are categorical and 7 are numerical.

- Imported the data and separated both train and test sets.

- Filled out the missing data, as the dataset is too small to remove missing values.
    - Categorical features were filled with median and the numerical ones with the mean.

- Normalized numerical features with StandardScaler

- Saved the processed data as train_data.csv and test_data.csv

- The data is [available here](https://archive.ics.uci.edu/dataset/46/hepatitis)
