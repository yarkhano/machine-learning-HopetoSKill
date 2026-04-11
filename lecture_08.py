#implementing logistic regression
#imported preprocessed data from previous part

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# 1. Load Data
dataset = sns.load_dataset("titanic")
print("Shape:", dataset.shape)
print(dataset.head(10))
print(dataset.describe()) # Statistical summary of numerical columns
print(dataset.info())

# 2. Starting on preprocessing
print("\nMissing Values:\n", dataset.isnull().sum())
print("\nData Types:\n", dataset.dtypes)

# Filling missing values
dataset["age"] = dataset["age"].fillna(dataset["age"].mean())
dataset["embarked"] = dataset["embarked"].fillna(dataset["embarked"].mode()[0])

# Dropping deck column because it has 77% of data missing
dataset.drop(columns=["deck"], inplace=True)
dataset.drop(columns=["embark_town"], inplace=True)
print("\nMissing Values After Initial Drops:\n", dataset.isnull().sum())

# 3. Label and One-Hot Encoding
label_encoder = LabelEncoder()
dataset["sex"] = label_encoder.fit_transform(dataset["sex"])
print("\nAfter Label Encoding (Sex):\n", dataset.head(10))

dataset = pd.get_dummies(dataset, columns=["embarked"], dtype=int)
print("\nAfter One-Hot Encoding (Embarked):\n", dataset.head(10))

# 4. Outlier Detection and Handling
# Note: We do this BEFORE scaling so the scaling math isn't distorted by outliers
q1 = dataset["fare"].quantile(0.25)
q3 = dataset["fare"].quantile(0.75)
dq = q3 - q1

lower_boundary = q1 - 1.5 * dq
upper_boundary = q3 + 1.5 * dq

print("\nq1:", q1)
print("q3:", q3)
print("IQR:", dq)
print("Lower Boundary:", lower_boundary)
print("Upper Boundary:", upper_boundary)

outliers = dataset[(dataset["fare"] < lower_boundary) | (dataset["fare"] > upper_boundary)]
print("Number of outliers detected:", len(outliers))

# Capping outliers
dataset["fare"] = dataset["fare"].clip(lower=lower_boundary, upper=upper_boundary)
outliers_after = dataset[(dataset["fare"] < lower_boundary) | (dataset["fare"] > upper_boundary)]
print("Number of outliers after clipping:", len(outliers_after))


#zscore is also implemented but in this case zscore is not good because it will move mean upward of fare
# z_scores = np.abs(zscore(dataset[["fare"]]))
# outliers = (z_scores>3).any(axis=1)
# rm_outliers = dataset[~outliers]


# 5. Feature Scaling
# Using standardization since it is robust for models like Logistic Regression
scaler = StandardScaler()
cols = ["age", "fare", "sibsp", "parch"]
dataset[cols] = scaler.fit_transform(dataset[cols])
print("\nAfter Scaling:\n", dataset.head(10))
print(dataset.describe())


# Normalization
#
# normalizer = MinMaxScaler()
# cols = ["age", "fare", "sibsp", "parch"]
# dataset[cols] = normalizer.fit_transform(dataset[cols])
# print("\nAfter Scaling:\n", dataset.head(10))
# print(dataset.describe())


# 6. Removing Duplicates
# ISSUE FIXED: changed 'drop_duplicated' to 'drop_duplicates' and added 'inplace=True'
dataset.drop_duplicates(inplace=True)

# 7. Feature Selection
correlation = dataset.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation)

# Removing adult_male column because it is highly redundant with the 'sex' column
dataset.drop(columns=["adult_male"], inplace=True)

print("\nFinal Preprocessing Complete. Final Shape:", dataset.shape)