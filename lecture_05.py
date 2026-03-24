import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
dataset = sns.load_dataset("titanic")

# 2. Handling Missing Values
dataset["age"] = dataset["age"].fillna(dataset["age"].mean())
dataset["embarked"] = dataset["embarked"].fillna(dataset["embarked"].mode()[0])

# Dropping unnecessary/messy columns
dataset.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male"], inplace=True)

# 3. Label & One-Hot Encoding
label_encoder = LabelEncoder()
dataset["sex"] = label_encoder.fit_transform(dataset["sex"])
dataset = pd.get_dummies(dataset, columns=["embarked"], dtype=int)

# 4. Outlier Handling (DO THIS BEFORE SCALING)
# Calculating IQR for 'fare'
q1 = dataset["fare"].quantile(0.25)
q3 = dataset["fare"].quantile(0.75)
iqr = q3 - q1

lower_boundary = q1 - 1.5 * iqr
upper_boundary = q3 + 1.5 * iqr

# Capping outliers using the calculated boundaries
dataset["fare"] = dataset["fare"].clip(lower=lower_boundary, upper=upper_boundary)

# 5. Feature Scaling
scaler = StandardScaler()
cols_to_scale = ["age", "fare", "sibsp", "parch"]
dataset[cols_to_scale] = scaler.fit_transform(dataset[cols_to_scale])

# 6. Removing Duplicates (Fixed typo and added inplace)
dataset.drop_duplicates(inplace=True)

# 7. Final Check
print("Final Shape:", dataset.shape)
print(dataset.head())

# Feature Selection (Correlation)
correlation = dataset.corr(numeric_only=True)
print("\nCorrelation with Survived:")
print(correlation["survived"].sort_values(ascending=False))