import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = "data/phishing_dataset.csv"
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")

# Display the first few rows
print(data.head())

# Display basic info about the dataset
print(data.info())

# Summary statistics
print(data.describe())

# Select only numeric columns for filling missing values
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Fill missing values in numeric columns with the mean
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())

# Verify if missing values are handled
missing_values_after = data.isnull().sum()
print("\nMissing Values After Handling:\n", missing_values_after)

from sklearn.preprocessing import LabelEncoder

# Label Encoding for the 'label' column (phishing vs benign)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

print("\nEncoded Labels:")
print(data['label'].head())

from sklearn.preprocessing import StandardScaler

# Features (excluding the label)
X = data.drop(columns=['label'])

# Label (target variable)
y = data['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")
print(X_scaled[:5])  # Display the first 5 scaled feature values

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining and Testing Set Sizes:")
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")
