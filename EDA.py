import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib

# Set visual style 
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Load and Inspect Data
df = pd.read_csv('credit_card_fraud_dataset.csv')
print("=== Initial Data Overview ===")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Data Cleaning
# Convert TransactionDate to datetime and extract features
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionMonth'] = df['TransactionDate'].dt.month

# 3. Feature Engineering
# Create time-based features
df['HourCategory'] = pd.cut(df['TransactionHour'],
                           bins=[0, 6, 12, 18, 24],
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df['TimeCategory'] = pd.cut(df['TransactionHour'],
                           bins=[0, 6, 12, 18, 24],
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# Create amount categories
df['AmountCategory'] = pd.qcut(df['Amount'], q=4,
                              labels=['Low', 'Medium', 'High', 'Very High'])

# 4. Visual EDA - First Graph Output
plt.figure(figsize=(18, 12))

# Transaction Amount Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['Amount'], kde=True, bins=30)
plt.title('Transaction Amount Distribution')

# Fraud vs Non-Fraud Transactions
plt.subplot(2, 2, 2)
fraud_counts = df['IsFraud'].value_counts()
plt.pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%')
plt.title('Class Distribution')

# Transaction Type Analysis
plt.subplot(2, 2, 3)
sns.countplot(x='TransactionType', hue='IsFraud', data=df)
plt.title('Transaction Type by Fraud Status')

# Location Analysis
plt.subplot(2, 2, 4)
top_locations = df['Location'].value_counts().nlargest(5).index
sns.countplot(x='Location', hue='IsFraud', 
              data=df[df['Location'].isin(top_locations)], order=top_locations)
plt.title('Top 5 Locations by Fraud Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Second Graph Output - Time-based Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Hourly fraud rate
hourly_rate = df.groupby('TransactionHour')['IsFraud'].mean()
sns.lineplot(x=hourly_rate.index, y=hourly_rate.values, ax=axes[0])
axes[0].set_title('Hourly Fraud Rate')

# Time category analysis
sns.barplot(x='TimeCategory', y='IsFraud', data=df, ax=axes[1])
axes[1].set_title('Fraud Rate by Time Category')

plt.tight_layout()
plt.show()

# 6. Feature Processing
# Separate features and target
X = df.drop(['TransactionID', 'TransactionDate', 'IsFraud'], axis=1)
y = df['IsFraud']

# Define preprocessing
numeric_features = ['Amount', 'MerchantID', 'TransactionHour', 'TransactionDay', 'TransactionMonth']
categorical_features = ['TransactionType', 'Location', 'HourCategory', 'AmountCategory']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# 7. Handle Class Imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n=== Class Distribution After SMOTE ===")
print(pd.Series(y_train_smote).value_counts())

# 8. Save Processed Data
processed_data = {
    'X_train': X_train_smote,
    'X_test': X_test,
    'y_train': y_train_smote,
    'y_test': y_test,
    'preprocessor': preprocessor
}

import joblib
joblib.dump(processed_data, 'processed_credit_data.pkl')
print("\nProcessing complete. Data saved to 'processed_credit_data.pkl'")