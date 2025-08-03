import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import build_Pipeline
from evaluate import evaluate_model

# Load data
df = pd.read_csv("data/heart_cleveland.csv")

# Preprocess
df['condition'] = df['condition'].apply(lambda x: 1 if x > 0 else 0)
features_to_drop = ['slope', 'chol', 'trestbps', 'fbs', 'restecg']
df = df.drop(columns=features_to_drop)

X = df.drop('condition', axis=1)
y = df['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate pipeline
pipeline = build_Pipeline()
evaluate_model(pipeline, X_train, y_train, X_test, y_test)

