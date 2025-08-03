# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from pipeline import build_Pipeline
from evaluate import evaluate_model, plot_results
import warnings
import joblib

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('data/heart_cleveland.csv')

# Drop weak/unrelated features
features_to_drop = ['slope', 'chol', 'trestbps', 'fbs', 'restecg']
df_cleaned = df.drop(columns=features_to_drop)

# Binary classification: 0 (no disease), 1 (has disease)
df_cleaned['condition'] = df_cleaned['condition'].apply(lambda x: 1 if x > 0 else 0)

# Split features and target
X = df_cleaned.drop(columns=['condition'])
y = df_cleaned['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Model list
models = [
    LogisticRegression(),
    RandomForestClassifier(random_state=42),
    SVC(probability=True),
    KNeighborsClassifier(),
    GaussianNB()
]

# Evaluate all models
for model in models:
    evaluate_model(model, x_train_scaled, y_train, x_test_scaled, y_test)

# Show bar plot of accuracies
plot_results()

pipeline = build_Pipeline()

pipeline.fit(X_train,y_train)

joblib.dump(pipeline,'models/final_model.pkl')
print(" Trained pipeline saved to 'models/final_model.pkl'")