# pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_Pipeline():
        pipeline= Pipeline([
            ('scalar',StandardScaler()),
            ('clf',RandomForestClassifier(random_state=42))
        ])
        return pipeline