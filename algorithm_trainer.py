from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

def _encode_features(X: pd.DataFrame):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X

def train_and_evaluate(X, y):
    results = []
    try:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)

        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y)

        X_proc = _encode_features(X)

        if not X_proc.empty:
            X_proc = pd.DataFrame(StandardScaler().fit_transform(X_proc), columns=X_proc.columns)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }

        if len(X_proc) < 8 or len(np.unique(y)) < 2:
            for name, model in models.items():
                try:
                    model.fit(X_proc, y)
                    acc = model.score(X_proc, y)
                    results.append((name, float(acc)))
                except Exception as e:
                    print(f"{name} failed on tiny set: {e}")
            return results

        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42, stratify=strat)

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results.append((name, float(acc)))
            except Exception as e:
                print(f"{name} failed: {e}")
    except Exception as e:
        print("Training failed:", e)
    return results