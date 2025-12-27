import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
import preprocessing

def train():
    # 1. Load Data
    print("Loading data...")
    df = preprocessing.load_data("e:/acm/problems_data.jsonl")
    df = preprocessing.combine_text_fields(df)
    df = preprocessing.add_features(df)
    df['cleaned_text'] = df['combined_text'].apply(preprocessing.clean_text)

    # 2. Split Data
    # Target for classification: problem_class (Easy, Medium, Hard)
    # Target for regression: problem_score
    print("Splitting data...")
    X = df[['cleaned_text', 'math_symbol_count', 'text_len', 'word_count']]
    y_class = df['problem_class']
    y_score = df['problem_score']

    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42
    )

    # 3. Model Definition
    # We need a pipeline that handles text (TF-IDF) and numeric features
    
    # Text pipeline
    text_features = 'cleaned_text'
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
    ])

    # Numeric pipeline
    numeric_features = ['math_symbol_count', 'text_len', 'word_count']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_features),
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # --- Classification Models ---
    print("Training Classification Models...")
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100),
        'SVM': SVC()
    }

    best_clf_name = None
    best_clf_score = 0
    best_clf_model = None

    for name, clf in classifiers.items():
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', clf)])
        model.fit(X_train, y_class_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_class_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_clf_score:
            best_clf_score = acc
            best_clf_name = name
            best_clf_model = model

    print(f"Best Classification Model: {best_clf_name}")

    # --- Regression Models ---
    print("Training Regression Models...")
    regressors = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    best_reg_name = None
    best_reg_score = float('inf') # Using MAE (lower is better)
    best_reg_model = None

    for name, reg in regressors.items():
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', reg)])
        model.fit(X_train, y_score_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_score_test, preds)
        print(f"{name} MAE: {mae:.4f}")
        
        if mae < best_reg_score:
            best_reg_score = mae
            best_reg_name = name
            best_reg_model = model
            
    print(f"Best Regression Model: {best_reg_name}")

    # 4. Save Models
    print("Saving models...")
    joblib.dump(best_clf_model, 'e:/acm/model_class.pkl')
    joblib.dump(best_reg_model, 'e:/acm/model_score.pkl')
    print("Models saved.")

if __name__ == "__main__":
    train()
