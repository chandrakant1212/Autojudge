
import pandas as pd
import numpy as np
import joblib
import nltk
import re
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
try:
    from preprocessing import clean_text, add_features, combine_text_fields
except ImportError:
    # If running from root, src.preprocessing might differ.
    # But clean_text is simple, let's redefine if needed or ensure path is right.
    # Assuming this script is in src/
    from preprocessing import clean_text, add_features, combine_text_fields

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

def load_and_merge_data():
    # Paths
    original_path = 'problems_data.jsonl'
    leetcode_path = 'leetcode_datase.csv'
    kattis_path = 'kattis_dataset.csv'

    print("Loading Original Data...")
    try:
        df_orig = pd.read_json(original_path, lines=True)
    except ValueError:
        df_orig = pd.read_json(original_path)
    
    # Process Original
    df_orig = combine_text_fields(df_orig) # Creates combined_text
    df_orig = df_orig[['combined_text', 'problem_class', 'problem_score']]

    print("Loading LeetCode Data...")
    df_leet = pd.read_csv(leetcode_path)
    df_leet['combined_text'] = df_leet['title'] + " " + df_leet['description']
    
    # Map LeetCode
    difficulty_map = {'Easy': 'Easy', 'Medium': 'Medium', 'Hard': 'Hard'}
    score_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
    df_leet['problem_class'] = df_leet['difficulty'].map(difficulty_map)
    df_leet['problem_score'] = df_leet['difficulty'].map(score_map)
    df_leet = df_leet[['combined_text', 'problem_class', 'problem_score']].copy()

    print("Loading Kattis Data...")
    df_kattis = pd.read_csv(kattis_path, usecols=['question', 'difficulty'])
    df_kattis = df_kattis.rename(columns={'question': 'combined_text'})
    
    # Map Kattis
    kattis_map = {'introductory': 'Easy', 'interview': 'Medium', 'competition': 'Hard'}
    kattis_score = {'introductory': 1, 'interview': 2, 'competition': 3}
    df_kattis['problem_class'] = df_kattis['difficulty'].map(kattis_map)
    df_kattis['problem_score'] = df_kattis['difficulty'].map(kattis_score)
    df_kattis = df_kattis[['combined_text', 'problem_class', 'problem_score']].dropna().copy()

    print("Merging datasets...")
    df_final = pd.concat([df_orig, df_leet, df_kattis], ignore_index=True)
    print(f"Total samples: {len(df_final)}")
    
    return df_final

def train_full():
    df = load_and_merge_data()
    
    print("Pre-processing text and extracting features (this may take a minute)...")
    # Apply cleaning (using imported function for consistency)
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Add features (using imported function)
    # add_features expects 'combined_text' and adds 'text_len', 'word_count', 'math_symbol_count'
    df = add_features(df)
    
    # Features and Targets
    X = df[['cleaned_text', 'text_len', 'word_count', 'math_symbol_count']]
    y_class = df['problem_class']
    # Ensure problem_score is numeric
    y_score = pd.to_numeric(df['problem_score'], errors='coerce').fillna(0)

    # Split
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42
    )

    # Pipeline Definition
    text_features = 'cleaned_text'
    numeric_features = ['text_len', 'word_count', 'math_symbol_count']

    text_transformer = TfidfVectorizer(max_features=1000)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_features),
            ('num', numeric_transformer, numeric_features)
        ])

    # 1. Classification (Random Forest)
    print("Training Classification Model (Random Forest)...")
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])
    clf_pipeline.fit(X_train, y_class_train)
    print("Classification Accuracy:", clf_pipeline.score(X_test, y_class_test))

    # 2. Regression (Random Forest)
    print("Training Regression Model (Random Forest)...")
    reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    reg_pipeline.fit(X_train, y_score_train)
    mae = mean_absolute_error(y_score_test, reg_pipeline.predict(X_test))
    print("Regression MAE:", mae)

    # Save Models
    print("Saving models...")
    joblib.dump(clf_pipeline, 'model_class.pkl')
    joblib.dump(reg_pipeline, 'model_score.pkl')
    print("Done.")

if __name__ == "__main__":
    train_full()
