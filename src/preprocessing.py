import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (idempotent)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text to match training logic:
    1. Lowercasing
    2. Removing HTML tags
    3. Keeping alphanumeric and math symbols ($)
    4. Removing stopwords
    """
    if pd.isna(text):
        return ""
    
    # 1. Lowercase and string conversion
    text = str(text).lower()
    
    # 2. Remove HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Keep alphanumeric and math symbols ($), remove others
    # This regex matches anything that is NOT (^) a-z, 0-9, whitespace (\s), or $
    text = re.sub(r'[^a-z0-9\s$]', '', text) 
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords
    # Create set inside function or ensure global availability
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

def load_data(file_path):
    """
    Loads data from a JSONL file.
    Expects fields: title, description, input_description, output_description, problem_class, problem_score
    """
    try:
        df = pd.read_json(file_path, lines=True)
        return df
    except ValueError:
        # Fallback if lines=True fails or format is different
        df = pd.read_json(file_path)
        return df

def combine_text_fields(df):
    """
    Combines title, description, input, output into a single 'text' column.
    """
    # Fill NAs with empty strings
    text_cols = ['title', 'description', 'input_description', 'output_description']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Create combined text
    df['combined_text'] = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['input_description'] + " " + 
        df['output_description']
    )
    
    return df

def add_features(df):
    """
    Extracts explicit features from the text:
    - text_len: Number of characters
    - word_count: Number of words
    - math_symbol_count: Count of '$' (proxy for Latex math)
    """
    # math symbols (LaTeX often uses $...$)
    df['math_symbol_count'] = df['combined_text'].apply(lambda x: x.count('$'))
    
    # Text length
    df['text_len'] = df['combined_text'].apply(len)
    
    # Word count (rough)
    df['word_count'] = df['combined_text'].apply(lambda x: len(str(x).split()))
    
    return df

if __name__ == "__main__":
    # Test run
    sample_path = "e:/acm/problems_data.jsonl"
    try:
        df = load_data(sample_path)
        print(f"Loaded {len(df)} rows.")
        df = combine_text_fields(df)
        print("Combined text fields.")
        
        df = add_features(df)
        print("Features extracted.")
        
        df['cleaned_text'] = df['combined_text'].apply(clean_text)
        print("Cleaned text. Sample:")
        print(df[['cleaned_text', 'math_symbol_count', 'text_len']].head())
    except Exception as e:
        print(f"Error: {e}")
