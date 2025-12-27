import streamlit as st
import pandas as pd
import joblib
import sys
import os

# --- Configuration ---
st.set_page_config(page_title="AutoJudge", page_icon="⚖️", layout="wide")

# --- Path Setup ---
# Helper to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    import preprocessing
except ImportError as e:
    st.error(f"Error importing preprocessing module: {e}")
    st.stop()

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load the trained models directly from the root directory."""
    try:
        clf_path = os.path.join(current_dir, 'model_class.pkl')
        reg_path = os.path.join(current_dir, 'model_score.pkl')
        
        if not os.path.exists(clf_path) or not os.path.exists(reg_path):
             return None, None

        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        return clf, reg
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

# --- UI Layout ---
st.title("⚖️ AutoJudge: AI Difficulty Predictor")
st.markdown("""
Predict the difficulty (Easy/Medium/Hard) and a numerical score (1-3) for your coding problem.
*Powered by Scikit-Learn*
""")

col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("How it works", expanded=False):
        st.write("""
        1. Enter your problem details.
        2. The model standardizes the text (removes HTML, stopwords).
        3. Features like TF-IDF, word count, and math symbol usage are extracted.
        4. A Random Forest model predicts the difficulty.
        """)

# --- Input Form ---
with st.form("main_form"):
    title = st.text_input("Problem Title", placeholder="e.g. Longest Path in Matrix")
    description = st.text_area("Problem Statement", height=200, help="Clean text usually works best.")
    
    c1, c2 = st.columns(2)
    with c1:
        input_desc = st.text_area("Input Specification", height=100)
    with c2:
        output_desc = st.text_area("Output Specification", height=100)
    
    submitted = st.form_submit_button("Analyze Problem", type="primary")

if submitted:
    if not description:
        st.warning("Please provide at least a problem description.")
    else:
        # Load models
        clf_model, reg_model = load_models()
        
        if clf_model is None:
            st.error("Models typically 'model_class.pkl' and 'model_score.pkl' not found. Please run the training notebook first.")
        else:
            # Create Dataframe
            raw_data = {
                'title': [title],
                'description': [description],
                'input_description': [input_desc],
                'output_description': [output_desc]
            }
            df = pd.DataFrame(raw_data)
            
            with st.spinner("Analyzing..."):
                try:
                    # 1. Combine Text
                    df = preprocessing.combine_text_fields(df)
                    
                    # 2. Add Explicit Features (math symbols, len)
                    df = preprocessing.add_features(df)
                    
                    # 3. Clean Text
                    df['cleaned_text'] = df['combined_text'].apply(preprocessing.clean_text)
                    
                    # 4. Predict
                    pred_class = clf_model.predict(df)[0]
                    pred_score = reg_model.predict(df)[0]
                    
                    # --- Results ---
                    st.success("Analysis Complete")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Difficulty", pred_class, delta_color="off")
                    with m2:
                         st.metric("Complexity Score", f"{pred_score:.2f}")
                    with m3:
                        st.metric("Word Count", df['word_count'][0])

                    # Colored Badge Logic
                    color_map = {"Easy": "green", "Medium": "orange", "Hard": "red"}
                    color = color_map.get(pred_class, "blue")
                    st.caption(f"Rated as :{color}[{pred_class}]")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    # st.exception(e) # Uncomment for debug
