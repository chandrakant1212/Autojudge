# AutoJudge: Predicting Programming Problem Difficulty

AutoJudge is a machine learning system that predicts the difficulty class (Easy, Medium, Hard) and a numerical difficulty score for programming problems based on their textual descriptions.

## Project Overview

The system aggregates data from **Codeforces**, **LeetCode**, and **Kattis** to train robust models.
It uses TF-IDF for text vectorization and Random Forest models for prediction.

- **Classification**: Predicts `Easy`, `Medium`, or `Hard`.
- **Regression**: Predicts a difficulty score (1-3).

## Setup Instructions

1. **Prerequisites**: Python 3.8+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Data**: Ensure datasets (`leetcode_dataset.csv`, `kattis_dataset.csv`, `problems_data.jsonl`) are in the root directory if you plan to retrain.

## Training
The training logic is contained in the Jupyter Notebook:
- **Notebook**: `notebooks/AutoJudge_Exploration.ipynb`
- Features: Data cleaning, TF-IDF extraction, Model Comparison (Logistic, SVM, Random Forest, Gradient Boosting).
- Output: Saves `model_class.pkl` and `model_score.pkl`.

## Web Interface
To launch the interactive difficulty predictor:
```bash
streamlit run app.py
```

## Project Structure
- `notebooks/`: Exploration and Training notebooks.
- `src/preprocessing.py`: Shared text cleaning logic.
- `app.py`: Streamlit web application.
- `requirements.txt`: Python dependencies.
