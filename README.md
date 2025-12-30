# AutoJudge: Predicting Programming Problem Difficulty

**AutoJudge** is a machine learning-powered system designed to predict the difficulty of competitive programming problems. By analyzing the problem's textual description, input/output specifications, and other features, it assigns a difficulty class (**Easy**, **Medium**, **Hard**) and a numerical complexity score.

🔗 **[Live Demo](https://autojudge-knhcflhftvbcjlt3w8z7ta.streamlit.app)** - Try it out now!

## 🚀 Features

- **Difficulty Classification**: Predicts whether a problem is `Easy`, `Medium`, or `Hard` using a Random Forest Classifier.
- **Complexity Scoring**: Estimates a continuous difficulty score (1-3) using a Random Forest Regressor.
- **Smart Text Processing**:
  - Cleans and standardizes problem descriptions (removes HTML, stopwords).
  - Extracts key features like Word Count and Math Symbol usage.
  - Utilizes TF-IDF vectorization for deep text analysis.
- **Interactive Web Interface**: Built with [Streamlit](https://streamlit.io/) for real-time analysis.

## � Model Performance

We evaluated multiple models (Logistic Regression, SVM, Random Forest, Gradient Boosting) to find the best performers.

### 🏆 Best Classification Model
- **Model**: **Random Forest Classifier**
- **Accuracy**: **61.88%**
- *Note: Classification is challenging due to the subjective nature of difficulty labels across different platforms.*

### 📉 Best Regression Model
- **Model**: **Random Forest Regressor**
- **Mean Absolute Error (MAE)**: **0.9409**
- *This indicates the predicted score is typically within ~0.94 of the actual difficulty score (on a scale of 1-3).*

## �🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/chandrakant1212/Autojudge>
   cd Autojudge
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup**:
   If you plan to retrain the models, ensure the dataset files (`leetcode_dataset.csv`, `kattis_dataset.csv`, `problems_data.jsonl`) are present in the root directory.

## 🖥️ Usage

### Run the Web App (Local)
To launch the interactive difficulty predictor locally:
```bash
streamlit run app.py
```
Open your browser and navigate to the provided local URL (usually `http://localhost:8501`).

### Training Models
The model training logic is encapsulated in Jupyter Notebooks.
- **Exploration & Training**: `notebooks/AutoJudge_Exploration.ipynb`
  - Handles data loading, cleaning, feature engineering, and model training.
  - Saves the trained models: `model_class.pkl` and `model_score.pkl`.

## 📁 Project Structure

- `app.py`: Main Streamlit application entry point.
- `src/`: Source code for helper functions.
  - `preprocessing.py`: Text cleaning and feature engineering logic.
- `notebooks/`: Jupyter notebooks for research and training.
- `requirements.txt`: List of Python dependencies.
- `*.pkl`: Serialized trained models (Classifier and Regressor).

---
*Built for the ACM Project.*
