# AutoJudge: Predicting Programming Problem Difficulty

**AutoJudge** is a machine learning-powered system designed to predict the difficulty of competitive programming problems. By analyzing the problem's textual description, input/output specifications, and other features, it assigns a difficulty class (**Easy**, **Medium**, **Hard**) and a numerical complexity score.

🔗 **[Live Demo](https://autojudge-knhcflhftvbcjlt3w8z7ta.streamlit.app)** - Try it out now!

---

## 📋 Project Overview

Competitive programming platforms host thousands of problems, but classifying them by difficulty is often inconsistent. **AutoJudge** automates this process using Natural Language Processing (NLP) and Machine Learning. The system takes a problem statement as input and predicts:
1.  **Difficulty Class**: Easy, Medium, or Hard.
2.  **Complexity Score**: A continuous score (1.0 - 3.0) representing the granularity of difficulty.

---

## 📊 Dataset Used

I created a comprehensive dataset by merging problems from three major competitive programming platforms:
*   **[LeetCode](https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset)**
*   **Codeforces** *(given in problem statement)*
*   **[Kattis](https://www.kaggle.com/code/mpwolke/coding-questions-solutions)**

The merged dataset ensures a diverse range of problem types, descriptions, and difficulty standards, making the model more robust.
*   **Total Samples**: ~Thousands of problems.
*   **Features Used**: Problem Title, Description, Input/Output details.

---

## 🧠 Approach and Models Used

### 1. Data Preprocessing
*   **Text Cleaning**: Removal of HTML tags, non-alphanumeric characters (preserving math symbols like `$`), and stopword removal.
*   **Feature Engineering**:
    *   **TF-IDF Vectors**: To capture key terms associated with difficulty.
    *   **Math Symbol Count**: Number of `$` signs, as harder problems often involve more mathematical notation.
    *   **Word Count**: Length of the problem description.

### 2. Machine Learning Models
I experimented with Logistic Regression, SVM, and Gradient Boosting, but the best performance was achieved with:

*   **Classification (Easy/Medium/Hard)**: **Random Forest Classifier**
*   **Regression (Complexity Score)**: **Random Forest Regressor**

---

## 📈 Evaluation Metrics

| Metric | Model | Score |
| :--- | :--- | :--- |
| **Accuracy** | Random Forest Classifier | **61.88%** |
| **MAE (Mean Absolute Error)** | Random Forest Regressor | **0.9409** |

*Note: Classification accuracy is ~62%, which is significant given the subjective nature of difficulty ratings across different platforms.*

---

## 💻 Steps to run the project locally

Follow these steps to set up AutoJudge on your local machine.

### Prerequisites
*   Python 3.8 or higher
*   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/chandrakant1212/Autojudge
    cd Autojudge
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Web App**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

---

## 🌐 Explanation of the Web Interface

The web interface is built using **Streamlit** for a seamless user experience.

1.  **Input Section**: Users enter the Problem Title, Description, Input metrics, and Output metrics.
2.  **Analyze Button**: Triggers the preprocessing pipeline and model inference.
3.  **Results Dashboard**:
    *   **Difficulty Prediction**: Displays Easy, Medium, or Hard.
    *   **Complexity Score**: Shows the precise regression score (e.g., 1.45).
    *   **Key Metrics**: Displays word count and other extracted features.

---

## 🎥 Demo Video

[**Watch the 3-Minute Demo Video Here**](https://drive.google.com/file/d/1plnoH4sEKQZSuOk8yDTiS6EaLaKFYWJo/view?usp=sharing)

**Video Highlights:**
*   Project explanation and motivation.
*   Walkthrough of the dataset and ML pipeline.
*   Live demonstration of the Web UI predicting problem difficulty.

---

## 👤 Contributors & Details

**Project by:** Chandrakant 
*   **Enrollment no.**: 23112027
*   **Contact**: chandrakant@ch.iitr.ac.in, [**LinkedIn**](https://www.linkedin.com/in/chandrakant-singariya-8b7a852a9)

*Built for the ACM Project.*
