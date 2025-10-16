# 🤖 AI Task Management System

An **AI-powered task classification system** that automatically categorizes task descriptions into predefined categories using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.

---

## 🚀 Project Overview

This project helps automate task management by classifying textual task descriptions into categories such as *Development*, *Testing*, *Design*, etc.  
It uses advanced NLP preprocessing, multiple feature extraction techniques, and machine learning models like **Naive Bayes** and **SVM**.

---

## 🧱 Project Structure

AI_Task_Management_System/
│
├── Data/
│ └── training_tasks.csv # Original dataset
│
├── output/
│ └── cleaned_training_tasks.csv # Cleaned dataset after preprocessing
│
├── models/
│ ├── naive_bayes_tfidf.pkl
│ ├── svm_tfidf.pkl
│ ├── svm_word2vec.pkl
│ ├── svm_bert.pkl
│ └── tfidf_vectorizer.pkl
│
├── src/
│ └── Complete system.ipynb # Main notebook (EDA, preprocessing, feature extraction, training)
│
├── README.md # Project documentation
└── .gitignore # Files and folders ignored by Git

yaml
Copy code

---

## ⚙️ Features

✅ **Exploratory Data Analysis (EDA)** – Data overview, category distribution, and null value checks.  
✅ **Data Cleaning & Preprocessing** – Lowercasing, punctuation removal, stopword removal, and stemming.  
✅ **NLP Feature Extraction**
- TF-IDF Vectorization  
- Word2Vec Embeddings  
- BERT Sentence Embeddings  
✅ **Model Training**
- Naive Bayes (TF-IDF)
- SVM (TF-IDF / Word2Vec / BERT)
✅ **Model Evaluation**
- Accuracy, Precision, Recall, F1-score  
✅ **Model Saving** using `joblib`  
✅ **Version Control** integrated with Git and GitHub  

---

## 🧠 Algorithms and Models Used

| Feature Extraction | Model Used         |
|--------------------|--------------------|
| TF-IDF             | Naive Bayes, SVM   |
| Word2Vec           | SVM                |
| BERT (MiniLM-L6-v2)| SVM                |

---

## 📊 Sample Results

| Model                | Accuracy | F1 Score |
|----------------------|-----------|-----------|
| Naive Bayes (TF-IDF) | 1.0000    | 1.0000    |
| SVM (TF-IDF)         | 1.0000    | 1.0000    |
| SVM (Word2Vec)       | 0.3975    | 0.2576    |
| SVM (BERT)           | 1.0000    | 1.0000    |

*(Note: Results may vary depending on dataset split and preprocessing.)*

---

## 🧩 Dependencies

Install all required packages before running the project:

```bash
pip install pandas numpy scikit-learn nltk gensim sentence-transformers matplotlib seaborn joblib
Alternatively, if you have a requirements.txt file, install all at once:

bash
Copy code
pip install -r requirements.txt
🪄 How to Run
Clone the repository

bash
Copy code
git clone https://github.com/EngineerSaurav1234/AI_Task_Management_System.git
cd AI_Task_Management_System
Run the notebook

bash
Copy code
jupyter notebook src/Complete system.ipynb
Execute cells sequentially

Load and clean data

Perform NLP preprocessing

Extract features (TF-IDF, Word2Vec, BERT)

Train and evaluate models

Save trained models

🧮 Evaluation Metrics
The following metrics are used to evaluate model performance:

Accuracy

Precision

Recall

F1 Score

Example (for SVM - TF-IDF):

markdown
Copy code
              precision    recall  f1-score   support

Development       1.00      1.00      1.00        20
Testing           1.00      1.00      1.00        15
Design            1.00      1.00      1.00        10

accuracy                              1.00        45
macro avg         1.00      1.00      1.00        45
weighted avg      1.00      1.00      1.00        45
💾 Output Files
Folder	Description
/output	Contains cleaned dataset
/models	Stores trained ML models (.pkl files)
/Data	Original input dataset
/src	Jupyter notebook with complete workflow

🧭 Version Control (Git & GitHub)
To manage project versions:

bash
Copy code
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/EngineerSaurav1234/AI_Task_Management_System.git
git push -u origin main
🧩 Future Enhancements
Add Flask / Streamlit web interface for predictions

Integrate real-time task input

Expand dataset with more labeled tasks

Deploy model using Docker / FastAPI

👨‍💻 Author
Saurav
💼 AI & Machine Learning Enthusiast
🌐 GitHub Profile
📧 https://github.com/EngineerSaurav1234

