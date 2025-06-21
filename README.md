# 🎯 Sentiment Analysis Web App

✅ **End-to-End Machine Learning Project: Data Collection ➔ Preprocessing ➔ Modeling ➔ Evaluation ➔ Deployment**  
✅ **Built using Python, NLP, Machine Learning, Streamlit**  
✅ **Fully deployed and accessible via web**

---

## 📖 Project Description

This is a complete **Sentiment Analysis Web Application** built entirely from scratch. The project demonstrates the full machine learning pipeline starting from data collection, preprocessing, feature engineering, model training, evaluation, and final deployment using Streamlit.

The application takes a movie review text as input and predicts whether the sentiment is **Positive** or **Negative**.

---

## 🔥 Key Technologies Used

- **Python 3.10**
- **Natural Language Processing (NLP)**
- **Machine Learning (Logistic Regression)**
- **Deep Learning (Optional Extension - LSTM planned)**
- **NLTK (Stopwords Removal, Text Cleaning)**
- **TF-IDF Vectorization**
- **scikit-learn (Modeling, Evaluation)**
- **Joblib (Model Serialization)**
- **Streamlit (Web App Development & Deployment)**

---

## 🗂 Project Workflow (Step-by-Step)

### 1️⃣ Dataset Selection

- **Dataset Used**: IMDB Movie Reviews Dataset
- Originally contains 50,000 reviews.
- We used a reduced sample for quick development: ~5000 reviews.

---

### 2️⃣ Data Preprocessing

- Lowercasing
- Removing special characters & punctuation
- Removing stopwords (using NLTK)
- Tokenization
- Cleaned data ready for feature extraction.

---

### 3️⃣ Feature Engineering

- **TF-IDF Vectorization**: Convert textual data into numerical feature vectors.
- Limited to top 5000 most frequent words.

---

### 4️⃣ Model Building

- **Algorithm Used**: Logistic Regression (for binary classification)
- **Train-Test Split**: 80% training, 20% testing
- **Model Accuracy Achieved**: 
    - On reduced dataset: **85%**
    - On full dataset (when tested earlier): ~**92%**

---

### 5️⃣ Model Evaluation

- Accuracy Score
- Simple and highly interpretable model
- Saved model using Joblib for later deployment

---

### 6️⃣ Deployment

- **Streamlit Web App** created
- Fully interactive user interface:
    - Input movie review
    - Predict sentiment instantly
    - Display appropriate success / error message
- Added professional UI:
    - Gradient banners
    - Colorful buttons
    - Balloons 🎈 for positive predictions
    - Images to enhance UI

---

## 💻 Folder Structure

```bash
sentiment-analysis-streamlit/
│
├── data/
│   └── imdb-dataset.csv
│
├── models/
│   └── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── scripts/
│   └── train_model.py
│   └── evaluate_model.py
│
├── app/
│   └── app.py
│   └── images/
│        └── banner.jpg
│
├── requirements.txt
├── README.md
└── .gitignore
## 🚀 Running Locally

## 🚀 Running Locally

1️⃣ Clone repository:

```bash
git clone https://github.com/Navyam-04/sentiment-analysis-streamlit.git
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run the app:

```bash
streamlit run app/app.py
```

---

## 🚀 Deployment

- Deployed using **Streamlit Cloud** (Free hosting)
- Live web application publicly accessible.

---

## 🔥 Skills Demonstrated

- Real-world machine learning project structure
- End-to-end ML pipeline building
- Data Cleaning & Preprocessing
- Feature Engineering (TF-IDF)
- Model Building & Evaluation
- Model Deployment
- Streamlit Web App Development
- GitHub version control
- Professional Coding Practices

---

## 📅 Timeline

| Task                | Status     | Notes |
|---------------------|------------|-------|
| Dataset Collection  | ✅ Completed | IMDB |
| Data Preprocessing  | ✅ Completed | NLTK |
| Feature Engineering | ✅ Completed | TF-IDF |
| Model Training      | ✅ Completed | Logistic Regression |
| Model Evaluation    | ✅ Completed | 91% accuracy |
| Streamlit App       | ✅ Completed | Beautiful UI |
| Deployment          | ✅ Completed | Streamlit Cloud |

---

## 🙋‍♀️ Author

**Mangali Navya**  
📧 Email: middenavya51@gmail.com  
🔗 LinkedIn: [Navya Mangali](https://www.linkedin.com/in/navya-mangali/)  
🐙 GitHub: [Navyam-04](https://github.com/Navyam-04/sentiment-analysis-streamlit)
