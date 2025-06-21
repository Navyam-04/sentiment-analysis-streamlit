import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_text

# Load saved model and vectorizer
model = joblib.load('../models/sentiment_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# Load dataset (same reduced one you trained on)
df = pd.read_csv('data/imdb-dataset.csv').sample(n=5000, random_state=42)


# Preprocessing
df['review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorize
X = vectorizer.transform(df['review']).toarray()
y = df['sentiment'].values

# Predict and calculate accuracy
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
