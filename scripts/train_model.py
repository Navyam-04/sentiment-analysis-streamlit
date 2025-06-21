import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from preprocessing import preprocess_text

print("✅ Step 1: Reading dataset...")
df = pd.read_csv('data/imdb-dataset.csv')
df = df.sample(n=5000, random_state=42)


print("✅ Step 2: Preprocessing text...")
df['review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("✅ Step 3: Vectorizing text...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['review']).toarray()
y = df['sentiment'].values

print("✅ Step 4: Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Step 5: Training model...")
model = LogisticRegression(max_iter=200)  # increase max_iter to ensure convergence
model.fit(X_train, y_train)

print("✅ Step 6: Evaluating model...")
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

print("✅ Step 7: Saving model and vectorizer...")

# Get absolute path to project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Save model and vectorizer
joblib.dump(model, os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))



print("🎯 All steps completed successfully! Model is ready for deployment.")
