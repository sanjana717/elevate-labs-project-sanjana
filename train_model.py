import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load data
df = pd.read_csv("dataset.csv")

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["career"]


# Train model
model = MultinomialNB()
model.fit(X, y)

import os
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "model/career_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
