import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Training Data
training_sentences = [
    "motivate me", "give me inspiration", "boost my confidence",
    "love quote", "romantic line", "something about love",
    "life quote", "something about life", "life advice"
]

training_labels = [
    "motivational", "motivational", "motivational",
    "love", "love", "love",
    "life", "life", "life"
]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, training_labels, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save Model
pickle.dump(model, open("../models/model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

print("Model saved successfully.")