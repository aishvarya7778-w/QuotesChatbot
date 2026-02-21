import pandas as pd
import random
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load dataset
data = pd.read_csv("../data/quotes.csv")

# Training data
training_sentences = [
    "motivate me",
    "give me inspiration",
    "love quote",
    "romantic line",
    "life quote",
    "something about life"
]

training_labels = [
    "motivational",
    "motivational",
    "love",
    "love",
    "life",
    "life"
]

import pandas as pd
import random
import pickle

# Load data
data = pd.read_csv("../data/quotes.csv")

# Load model & vectorizer
model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

print("Quotes Chatbot Ready! Type 'exit' to stop.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    user_vec = vectorizer.transform([user_input])
    predicted_category = model.predict(user_vec)[0]
    
    filtered_quotes = data[data["category"] == predicted_category]
    print("Bot:", random.choice(filtered_quotes["quote"].values))