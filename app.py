import streamlit as st
import pandas as pd
import random
import pickle

# Load data
data = pd.read_csv("data/quotes.csv")

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("Quotes Recommendation Chatbot 💬")
st.write("Type your mood or request and get a quote!")

user_input = st.text_input("Enter your message:")

if st.button("Get Quote"):
    if user_input:
        user_vec = vectorizer.transform([user_input])
        predicted_category = model.predict(user_vec)[0]
        filtered_quotes = data[data["category"] == predicted_category]
        quote = random.choice(filtered_quotes["quote"].values)
        st.success(quote)
    else:
        st.warning("Please enter some text.")