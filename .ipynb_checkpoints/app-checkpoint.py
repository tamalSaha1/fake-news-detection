import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Website title
st.title("Fake News Detection System")

st.write("Enter a news article below to check if it is Real or Fake.")

# User input
news_text = st.text_area("Enter News Text")

# Prediction button
if st.button("Check News"):

    news_vector = vectorizer.transform([news_text])

    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.success("✅ This looks like Real News")
    else:
        st.error("❌ This looks like Fake News")