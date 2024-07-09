import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the trained model and vectorizer
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_category(emails):
    emails_count = vectorizer.transform(emails)
    predictions = classifier.predict(emails_count)
    return predictions

def main():
    st.title("Email Spam Detection")

    email_input = st.text_area("Enter the email content:")

    if st.button("Predict"):
        result = predict_category([email_input])
        if result[0] == 0:
            st.write("This email is not spam (Ham).")
        else:
            st.write("This email is spam.")

if __name__ == "__main__":
    main()
