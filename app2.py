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
    # Set page title and favicon
    st.set_page_config(page_title="Email Spam Detection", page_icon="📧")

    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            color: #333;
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
            background-image: url('https://media.istockphoto.com/photos/scams-in-the-www-picture-id147292100?k=6&m=147292100&s=612x612&w=0&h=Frrp2BeIHeNRoObYSVtGJyyQJyhKZS8bUnseD8rPYmw=');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .st-bj {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("Email Spam Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app predicts whether an email is spam or not spam (ham). "
        "It uses a Naive Bayes classifier trained on email data."
    )

    # Main content
    st.subheader("Enter the email content:")
    email_input = st.text_area("")

    # Prediction button
    if st.button("Predict"):
        if email_input.strip() != "":
            result = predict_category([email_input])
            if result[0] == 0:
                st.success("This email is not spam (Ham).")
            else:
                st.error("This email is spam.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
