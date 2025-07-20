import streamlit as st
import pickle

def run_classifier():
    st.title("Spam Email/SMS Classifier")

    with open("spam_classifier_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

    user_input = st.text_area("Enter your message")

    if st.button("Classify"):
        transformed = vectorizer.transform([user_input])
        result = model.predict(transformed)

        if result[0] == 1:
            st.error("Spam ❌")
        else:
            st.success("Not Spam ✅")
