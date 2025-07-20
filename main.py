import streamlit as st
import json
import os
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# CSS Styling
st.markdown("""
    <style>
    .navbar {
        background-color: #111;
        padding: 1rem;
        color: white;
        font-size: 24px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    <div class="navbar">📧 Spam Email Classifier</div>
""", unsafe_allow_html=True)

# Paths
USER_DB = "users.json"
MODEL_PATH = "spam_classifier_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
HISTORY_PATH = "history.json"

# Load Model & Vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Utility Functions
def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

def register(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = password
    save_users(users)
    return True, "Registration successful"

def login(username, password):
    users = load_users()
    if username in users and users[username] == password:
        return True, "Login successful"
    return False, "Invalid credentials"

def classify_email(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

def log_classification(username, text, result):
    history = load_history()
    user_logs = history.get(username, [])
    user_logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "email": text,
        "result": result
    })
    history[username] = user_logs
    save_history(history)

def show_dashboard(username):
    history = load_history()
    user_logs = history.get(username, [])
    total = len(user_logs)
    spam_count = sum(1 for log in user_logs if log['result'] == "Spam")
    not_spam_count = total - spam_count

    st.title("📊 Dashboard")
    st.markdown(f"**Total Emails Classified:** {total}")
    st.markdown(f"✅ Not Spam: {not_spam_count}")
    st.markdown(f"🚫 Spam: {spam_count}")

    if user_logs:
        st.subheader("🕓 Classification History")
        st.dataframe(user_logs[::-1])  # Show latest first
    else:
        st.info("No history yet. Start classifying some emails!")

# Main App
def main():
    st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    menu = ["Login", "Sign Up"]
    if st.session_state.logged_in:
        menu = ["Dashboard", "Spam Classifier", "Logout"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, msg = login(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(msg)
            else:
                st.error(msg)

    elif choice == "Sign Up":
        st.title("✍️ Register")
        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        if st.button("Register"):
            success, msg = register(username, password)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    elif choice == "Spam Classifier" and st.session_state.logged_in:
        st.title("📨 Spam Classifier")
        st.markdown(f"Welcome, **{st.session_state.username}**")
        email_text = st.text_area("Enter Email Text")
        if st.button("Classify"):
            if email_text.strip():
                result = classify_email(email_text)
                st.subheader(f"📌 Prediction: {result}")
                log_classification(st.session_state.username, email_text, result)
            else:
                st.warning("Please enter some text.")

    elif choice == "Dashboard" and st.session_state.logged_in:
        show_dashboard(st.session_state.username)

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully.")

if __name__ == "__main__":
    main()
