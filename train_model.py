import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load your dataset
data = pd.read_csv("spam.csv", encoding='latin-1')  # adjust name if needed
data = data[['v1', 'v2']]  # Keep only relevant columns
data.columns = ['label', 'text']  # Rename columns

# 2. Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# 4. Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 7. Save model and vectorizer
with open("spam_classifier_model.pkl", "wb") as m:
    pickle
import pickle

# Save the trained model
with open("spam_classifier_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
