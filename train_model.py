import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Read the dataset
df = pd.read_csv("dataset/twitter_training.csv", header=None)

# Rename columns
df.columns = ["ID", "Entity", "Sentiment", "Review"]

# Keep only Review and Sentiment
df = df[["Review", "Sentiment"]]

# Remove empty reviews
df = df.dropna()

# Remove Irrelevant sentiment
df = df[df["Sentiment"] != "Irrelevant"]

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

df["Review"] = df["Review"].apply(clean_text)

X = df["Review"]
y = df["Sentiment"]
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)
print("Input shape:", X.shape)
print("Output shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model= LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Trained Successfully")
y_pred = model.predict(X_test)
accuracy= accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Test with a new review
new_review = ["I absolutely love this game"]

# Clean the review
new_review = [clean_text(new_review[0])]

# Convert to TF-IDF
new_review_vector = vectorizer.transform(new_review)

# Predict
prediction = model.predict(new_review_vector)

print("\nTest Review:", new_review[0])
print("Predicted Sentiment:", prediction[0])

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")