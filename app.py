from flask import Flask, render_template, request
import joblib
import re

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model/sentiment_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review = clean_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    return render_template("index.html", prediction=prediction, review=review)


if __name__ == "__main__":
    app.run(debug=True)
    