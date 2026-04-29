import nltk
nltk.download('stopwords')

from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- SETUP ---------------- #

STOPWORDS = set(stopwords.words("english"))
app = Flask(__name__)

# Load models once (important)
predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    return "Server is running ✅"

# ---------------- PREDICT ---------------- #

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Request received")

        # CSV FILE
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            data = pd.read_csv(file)

            print("CSV received")

            predictions = bulk_prediction(data)

            return send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

        # TEXT INPUT
        elif request.is_json:
            data = request.get_json()
            text_input = data.get("text", "")

            print("Text received:", text_input)

            if text_input.strip() == "":
                return jsonify({"error": "Empty input"}), 400

            result = single_prediction(text_input)

            print("Prediction:", result)

            return jsonify({"prediction": result})

        else:
            return jsonify({"error": "Invalid input"}), 400

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- FUNCTIONS ---------------- #

def preprocess(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return " ".join(review)


def single_prediction(text_input):
    review = preprocess(text_input)

    X = cv.transform([review]).toarray()
    X = scaler.transform(X)

    pred = predictor.predict_proba(X).argmax(axis=1)[0]

    return "Positive" if pred == 1 else "Negative"


def bulk_prediction(data):
    corpus = [preprocess(str(text)) for text in data["Sentence"]]

    X = cv.transform(corpus).toarray()
    X = scaler.transform(X)

    preds = predictor.predict_proba(X).argmax(axis=1)
    preds = ["Positive" if x == 1 else "Negative" for x in preds]

    data["Predicted sentiment"] = preds

    output = BytesIO()
    data.to_csv(output, index=False)
    output.seek(0)

    return output

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)