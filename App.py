"""
FUDMA Student Sentiment Analysis API
------------------------------------
Flask service used by the FUDMA Student Sentiment Analysis System.

POST /analyze   {"text": "..."}  ->  {"sentiment": "POSITIVE", "score": 0.85, "confidence": 85, "engine": "vader"}
GET  /health    ->  {"status": "ok"}

Engine: VADER (primary). TextBlob is used as a backup when VADER fails
or finds no sentiment-bearing words in the text.
"""

import os
from flask import Flask, jsonify, request
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

API_KEY = os.environ.get("SENTIMENT_API_KEY")  # optional shared secret
MAX_CHARS = 5000

# Standard VADER thresholds
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def classify(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "POSITIVE"
    if score <= NEG_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def confidence_for(sentiment: str, score: float, neutral_share: float) -> float:
    """Confidence percentage (0-100)."""
    if sentiment == "NEUTRAL":
        return round(max(0.0, min(1.0, neutral_share)) * 100, 2)
    return round(min(1.0, abs(score)) * 100, 2)


def analyze_with_vader(text: str):
    scores = analyzer.polarity_scores(text)
    if scores["pos"] == 0 and scores["neg"] == 0:
        return None
    compound = float(scores["compound"])
    sentiment = classify(compound)
    return {
        "sentiment": sentiment,
        "score": round(compound, 4),
        "confidence": confidence_for(sentiment, compound, scores["neu"]),
        "engine": "vader",
    }


def analyze_with_textblob(text: str):
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    sentiment = classify(polarity)
    return {
        "sentiment": sentiment,
        "score": round(polarity, 4),
        "confidence": confidence_for(sentiment, polarity, 1 - subjectivity),
        "engine": "textblob",
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required"}), 400
    text = text.strip()[:MAX_CHARS]

    try:
        result = analyze_with_vader(text)
    except Exception:
        result = None

    if result is None:
        result = analyze_with_textblob(text)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

