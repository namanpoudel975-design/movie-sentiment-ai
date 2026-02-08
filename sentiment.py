import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "review": [
        "Amazing movie I loved it",
        "Fantastic film with great acting",
        "Very good and enjoyable movie",
        "Excellent story and direction",
        "Best movie ever",

        "Worst movie ever",
        "Terrible film and bad acting",
        "Very boring and disappointing",
        "Waste of time",
        "Bad plot and horrible scenes",

        "The movie was not good",
        "I did not like the movie",
        "Not worth watching",
        "Not enjoyable at all",
        "The film was not interesting"
    ],
    "sentiment": [
        "positive","positive","positive","positive","positive",
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# VECTORIZE (THIS CREATES X_train_vec)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction function
def predict_sentiment(review):
    review = review.lower()

    # 1️⃣ Handle negation first
    negation_phrases = [
        "not good", "not great", "not amazing", "not nice",
        "not fantastic", "not excellent", "not well"
    ]

    for phrase in negation_phrases:
        if phrase in review:
            return "negative"

    # 2️⃣ Keyword lists
    positive_words = [
        "good", "great", "amazing", "awesome",
        "fantastic", "excellent", "nice",
        "love", "well", "enjoyed", "liked"
    ]

    negative_words = [
        "bad", "worst", "boring", "terrible",
        "awful", "hate", "poor", "disappointing"
    ]

    # 3️⃣ Check negative first
    for word in negative_words:
        if word in review:
            return "negative"

    # 4️⃣ Check positive
    for word in positive_words:
        if word in review:
            return "positive"

    # 5️⃣ If unclear → neutral (VERY IMPORTANT)
    return "neutral"



