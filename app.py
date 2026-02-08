from flask import Flask, render_template, request
from sentiment import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        review = request.form["review"]
        result = predict_sentiment(review)
    return render_template("index.html", result=result)

if __name__ == "__main__":
   import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
