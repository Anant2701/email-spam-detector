from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model/spam_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    proba = model.predict_proba([message])[0]
    confidence = round(proba[1] * 100, 2)  # probability of being spam
    result = "SPAM" if proba[1] > 0.5 else "NOT SPAM"
    return render_template('index.html', message=message, result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
