import streamlit as st
import pickle
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# load the models and vectorizer
logistic_model = pickle.load(open('../models/logistic_model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

bert_model = TFBertForSequenceClassification.from_pretrained('../model/bert_model')
bert_tokenizer = BertTokenizer.from_pretrained('../models/bert_tokenizer')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    model_choice = request.form['model_choice']

    if model_choice == 'logistic':
        news_tfidf = vectorizer.transformers([news_text])
        prediction = logistic_model.predict(news_tfidf)
        prediction_prob = logistic_model.predict_proba(news_tfidf)
        response = {'prediction': int(prediction[0], 'probability': prediction_prob[0].tolist())}

    elif model_choice == 'bert':
        inputs = bert_tokenizer(news_text, return_tensors='tf', truncation=True, padding= True)
        outputs = bert_model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis = -1).numpy()[0]
        prediction = int(tf.argmax(probabilities))
        response = {'prediction': prediction, 'probability': probabilities.tolist()}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)