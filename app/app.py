import streamlit as st
import pickle
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# load the models and vectorizer
logistic_model = pickle.load(open('../models/logistic_model.pkl', 'rb'))
