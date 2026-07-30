# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

import re
from tensorflow.keras.preprocessing import sequence

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = text.split()

    # Encode
    encoded_review = [word_index.get(word, 2) for word in words]

    # Pad
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

##Prediction function
def predict_sentiment(review):
    preprocesses_input=preprocess_text(review)

    prediction=model.predict(preprocesses_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
    return sentiment, prediction[0][0]

##sreamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    ##Make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
    
