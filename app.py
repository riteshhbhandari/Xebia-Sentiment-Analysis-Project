import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from joblib import load
import spacy
import time
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_lg")

__model = None

# for preprocessing
def preprocess(text):
    doc = nlp(text)

    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)

# vectorise
def vectorize(text):
    return nlp(text).vector.reshape(1, -1)

# for finding the bias
def get_bias(text):
    clean_text = preprocess(text)
    embeddings = vectorize(clean_text)
    
    prediction = __model.predict(embeddings)[0]

    if prediction == -1:
        return "Negative"
    elif prediction == 1:
        return "Positive"
    else:
        return "Neutral"

# for loading the model
def load_saved_model():
    global __model
    
    with open("model.joblib", "rb") as f:
        __model = load(f)

# Main function to run the Streamlit app
def main():

    load_saved_model()

    st.set_page_config(page_title="Sentiment Analysis App", page_icon=":smiley:", layout="centered")

    st.title('Sentiment Analysis :speech_balloon:')

    # Predict Page
    st.header('Predict Sentiment :mag:')
    text = st.text_area('Enter your comment here:', height=200, placeholder="Type your comment here...")
    if st.button('Predict'):
        if text:
            prediction = get_bias(text)
            st.write("## Results")
            if prediction == "Positive":
                st.success(f'**Predicted Sentiment: {prediction}** :smile:')
            elif prediction == "Negative":
                st.error(f'**Predicted Sentiment: {prediction}** :angry:')
            else:
                st.warning(f'**Predicted Sentiment: {prediction}** :neutral_face:')
            
            st.balloons()
        else:
            st.error('Please enter some text to analyze.')
if __name__ == '__main__':
    main()
