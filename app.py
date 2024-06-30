import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


def transform_text(text):
    ps = PorterStemmer()
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    stopwords_list = stopwords.words('english')
    words = [word for word in words if word not in stopwords_list and word not in string.punctuation]
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/ SMS Spam Classifier")

input_sns = st.text_area("Enter Email/SMS")

if st.button("Predict"):   
    
#1.preprocessing
    transformed_sms=transform_text(input_sns)

    vector_input = tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]
    if result == 1:
       st.header("Spam")
    else:
       st.header("Not Spam") 


