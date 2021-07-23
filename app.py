import streamlit as st
import pickle
import re
import nltk
# nltk.download('punkt')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stem = PorterStemmer()
stop_words = stopwords.words('english')

tfidf = pickle.load(open('../../Documents/GitHub/Spam-Ham_CLassifier/vectorizer.pkl', 'rb'))
model = pickle.load(open('../../Documents/GitHub/Spam-Ham_CLassifier/model.pkl', 'rb'))

st.title('Spam SMS Classifier')
input_text = st.text_area('Enter the message')
button = st.button('Go')
#1. Preprocess
#2.vectorize
#3. predict
#4. display
#4. display


def text_transform(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = nltk.word_tokenize(text)
    text = ' '.join([stem.stem(word) for word in text if word not in set(stop_words)])
    return text

tranformed_text = text_transform(input_text)

vals = tfidf.transform([tranformed_text]).toarray()
res = model.predict(vals)[0]

if button:
    if res == 1:
        st.header('Spam')
    else:
        st.header('Ham')
