import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
model = pickle.load(open(r"nb.pkl",'rb'))

# Load the CountVectorizer used for training
##with  as f:
bow = pickle.load(open(r"bow.pkl",'rb'))

st.title("Email Spam/Ham Classifier")

# Input email text
Email = st.text_input("Paste the email here:")

# Check if the email input is not empty
if Email:

    data = bow.transform([Email]).toarray()


    spam_ham = model.predict(data)[0]


    if st.button('Submit'):
        st.write("The email is:",spam_ham)
