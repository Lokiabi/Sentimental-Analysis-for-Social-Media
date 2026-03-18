import streamlit as st
import pandas as pd
from twitter_fetch import fetch_tweets
from emotion_model import predict_emotion

st.set_page_config(page_title="Deep Learning Twitter Sentiment", layout="wide")
st.title("🤖 Twitter Emotion Classifier using BERT")

query = st.text_input("Enter a topic or hashtag", "#climatechange")

if st.button("Classify"):
    with st.spinner("Fetching and analyzing tweets..."):
        df = fetch_tweets(query)
        if not df.empty:
            df[['Emotion', 'Probabilities']] = df['Tweet'].apply(lambda x: pd.Series(predict_emotion(x)))
            st.subheader("Sample Tweets")
            st.dataframe(df[['Timestamp', 'Tweet', 'Emotion']].head(10))
            
            st.subheader("Emotion Distribution")
            st.bar_chart(df['Emotion'].value_counts())
        else:
            st.warning("No tweets found.")