import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text


def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]

    # Lemmatize the tokens
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']


def visualize_sentiment(sentiment_scores):
    plt.figure(figsize=(8, 6))
    sns.histplot(sentiment_scores, kde=True)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution')
    st.pyplot()


st.title('Web Scraping and Sentiment Analysis')

url = st.text_input('Enter the URL:', '')

if st.button('Scrape and Analyze'):
    try:
        text = scrape_website(url)
        preprocessed_text = preprocess_text(text)
        sentiment_score = perform_sentiment_analysis(preprocessed_text)

        st.subheader('Scraped Text:')
        st.write(text)

        st.subheader('Sentiment Score:')
        st.write(sentiment_score)

        st.subheader('Sentiment Visualization:')
        visualize_sentiment([sentiment_score])

    except Exception as e:
        st.error(f'Error: {e}')
