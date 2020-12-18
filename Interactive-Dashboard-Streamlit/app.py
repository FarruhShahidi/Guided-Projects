import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


st.title("Sentiment analysis of tweets of US airlines")
st.sidebar.title("Sentiment analysis of tweets of US airlines")

st.markdown("This app is a streamlit dashboard to analyze the sentiment of tweets 🤖 ")


st.sidebar.markdown("This app is a streamlit dashboard to analyze the sentiment of tweets 🐧 ")

DATA_URL = ("/home/rhyme/Desktop/Project/Tweets.csv")
@st.cache(persist=True)
def load_data():
    df = pd.read_csv(DATA_URL)
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    return df

df = load_data()

st.write(df)

st.sidebar.subheader("Show random tweet")

random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(df.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key = '1')

sentiment_count = df['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == "Histogram":
        fig = px.bar(sentiment_count, x = 'Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.subheader("Location of the tweets")
hour = st.sidebar.slider("Hour of day", 0, 23)

df_modified = df[df['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Close", True, key='1'):
    st.markdown("### Tweet locations based on time")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(df_modified), hour, (hour + 1) % 24))
    st.map(df_modified)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(df_modified)
st.sidebar.subheader("Breakdown of ariline tweets by sentiment")
choice = st.sidebar.multiselect('Pick an airline', ("US Airways", "United", "American", "Southwest", "Delta", "Virgin America"), key='0')

if len(choice):
    df_choice = df[df.airline.isin(choice)]
    fig_choice = px.histogram(df_choice, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',
    facet_col='airline_sentiment', labels={'airline_sentiment': 'tweets'}, height=600, width=800)
    st.plotly_chart(fig_choice)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio("Choose a sentiment for word cloud", ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Close", True, key='3'):
    st.header("Word cloud for %s sentiment" % (word_sentiment))
    df_new = df[df['airline_sentiment']==word_sentiment]
    words = ' '.join(df_new['text'])
    processed_words = ' '.join([word for word in words.split() if
    'http' not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=400, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
