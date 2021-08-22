import streamlit as st
import json, requests
import nltk
import re
from nltk.corpus import sentiwordnet as swn
import timeit
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import pandas as pd
import numpy as np
import urllib.request, urllib.parse, urllib.error
import seaborn as sns
import matplotlib.pyplot as plt
import time
import requests
from datetime import datetime
from pathlib import Path
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import praw

CLIENT_ID = "_X6SXrJsp13mexOwy5ZjBQ"
CLIENT_SECRET = "hJgrooGc5eljGaqRp5cV4loZqVCZMg"
USERNAME = "crytosubscraper"
PASSWORD = "NYSE: IVHtothemoon"

# Set up reddit client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent="web:cryptosubs:1",
    ratelimit_seconds=0,
    username=USERNAME,
    password=PASSWORD,
)

'''
# CryptoSubs 💹

Perform sentiment analysis and chart accuracy of subreddits.

'''

# Dictionary with all implemented crypto subreddits, attached to their respective coin symbols
subdict = {'bitcoin': 'BTC', 'btc': 'BTC', 'ethereum': 'ETH', 'litecoin':'LTC','DogeCoin': 'DOGE'}

# Allows you to select a subreddit to analyze
sub = st.selectbox(
    'Choose a subreddit to analyse',
     ['bitcoin', 'btc', 'ethereum', 'litecoin', 'DogeCoin'])

# Allows you to select a subreddit to compare the first one to (optional)
compare = st.selectbox(
    'Choose a subreddit to compare it with (optional)',
     ['none', 'bitcoin', 'btc', 'ethereum', 'litecoin', 'DogeCoin'])


# preliminary sub information finding
def get_sub_name(sub):
    return f"r/{sub}"

sub_name = get_sub_name(sub)
f"## {sub_name}"

subreddit = reddit.subreddit(sub)

f"""
> *{subreddit.title}*


> {subreddit.public_description}
"""


def get_reddit_data(sub):
    '''
    Parameters:
    sub (str): Reddit subreddit to scrape top posts/comments from
    example: 'bitcoin'

    Returns:
    dfs (DataFrame): DataFrame containing post title's, date/time posted, and the number of upvotes

    '''

	#bar progression code
    loading_message = st.empty()
    loading_message.text(f"Scraping r/{sub}...")
    bar = st.progress(0)

    # create lists to hold the post title (text), time in utc format, and the number of upvotes
    text = []
    comment_text = []
    times = []
    up = []

    # the title, time, and upvotes can be found in "children", which can be found in "data"
    NUM_POSTS = 1000
    posts = reddit.subreddit(sub).top("year", limit=NUM_POSTS)
    for i, post in enumerate(posts):
	    # Scrape posts for title, date, and score
        bar.progress((i + 1) / NUM_POSTS)
        text.append(post.title)
        times.append(post.created_utc)
        up.append(post.score)

        # Scrape comments
        post.comment_sort = "top"
        comments = post.comments.list()[:3]
        comment_text.append("\n".join([c.body for c in comments]))

	#more bar progression code
    loading_message.text("All done!")
    bar.progress(1.0)

    text = np.array(text)
    times = np.array(times)
    up = np.array(up)

    dfs = pd.DataFrame({'Post Title' : text, 'Time' : times, 'Up Votes' : up, 'Comments': comment_text})

    return dfs

def get_reddit_dataset(sub):
    dt_string = datetime.now().strftime("%m-%Y")
    dataset_name = f"{sub}_subreddit_{dt_string}.csv"
    d_file = Path(f"datasets/{dataset_name}")
    if d_file.is_file():
        txt = pd.read_csv(d_file, sep='\t', encoding='utf-8', index_col=[0])
        return txt
    else:
        txt = get_reddit_data(sub)

        # Drop posts with the same post titles
        txt.drop_duplicates(subset='Post Title', keep='first', inplace=True)
        txt.to_csv(f"datasets/{dataset_name}", sep='\t', encoding='utf-8')
        return txt

def get_nltk():
    # Initialize NLTK's sentiment intensity analyzer
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    new_words = {
        'crushes': 10,
        'beats': 80,
        'misses': -50,
        'trouble': -100,
        'falls': -100,
        'hold': 90,
        'holding': 90,
        'moon': 100,
        'dip': -100,
        'hodl': 90,
        'hodling': 90,
        'up': 50,
        'down': -50,
        'overtaking': 100,
        'upgrade': 50,
        'ðŸš€':100,
        "risk": -100,
        "sell": -100,
        "sold": -100,
        "scam": -100,
        "panic": -100,
        "dip": -50,
        "buy": 100,
        "bought": 100,
    }
    sid.lexicon.update(new_words)
    return sid

def sentiment_analyze(txt):
    sid = get_nltk()

    # create lists to append our sentiment scores too
    sen = []
    neg = []
    neu = []
    pos = []
    com = []

    # Iterate through all the post titles in our DataFrame
    max_upvotes = max(txt['Up Votes'])

    for title, comment, upvotes in zip(txt['Post Title'], txt['Comments'], txt['Up Votes']):
        sentence = "\n".join([title, str(comment)])
        sen.append(sentence)
        # upvotes/max_upvotes weights each post based on upvote
		# Dividing by max_upvotes is necessary to keep the system equal for smaller subs vs. bigger subs
        neg.append(sid.polarity_scores(sentence)["neg"]* upvotes/max_upvotes)
        neu.append(sid.polarity_scores(sentence)["neu"]* upvotes/max_upvotes)
        pos.append(sid.polarity_scores(sentence)["pos"]* upvotes/max_upvotes)
        com.append(sid.polarity_scores(sentence)["compound"]* upvotes/max_upvotes)

    # convert all the lists to NumPy array'
    sen = np.array(sen)
    neg = np.array(neg)
    neu = np.array(neu)
    pos = np.array(pos)
    com = np.array(com)

    # create a DataFrame with all of our Sentiment Scores
    df = pd.DataFrame({'Post Title' : txt['Post Title'], 'Negative' : neg, 'Neutral' : neu, 'Positive' : pos, 'Compound' : com})
    df = df.set_index('Post Title')

    # Merge our original DataFrame (txt) with our DataFrame containing the sentiment of the post title's
    txt = txt.set_index('Post Title')
    txt = txt.join(df)

    txt['Date'] = pd.to_datetime(txt['Time'] , unit='s').dt.date
    txt = txt.sort_values('Date')
    positive = txt[['Time', 'Up Votes', 'Positive']]
    negative = txt[['Time', 'Up Votes', 'Negative']]

    # Remove any columns we don't need
    txt.drop(['Time', 'Up Votes', 'Negative', 'Neutral', 'Positive'], axis = 1, inplace = True)
    txt.reset_index(inplace = True)

    # This DataFrame will be used later on to take our analysis a step further
    txt = txt.drop('Post Title', axis=1)
    txt = txt[['Date', 'Compound']]
    txt = txt.groupby('Date').mean()

    # Finalize preprocessing for Negative and Positive DF's
    negative['Date'] = pd.to_datetime(negative['Time'] , unit = 's').dt.date
    positive['Date'] = pd.to_datetime(positive['Time'] , unit = 's').dt.date

    negative = negative.groupby('Date').mean()
    postive = positive.groupby('Date').mean()
    negative.reset_index(inplace = True)
    positive.reset_index(inplace = True)

    negative = negative[['Date', 'Negative']]
    positive = positive[['Date', 'Positive']]
    negative.set_index('Date', inplace = True)
    positive.set_index('Date', inplace = True)

    return (txt, negative, positive)


def set_reliability_graph(sub, txt, plot=True):
    # Gets the name of the sub and cryptocurrency
    symbol = subdict[sub]
    sub_name = get_sub_name(sub)

    # Obtains a JSON containing the necessary stock data for the coin chosen, and strips it to 
    # ['Time Series (Digital Currency Daily)'], which has all of the open/close data we need
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={subdict[sub]}&market=USD&apikey=3d8fc159-9660-484a-83eb-f352aaf6742d'
    r = requests.get(url)
    crypdata = r.json()

    df = pd.DataFrame.from_dict(crypdata['Time Series (Digital Currency Daily)'], orient='index')
    df['Date'] = pd.to_datetime(df.index).date

    f'**{sub_name}**'
    df

    # Plot coin close value in order to create a historical graph of the crypto
    fig = go.Figure(
          data = [
               go.Scatter(
                  x = df['Date'],
                  y = pd.to_numeric(df['4a. close (USD)']),
                  mode = 'lines',
                  name = 'Multiplied (True Accuracy)',
                  line = {'color': '#FFA500'}
              ),
          ]
    )
    fig.update_layout(
        title = f'{symbol} value graph',
        xaxis_title = 'Date',
        yaxis_title = 'Price',
        xaxis_rangeslider_visible = False
    )
    st.write(fig)

    # Join dataframes on date index
    df = df.merge(txt, how='inner', on='Date')

    x = []
    y = []

    # Overall accuracy score
    accuracy = 0
    max_market_change = float('-inf')
    sentiment_avg = 0

	
	#Goes through the entire dataframe, finding the gain/loss of the coin for that day and multiplying that with
	#the average sentiment score of that day
    for _, row in df.iterrows():
        market_change = float(row['4a. close (USD)']) - float(row['1a. open (USD)'])
        acc = row['Compound'] * market_change * 1000
        max_market_change = max(max_market_change, market_change)
        accuracy += acc
        y.append(acc)
        x.append(row['Date'])
        sentiment_avg += row['Compound'] * 1000

    y = [(c / max_market_change) for c in y]
    
	#plots previously set up reliability graph
    fig = go.Figure(
          data = [
               go.Scatter(
                  x = x,
                  y = y,
                  mode = 'lines',
                  name = 'Multiplied (True Accuracy)',
                  line = {'color': '#FFA500'}
              ),
          ]
    )
    fig.update_layout(
        title = f'Reliability graph for {symbol}',
        xaxis_title = 'Date',
        yaxis_title = 'Price * Sentiment',
        xaxis_rangeslider_visible = False
    )
    if plot:
        st.write(fig)

	# accuracy
    accuracy = ((accuracy)/max_market_change)/len(txt)
    f"The cumulative accuracy score for **{sub_name}** is {accuracy}."
    f"Cumulative accuracy is calculated by taking the average of the daily gain/loss of the previous year, and multiplying that with the sentiment score for that day."
    if accuracy > 1:
	    f"A cumulative accuracy of over 1 is considered extremely good."

	# sentiment
    sentiment_avg = sentiment_avg/len(txt)
    f"The cumulative sentiment score for **{sub_name}** is {sentiment_avg}."
    if sentiment_avg > 90:
	    f"**Warning:** The extremely high Sentiment score for **{sub_name}** indicates its accuracy score may be unreliable. Scores of over 90 indicate the community may be an echo chamber."
    elif sentiment_avg < 0:
	    f"**Warning:** The extremely low Sentiment score for **{sub_name}** indicates its accuracy score may be unreliable. Scores of under 0 indicate the community may be a parody or ironic."

    return (x, y)


def set_sentiment_graphs(sub, sentiment_txt, neg, pos):
    # plots a pure sentiment graph, with Negative and Positive added as well
    fig = go.Figure(
          data = [
               go.Scatter(
                  x = sentiment_txt.index,
                  y = sentiment_txt['Compound'],
                  mode = 'lines',
                  name = 'Compound',
                  line = {'color': '#FFA500'}
              ),
              go.Scatter(
                  x = neg.index,
                  y = neg['Negative'],
                  mode = 'lines',
                  name = 'Negative',
                  line = {'color': '#ff6961'}
              ),
              go.Scatter(
                  x = pos.index,
                  y = pos['Positive'],
                  mode = 'lines',
                  name = 'Positive',
                  line = {'color': '#77DD77'}
              )
          ]
    )


    fig.update_layout(
        title = f'Compound sentiment graph for {sub}',
        xaxis_title = 'Date',
        yaxis_title = f'Compound sentiment',
        xaxis_rangeslider_visible = False
    )
    st.write(fig)


def set_compare_graph(x, y, name, other_x, other_y, other_name):
    # plots a reliability graph against another crypto sub
    fig = go.Figure(
            data = [
                go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    name = f'Multiplied (True Accuracy) of {name}',
                    line = {'color': '#FFA500'}
                ),
				# sub being compared (2nd)
                go.Scatter(
                    x = other_x,
                    y = other_y,
                    mode = 'lines',
                    name = f'Multiplied (True Accuracy) of {other_name}',
                    line = {'color': '#0000FF'}
                )
            ]
    )
    fig.update_layout(
        title = f'Reliability graph for {name} vs {other_name}',
        xaxis_title = 'Date',
        yaxis_title = f'Price * Sentiment',
        xaxis_rangeslider_visible = False
    )
    st.write(fig)


# Scraping reddit for dataset
'### Reddit dataset'
txt = get_reddit_dataset(sub)


txt['Time'] = pd.to_datetime(txt['Time'], unit='s', origin='unix')
txt = txt.sort_values(by=['Up Votes'], ascending=False)

# Outputs the reddit dataset found
txt 

f"""
    These are the top posts from **{sub_name}** the past year.
    We looked at {len(txt)} posts.
    The oldest post was made {min(txt['Time'])} and the newest post was made {max(txt['Time'])}.
"""

'### Sentiment analysis dataset'

# Analayzes sentiment of the reddit dataset, stores in sentiment_txt
(sentiment_txt, neg, pos) = sentiment_analyze(txt)
sentiment_txt
f"""
    Using `nltk`, we can determine the overall compound sentiment for each date on the subreddit.
    Higher is better outlook, lower is worse outlook.
"""

# Primarily a debug tool, but can also be utilized by users to find out what specific dips/peaks could be related to
with st.expander("Does the sentiment analysis make sense?"):
    i = st.number_input("Check sentiment for dataset index", value=0, min_value=0)
    sid = get_nltk()

    f"""
    **Post Title**
    ```
    {txt.loc[i]['Post Title']}
    ```

    **Date**: {txt.loc[i]['Time']}

    **Post Title sentiment score:**
    """
    st.write(sid.polarity_scores(txt.loc[i]['Post Title']))

    comment_text = txt.loc[i]['Comments']
    comment_text = "".join([c for c in comment_text if c.isalnum() or c == " "])

    f"""
    **Comments**
    ```
    {comment_text}
    ```
    **Comment sentiment score:**
    """
    st.write(sid.polarity_scores(txt.loc[i]['Comments']))

'### Coin performance'


if compare != 'none':
    #no comparison done
    f"""
    But how did this match up with reality? We can plot the performance of both coins against
    compound sentiment to see which subreddit is more accurate.
    """
    txt_compare = get_reddit_dataset(compare)
    (sentiment_txt_compare, _, _) = sentiment_analyze(txt_compare)
    (other_x, other_y) = set_reliability_graph(compare, sentiment_txt_compare, plot=False)
    (x, y) = set_reliability_graph(sub, sentiment_txt)
    set_compare_graph(x, y, sub, other_x, other_y, compare)
else:
    #comparison done with other sub
    f"""
    But how did this match up with reality? We can plot the performance of the coin against
    compound sentiment to see if **{sub_name}** got it right.
    """

    set_reliability_graph(sub, sentiment_txt)
    set_sentiment_graphs(sub, sentiment_txt, neg, pos)
