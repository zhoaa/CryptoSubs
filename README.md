# CryptoSubs

CryptoSubs is a web app designed to analyze historical positive and negative sentiment on a cryptocurrency subreddit day by day, and compares those values to the actual price change of the cryptocurrency on those days.

This allows us to find an "accuracy" score for the subreddit based on its sentiment. 

Which subreddits are best for a budding investor?

When r/BTC is celebrating, is Bitcoin mooning? 

Is r/DogeCoin genius or delusional?

These questions can begin to be answered with data from CryptoSubs.

# Setup

Pip Installations (```pip install```):
```streamlit```
```praw```
```nltk```
```pandas```
```numpy```
```plotly```
```seaborn```
```pathlib```
```matplotlib```
```DateTime```

Setup:
- ```git clone``` peddiehacks2021 on Github
- On command line, ```cd``` into the peddiehacks2021 folder
- ```pip install``` all of the above
- _Optional: If you wish to generate your own datasets instead of using the ones given above, you can delete the .csv files within datasets (do not delete the datasets folder itself)_
- On command line, run ```streamlit run app.py```
