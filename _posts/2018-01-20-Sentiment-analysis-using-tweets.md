---
layout: post
title: Sentiment Analysis
subtitle: From twitter data 
image: /img/twitter.png
bigimg: /img/twitter.png
---

Sentiment analysis categorizes opinions of users on a given subject into three broad categories - positive, negative and neutral.
These opinions could be expressed through natural langauge or text. In this post, I will analyse emotion or sentiment of textual expressions put out on twitter.
<p> Twitter is a popular platform for people to voice their opinions and can be used for a fairly accurate assesment of public
sentiment on a subject. Twitter provides us with a public API to query their database and these results are the input for my task. </p>

<p> I will be using the public API provided by twitter to query their database. One of the limitations of this API is that it can only get you tweets that are two weeks old or less. Tweets older than that will have to be done using other methods. This restriction limited the topics I could run sentiment analysis on and I had to choose a recent event around the time of this project which was the Golden Globes 2018 award ceremony. One of the highlights of the event was Oprah's speech and how it generated a widely popular sentiment of she running for POTUS. I will be carry out sentiment analysis on Oprah for President to understand if twitter users favor or oppose the idea. </p>

#### Twitter API access
As a first step, I will want access to the twitter API. This can be done by creating a new application on the [Twitter - Developer](https://apps.twitter.com/). This will give me a consumer key, consumer secret code, and access token/secret code, all of which are needed to get access to the API.
<p> </p>
#### Query preparation
<p> Once I am ready to query the database, I need to determine the search terms. For the sentiment in my task, search terms could be "Oprah" and "President"  or "Oprah" and "POTUS" or "Oprah" and "Prez" and so on. To limit the number of tweets to analyse, I will limit my query to "Oprah" and "POTUS". </p>
<p> The event date was Jan 7th, hence I will run the query from Jan 7th upto the present which happens to be Jan 18th.</p>
<p> I will consider original tweets only as the nummber of tweets would be massive if retweets are considered. I would also have to address rate limit issues if retweets are queried. </p>
<p> I will search for tweets using cursor pagination and a max number of tweets specified. An useful link on </p> [twitter API search](https://stackoverflow.com/questions/22469713/managing-tweepy-api-search/22473254#22473254)
<p> All the tweets queried will be enoded in the UTF8 format. </p> 
<p> </p>
#### Twitter query code
```python

import tweepy

# UTF8 encoding
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# fill in your Twitter credentials 
consumer_key = "abcdef"
consumer_secret = "abcdef"
access_token = "abcdef"
access_token_secret = "abcdef"

# An instance of REST API created
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Query the database
query = '"Oprah" and "POTUS" -filter:retweets'
max_tweets = 10000
lang = "en"
searched_tweets = [status for status in tweepy.Cursor(api.search, q=query, langauge=lang, since ="2018-01-07").items(max_tweets)]
```



