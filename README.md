# Twitter

import tweepy

access_token = "..."
access_token_secret = "..."
consumer_key = "..."
consumer_secret = "..."

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        if self.num_tweets < 1000:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)
        
import json

l = MyStreamListener()
stream = tweepy.Stream(auth, l)

stream.filter(track=['covid19 vaccine', 'covid-19 vaccine', 'coronavirus vaccine', 'covid19 vaccines', 'covid-19 vaccines', 'coronavirus vaccines', 'vaccine', 'vaccines'])

# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open connection to file
tweets_file = open(tweets_data_path, "r")

# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)

# Close connection to file
tweets_file.close()

# Print the keys of the first tweet dict
print(tweets_data[0].keys())

import pandas as pd

df = pd.DataFrame(tweets_data, columns=["text","lang"])
print(df)


# Print tweet text
print(tweet['text'])

# Print tweet id
print(tweet["id"])

# Print user handle
print(tweet["user"]["screen_name"])

# Print user follower count
print(tweet["user"]["followers_count"])

# Print user location
print(tweet['user']['location'])

# Print user description
print(tweet['user']['description'])

#Sentiment Analysis
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

sentiment_scores = df['text'].apply(sid.polarity_scores)
sentiment = sentiment_scores.apply(lambda x: x['compound'])
# Print out the text of a positive tweet
print(df[sentiment > 0.6]['text'].values[0])
# Print out the text of a negative tweet
print(df[sentiment < -0.6]['text'].values[0])
df['sentiment_score'] = sentiment_scores

#Text Blob
from textblob import TextBlob

# Create a function to get the subjectivity
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)

# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
 if score < 0:
  return 'Negative'
 elif score == 0:
  return 'Neutral'
 else:
  return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)

#Show # of positive,neural,negative
df['Analysis'].value_counts()

#Graph of sentiment analysis
import seaborn as sns
import matplotlib.pyplot as plt

colors = ["#CD4FDE"]
sns.set_palette(sns.color_palette(colors))
plt.title('Sentiment Analysis')
df['Analysis'].value_counts().plot(kind = 'bar')
plt.show()

#Wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def wordcloud_by_province(tweets):
    
    stopwords = set(STOPWORDS)
    stopwords.update(["covid", "covid19", "vaccines", "vaccine", "https", "will", "coronavirus", "rt"])
   
    wordcloud = WordCloud(max_font_size=50, max_words=20, background_color="white",stopwords=stopwords, random_state = 2016).generate(" ".join([i for i in tweets['text'].str.upper()]))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")
  
wordcloud_by_province(df) 
