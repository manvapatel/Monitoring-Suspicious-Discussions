import re

import sys
from nltk.stem.porter import PorterStemmer

import json

import pymongo
import string

from collections import Counter

from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords
import pandas as pd

import numpy as np

def preprocess_word(word):

    # Remove punctuation

    word = word.strip('\'"?!,.():;')

    # Convert more than 2 letter repetitions to 2 letter

    # funnnnny --> funny

    word = re.sub(r'(.)\1+', r'\1\1', word)

    # Remove - & '

    word = re.sub(r'(-|\')', '', word)

    return word





def is_valid_word(word):

    # Check if word begins with an alphabet

    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)





def handle_emojis(tweet):

    # Smile -- :), : ), :-), (:, ( :, (-:, :')

    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)

    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D

    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)

    # Love -- <3, :*

    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)

    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;

    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)

    # Sad -- :-(, : (, :(, ):, )-:

    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)

    # Cry -- :,(, :'(, :"(

    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)

    return tweet





def preprocess_tweet(tweet):

    processed_tweet = []

    # Convert to lower case

    tweet = tweet.lower()

    # Replaces URLs with the word URL

    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', tweet)

    # Replace @handle with the word USER_MENTION

    tweet = re.sub(r'@', ' ', tweet)

    # Replaces #hashtag with hashtag

    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)

    # Remove RT (retweet)

    tweet = re.sub(r'\brt\b', '', tweet)

    # Replace 2+ dots with space

    tweet = re.sub(r'\.{2,}', ' ', tweet)

    # Strip space, " and ' from tweet

    tweet = tweet.strip(' "\'')

    # Replace emojis with either EMO_POS or EMO_NEG

    tweet = handle_emojis(tweet)

    # Replace multiple spaces with a single space

    tweet = re.sub(r'\s+', ' ', tweet)

    words = tweet.split()



    for word in words:

        word = preprocess_word(word)
        if is_valid_word(word):

            if use_stemmer:

                word = str(porter_stemmer.stem(word))

            processed_tweet.append(word)


        
    return ' '.join(processed_tweet)



conn =pymongo.MongoClient('localhost', 27017)

print('Connected successfully to MongoDB!')

#print(conn.database_names())

db=conn.TweetScraper1
#print(db.collection_names())

coll=db.tweetScrap
#print(coll.count())

results=coll.find()

#print(results)

list_results=list(results)

use_stemmer = False
if use_stemmer:
        porter_stemmer = PorterStemmer()

fname='G:/msd project/nlp/corey_test/django_project/blog/tweet_scrap.csv'
save_to_file = open(fname, 'w')
for element in list_results:

    msg = element['text']
    user = element['usernameTweet']
    id = element['ID']
    tweet = ''.join(msg)
    tweet_user = ''.join(user)
    tweet_id = ''.join(id)
    
    processed_tweet = preprocess_tweet(tweet)
    #print(processed_tweet)
    #mycol=db['ProcessedTweetBlackMatter']
    #mydict={"username":tweet_user , "ID":tweet_id , "tweets":processed_tweet}
    #x=mycol.insert_one(mydict)
    save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))

save_to_file.close()    
   