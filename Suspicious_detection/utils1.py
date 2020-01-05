# -*- coding: utf-8 -*-
# file: utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import spacy
#nlp = spacy.load('en')
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def load_word_vec(word_index=None, embedding_dim=100):
    fname = './glove.twitter.27B/glove.twitter.27B.'+str(embedding_dim)+'d.txt' \
        if embedding_dim != 300 else 'E:/python/SA-DL/glove.42B.300d/glove.42B.300d.txt'
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if tokens[0] in word_index.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def read_dataset(type='twitter', mode='train', embedding_dim=100, max_seq_len=40, max_aspect_len=3, polarities_dim=2):
    print("preparing data...")
    fname = {
        'twitter': {
            'train': 'E:/python/aspect_test/SA-DL1/datasets/acl-14-short-data/combine_tweet2.csv',
            'test': 'E:/python/aspect_test/SA-DL1/datasets/acl-14-short-data/validate.csv'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }

    texts_raw_without_aspects = []    

    text = ''
    texts_raw = []
    texts_left = []
    texts_left_with_aspects = []
    texts_right = []
    texts_right_with_aspects = []
    aspects = []
    polarities = []

    with open(fname[type][mode], 'r', encoding='utf-8', newline='\n', errors='ignore') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id, polarity, aspect, text_raw = line.split(',')
            text +=text_raw
            aspects.append(aspect)
            texts_raw.append(text_raw)
            polarities.append(int(polarity))
    print("number of {0} {1} data: {2}".format(type, mode ,len(lines)))

    text_words = text.strip().split()
    print('tokenizing...')
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(text_words)
    #test_aspect_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_aspect_terms))
    word_index = tokenizer.word_index

    #label_encoder = LabelEncoder()
    #integer_category = label_encoder.fit_transform(aspects)
    #dummy_category = to_categorical(integer_category)

    texts_raw_indices = tokenizer.texts_to_sequences(texts_raw)
    #print(texts_raw_indices)
    texts_raw_indices = pad_sequences(texts_raw_indices, maxlen=max_seq_len)
    #print(texts_raw_indices)
    aspects_indices = tokenizer.texts_to_sequences(aspects)
    #print(aspects_indices)
    aspects_indices = pad_sequences(aspects_indices, maxlen=max_aspect_len)
    #print(aspects_indices)
   
    polarities_matrix = K.eval(tf.one_hot(indices=polarities, depth=polarities_dim))



    if mode == 'test':
        return texts_raw_indices, aspects_indices, polarities_matrix


    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embedding_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_word_vec(word_index, embedding_dim)
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return texts_raw_indices, aspects_indices, polarities_matrix, \
           embedding_matrix, \
           tokenizer

def read_testdata(type='twitter', embedding_dim=100, max_seq_len=40, max_aspect_len=3, polarities_dim=2): 
    print("preparing data...")
    fname = 'E:/python/aspect_test/SA-DL1/datasets/acl-14-short-data/antitrump_test.csv'
    tname = '/home/cedept/msd/msdvenv/SA-DL/datasets/semeval14/aspect_test.txt'
   
    text = ''
    texts_raw = []
    aspects = []

    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id, aspect, text_raw = line.split(',')
            text +=text_raw
            aspects.append(aspect)
            texts_raw.append(text_raw)
    print("file read" )

    text_words = text.strip().split()
    print('tokenizing...')
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(text_words)
    #test_aspect_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_aspect_terms))
    word_index = tokenizer.word_index 
    
    texts_raw_indices = tokenizer.texts_to_sequences(texts_raw)
    print(texts_raw_indices)
    texts_raw_indices = pad_sequences(texts_raw_indices, maxlen=max_seq_len) 
    print(texts_raw_indices) 
    aspects_indices = tokenizer.texts_to_sequences(aspects)
    #print(aspects_indices)
    aspects_indices = pad_sequences(aspects_indices, maxlen=max_aspect_len)
    #print(aspects_indices)

    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embedding_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_word_vec(word_index, embedding_dim)
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return texts_raw_indices, aspects_indices, aspects


if __name__ == '__main__':
    read_dataset()
