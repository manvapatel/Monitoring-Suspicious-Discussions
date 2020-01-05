import pandas as pd
#import spacy
#nlp = spacy.load('en')
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
#from pymagnitude import MagnitudeUtils
from pymagnitude import *
glove = Magnitude('blog/magnitude_data/glove.twitter.27B.100d.magnitude')

fname='E:/python/aspect_test/traindata_extra1.csv'
      
aspects = []
tweets = []
with open(fname, 'r', encoding='utf-8', errors='ignore') as csv:
    lines = csv.readlines()
    total = len(lines)
    for i, line in enumerate(lines):
          tweet_id, aspect, text_raw = line.split(',')
          aspects.append(aspect)
          tweets.append(text_raw.split(' '))
    #print(tweets)  

aspect_tokenized = glove.query(tweets, pad_to_length = 70)    
#print(aspect_tokenized)  
label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(aspects)
dummy_category = to_categorical(integer_category)


class AspectGeneration:
   
    def trainaspect(self):  
      train = 0
      if train:
        aspect_categories_model = Sequential()

        aspect_categories_model.add(Conv1D(256, kernel_size=3, padding='valid', activation='relu', input_shape=(70,100)))
        aspect_categories_model.add(Conv1D(128, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        #aspect_categories_model.add(Conv1D(64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.025)))
        aspect_categories_model.add(MaxPooling1D(pool_size=3))
        aspect_categories_model.add(Flatten())
        aspect_categories_model.add(Dense(64, activation = "relu", kernel_regularizer=regularizers.l2(0.01)))
        #aspect_categories_model.add(Flatten())
        aspect_categories_model.add(Dropout(0.5))
        aspect_categories_model.add(Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))


        aspect_categories_model.summary()
        aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = aspect_categories_model.fit(aspect_tokenized, dummy_category, batch_size=128, epochs=20, validation_split=0.2, verbose=1)
        pd.DataFrame(history.history).to_csv("history.csv")
        aspect_categories_model.save('aspect_saved_model_200.h5')
        print("model trained")
     
        gname='E:/python/aspect_test/testdata_new.csv'
        test_reviews = []
        with open(gname, 'r', encoding='utf-8', errors='ignore') as csv:
            lines = csv.readlines()
            total = len(lines)
            for i, line in enumerate(lines):
               tweet_id, text_raw = line.split(',')
               test_reviews.append(text_raw.split(' '))
               #print(test_reviews)

        test_aspect_terms = glove.query(test_reviews, pad_to_length = 70)   
        #print(test_aspect_terms)  

        test = aspect_categories_model.predict_classes(test_aspect_terms)
        
        test_aspect_categories = label_encoder.inverse_transform(test)
        #print(test_aspect_categories)

        for i in range(26):
             print("Review %s is expressing opinion about aspect %s" % (str(i+1), test_aspect_categories[i]))
             #print("Review is expressing opinion about aspect %s" % (test_aspect_categories[i]))
          

      else:
          gname='E:/python/Django_Blog/03-Templates/django_project/blog/tweet_scrap.csv'
          print('loading saved model...')
          aspect_categories_model = load_model('blog/aspect_saved_model_200.h5')
          print(aspect_categories_model.summary())

          test_reviews = []
          texts = []
          author = []
          text_org = []
          with open(gname, 'r', encoding='utf-8', errors='ignore') as csv:
            lines = csv.readlines()
            total = len(lines)
            for i, line in enumerate(lines):
               tweet_id, tweet_user, text_raw = line.split(',')
               author.append(tweet_user)
               texts.append(text_raw)
               test_reviews.append(text_raw.split(' '))
               #print(test_reviews)
          #glove = self.glove1
          test_aspect_terms = glove.query(test_reviews, pad_to_length = 70)   
          #print(test_aspect_terms)  

          test = aspect_categories_model.predict_classes(test_aspect_terms)
          #label_encoder = self.label_encoder1
          test_aspect_categories = label_encoder.inverse_transform(test)
          #print(test_aspect_categories)

          for i in range(total):
             print("Review %s is expressing opinion about aspect %s" % (str(i+1), test_aspect_categories[i]))
             #print("Review is expressing opinion about aspect %s" % (test_aspect_categories[i]))
             text_org.append(texts[i].rstrip())
             texts[i] = texts[i].rstrip() + ' ' + test_aspect_categories[i]
          print(texts)
      return texts, test_aspect_categories, author, text_org

if __name__ == '__main__':
    model = AspectGeneration()
    model.trainaspect()     