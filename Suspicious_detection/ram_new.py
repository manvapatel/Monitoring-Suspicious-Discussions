# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import TensorBoard, LambdaCallback
#from tensorflow.python.keras.utils import plot_model
from utils1 import read_dataset, read_testdata
from custom_metrics import f1
from attention_layer import Attention
import tensorflow as tf
from tensorflow.python.keras import initializers, regularizers, optimizers, backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Embedding, Bidirectional, Lambda, Flatten, Dropout
from tensorflow.python.keras.utils import CustomObjectScope
#from keras.models import load_model
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from sklearn.preprocessing import LabelEncoder
#from keras.utils import to_categorical


class RecurrentAttentionMemory:
    def __init__(self):
        self.HOPS = 5
        self.SCORE_FUNCTION = 'mlp'  # scaled_dot_product / mlp (concat) / bi_linear (general dot)
        self.DATASET = 'twitter'  # 'twitter', 'restaurant', 'laptop'
        self.POLARITIES_DIM = 2
        self.EMBEDDING_DIM = 300
        self.LEARNING_RATE = 0.001
        self.INITIALIZER = initializers.RandomUniform(minval=-0.05, maxval=0.05)
        self.REGULARIZER = regularizers.l2(0.001)
        self.LSTM_PARAMS = {
            'units': 200,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'kernel_initializer': self.INITIALIZER,
            'recurrent_initializer': self.INITIALIZER,
            'bias_initializer': self.INITIALIZER,
            'kernel_regularizer': self.REGULARIZER,
            'recurrent_regularizer': self.REGULARIZER,
            'bias_regularizer': self.REGULARIZER,
            'dropout': 0.4,
            'recurrent_dropout': 0,
        }
        self.MAX_SEQUENCE_LENGTH = 80
        self.MAX_ASPECT_LENGTH = 10
        self.BATCH_SIZE = 200
        self.EPOCHS = 20

        self.texts_raw_indices, self.aspects_indices, self.polarities_matrix, \
        self.embedding_matrix, \
        self.tokenizer = \
            read_dataset(type=self.DATASET,
                         mode='train',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

                    
    def train(self):
        train = 1
        tbCallBack = TensorBoard(log_dir='./ram_logs', histogram_freq=0, write_graph=True, write_images=True)
        def modelSave(epoch, logs):
            if (epoch + 1) % 5 == 0:
                self.model.save('ram_saved_model2.h5')
        msCallBack = LambdaCallback(on_epoch_end=modelSave)

        if train:
          print('Build model...')
          inputs_sentence = Input(shape=(self.MAX_SEQUENCE_LENGTH,), name='inputs_sentence')
          inputs_aspect = Input(shape=(self.MAX_ASPECT_LENGTH,), name='inputs_aspect')
          nonzero_count = Lambda(lambda xin: tf.count_nonzero(xin, dtype=tf.float32))(inputs_aspect)
          sentence = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                        output_dim=self.EMBEDDING_DIM,
                        input_length=self.MAX_SEQUENCE_LENGTH,
                        mask_zero=True,
                        weights=[self.embedding_matrix],
                        trainable=False, name='sentence_embedding')(inputs_sentence)
          aspect = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                        output_dim=self.EMBEDDING_DIM,
                        input_length=self.MAX_ASPECT_LENGTH,
                        mask_zero=True,
                        weights=[self.embedding_matrix],
                        trainable=False, name='aspect_embedding')(inputs_aspect)
          memory = Bidirectional(LSTM(return_sequences=True, **self.LSTM_PARAMS), name='memory')(sentence)
          aspect = Bidirectional(LSTM(return_sequences=True, **self.LSTM_PARAMS), name='aspect')(aspect)
          x = Lambda(lambda xin: K.sum(xin[0], axis=1) / xin[1], name='aspect_mean')([aspect, nonzero_count])
          shared_attention = Attention(score_function=self.SCORE_FUNCTION,
                                    initializer=self.INITIALIZER, regularizer=self.REGULARIZER,
                                    name='shared_attention')
          for i in range(self.HOPS):
           x = shared_attention((memory, x))
            
          x = Flatten()(x)
          x = Dropout(0.4)(x)
          x = Dense(self.POLARITIES_DIM)(x)
          predictions = Activation('softmax')(x)
          model = Model(inputs=[inputs_sentence, inputs_aspect], outputs=predictions)
          model.summary()
          model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.LEARNING_RATE), metrics=['acc', f1])
          #plot_model(model, to_file='model.png')
          self.model = model
          self.shared_attention = shared_attention

          texts_raw_indices, \
          aspects_indices, \
          polarities_matrix = \
              read_dataset(type=self.DATASET,
                           mode='test',
                           embedding_dim=self.EMBEDDING_DIM,
                           max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

          history = self.model.fit([self.texts_raw_indices, self.aspects_indices], self.polarities_matrix,
                       validation_split=0.2, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, callbacks=[msCallBack])

          pd.DataFrame(history.history).to_csv("history.csv")

          #aspect_categories_model.fit(self.test_aspect_terms, self.dummy_category, epochs=5, verbose=1)
          texts_raw_indi, aspects_indi, aspects  = \
              read_testdata(type=self.DATASET, embedding_dim=self.EMBEDDING_DIM, max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

          predictions = self.model.predict([texts_raw_indi, aspects_indi], batch_size=self.BATCH_SIZE, verbose=1)  
          for i in range(len(texts_raw_indi)):
             print("X:%s, predicted=%s, value=%s" % (texts_raw_indi[i], predictions[i], np.argmax(predictions[i])))
          
          new_file=open("E:\\python\\aspect_test\\SA-DL1\\newfile.txt",mode="w",encoding="utf-8")
          for i in range(len(texts_raw_indi)):
                new_file.write("Review %s is expressing a %s , %s opinion about aspect %s" % (str(i+1), predictions[i], np.argmax(predictions[i]), aspects[i]))

                #print("Review %s is expressing a %s , %s opinion about aspect %s" % (str(i+1), predictions[i], np.argmax(predictions[i]), aspects[i]))


        else:
          print('loading saved model...')
          self.model = load_model('ram_saved_model1.h5', custom_objects={'Attention' : Attention, 'f1' : f1})
          print(self.model.summary())
          
          texts_raw_indi, aspects_indi, aspects  = \
              read_testdata(type=self.DATASET, embedding_dim=self.EMBEDDING_DIM, max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

          predictions = self.model.predict([texts_raw_indi, aspects_indi], batch_size=self.BATCH_SIZE, verbose=1)  
          for i in range(len(texts_raw_indi)):
             print("X:%s, predicted=%s, value=%s" % (texts_raw_indi[i], predictions[i], np.argmax(predictions[i])))

          for i in range(len(texts_raw_indi)):
                print("Review %s is expressing a %s , %s opinion about aspect %s" % (str(i+1), predictions[i], np.argmax(predictions[i]), aspects[i]))



if __name__ == '__main__':
    model = RecurrentAttentionMemory()
    model.train()
    #print(len(model.model.get_weights()))
