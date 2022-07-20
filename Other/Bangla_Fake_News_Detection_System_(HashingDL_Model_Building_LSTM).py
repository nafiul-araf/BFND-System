#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import pickle
with open(r'/content/drive/MyDrive/AutoEncoders/corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

import pandas as pd
df=pd.read_csv(r'/content/drive/MyDrive/AutoEncoders/final_data_autoencoder.csv')

import numpy as np
y=df['label']


# In[3]:


corpus[13:15]


# In[4]:


import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, Bidirectional, LSTM
from keras.layers.merge import concatenate
from keras import regularizers
from keras.preprocessing.text import one_hot, hashing_trick
from keras.preprocessing.sequence import pad_sequences


# In[5]:


voc_size=150000
hashing_corpus=[hashing_trick(words, n=voc_size, hash_function='md5', lower=False) for words in corpus]


# In[6]:


hashing_corpus


# In[7]:


sent_length=300
embedded_docs=pad_sequences(hashing_corpus, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[8]:


len(hashing_corpus)


# In[9]:


embedding_vector_features=500
emb_layer=Embedding(voc_size, embedding_vector_features)


# In[10]:


def define_model(input_length, vocab_size):
    input=Input(shape=(input_length,))
    x=emb_layer(input)
    
    lstm_1=Bidirectional(LSTM(32))(x)
    lstm_1=Dropout(0.20)(lstm_1)
    
    lstm_2=Bidirectional(LSTM(64))(x)
    
    lstm_3=Bidirectional(LSTM(128))(x)
    lstm_3=Dropout(0.30)(lstm_3)
    
    
    lstm_layers=concatenate([lstm_1, lstm_2, lstm_3])
    
    dense=Dense(256, activation='relu')(lstm_layers)
    drop=Dropout(0.2)(dense)
    output=Dense(1, activation='sigmoid')(drop)
    model=Model(inputs=input, outputs=output)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model


# In[11]:


model=define_model(input_length=sent_length, vocab_size=voc_size)


# In[12]:


len(embedded_docs)


# In[13]:


X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.15, random_state=42)


# In[14]:


tf.keras.backend.clear_session()


# In[15]:


model.fit(X_train, y_train, epochs=7, validation_split=0.15, batch_size=64, use_multiprocessing=True)


# In[16]:


model.save('/content/drive/MyDrive/CSE-400(Project Thesis)/hash_lstm.h5')

