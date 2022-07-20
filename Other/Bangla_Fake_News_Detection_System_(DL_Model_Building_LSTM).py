#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[3]:


with open('/content/gdrive/MyDrive/AutoEncoders/corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[4]:


df=pd.read_csv('/content/gdrive/MyDrive/AutoEncoders/final_data_autoencoder.csv')
df.head()


# In[5]:


y=df['label']


# In[6]:


import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, Bidirectional, LSTM
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split


# In[7]:


voc_size=150000
onehot_repr=[one_hot(words, voc_size) for words in corpus] 
onehot_repr


# In[8]:


sent_length=300
embedded_docs=pad_sequences(onehot_repr, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[9]:


len(embedded_docs)


# In[10]:


X_final=np.array(embedded_docs)
y_final=np.array(y)

X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.15, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[11]:


embedding_vector_features=500
emb_layer=Embedding(voc_size, embedding_vector_features)
print(emb_layer)


# In[12]:


def lstm_model(input_length, vocab_size):
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


# In[13]:


model=lstm_model(input_length=sent_length, vocab_size=voc_size)


# In[14]:


model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64, use_multiprocessing=True)


# In[15]:


model.evaluate(X_test, y_test)


# In[16]:


model.save(r'/content/gdrive/MyDrive/CSE-400(Project Thesis)/model_lstm.h5')

