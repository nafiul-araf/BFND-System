#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import pickle
with open(r'/content/drive/MyDrive/AutoEncoders/corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

import pandas as pd
df=pd.read_csv(r'/content/drive/MyDrive/AutoEncoders/final_data_autoencoder.csv')

import numpy as np
y=df['label']
df.head()


# In[4]:


corpus[13:15]


# In[5]:


import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Add, LSTM, Dense, AveragePooling1D, Dropout, Embedding, Flatten, Activation, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


# In[6]:


voc_size=150000
onehot_repr=[one_hot(words, voc_size) for words in corpus] 
onehot_repr


# In[7]:


sent_length=300
embedded_docs=pad_sequences(onehot_repr, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[8]:


embedding_vector_features=500
emb_layer=Embedding(voc_size, embedding_vector_features)


# In[9]:


def define_model(input_length, vocab_size):
    input=Input(shape=(input_length,))
    embedding_layer_1=emb_layer(input)
    conv1=Conv1D(filters=32, kernel_size=4, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_1)
    batch1=BatchNormalization(scale=False)(conv1)
    pool1=MaxPooling1D(pool_size=2)(batch1)
    flat1=Flatten()(pool1)
    
    embedding_layer_2=emb_layer(input)
    conv2=Conv1D(filters=64, kernel_size=6, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_2)
    batch2=BatchNormalization(scale=False)(conv2)
    pool2=MaxPooling1D(pool_size=2)(batch2)
    flat2=Flatten()(pool2)
    
    embedding_layer_3=emb_layer(input)
    conv3=Conv1D(filters=128, kernel_size=8, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_3)
    batch3=BatchNormalization(scale=False)(conv3)
    pool3=MaxPooling1D(pool_size=2)(batch3)
    flat3=Flatten()(pool3)
    
    CNN_layer=concatenate([flat1, flat2, flat3])
    
    x=emb_layer(input)
    
    LSTM_layer1=Bidirectional(LSTM(128))(x)
    LSTM_layer1=Dropout(0.20)(LSTM_layer1)
    
    LSTM_layer2=Bidirectional(LSTM(64))(x)
    LSTM_layer2=Dropout(0.10)(LSTM_layer2)
    
    LSTM_layer3=Bidirectional(LSTM(32))(x)
    LSTM_layer3=Dropout(0.10)(LSTM_layer3)
    
    LSTM_layer=concatenate([LSTM_layer1, LSTM_layer2, LSTM_layer3])
    
    CNN_LSTM_layer=concatenate([LSTM_layer, CNN_layer])
    
    dense1=Dense(256, activation='relu')(CNN_LSTM_layer)
    outputs=Dense(1, activation='sigmoid')(dense1)
    model=Model(inputs=input, outputs=outputs)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model


# In[10]:


model=define_model(input_length=sent_length, vocab_size=voc_size)


# In[11]:


len(embedded_docs)


# In[12]:


X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.15, random_state=42)


# In[13]:


model.fit(X_train, y_train, epochs=5, validation_split=0.15, batch_size=64, use_multiprocessing=True)


# In[14]:


#model.save('/content/drive/MyDrive/CSE-400(Project Thesis)/cnn_lstm.h5')


# In[15]:


model.evaluate(X_test, y_test)

