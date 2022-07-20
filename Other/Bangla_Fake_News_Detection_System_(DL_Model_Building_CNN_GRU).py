#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import pickle
import pandas as pd
import numpy as np


# In[3]:


with open(r'/content/drive/MyDrive/AutoEncoders/corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[4]:


df=pd.read_csv(r'/content/drive/MyDrive/AutoEncoders/final_data_autoencoder.csv')
df.head()


# In[5]:


y=df['label']


# In[6]:


import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Embedding, Flatten, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
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

X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.10, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[11]:


embedding_vector_features=500
emb_layer=Embedding(voc_size, embedding_vector_features)
print(emb_layer)


# In[12]:


def define_model(input_length, vocab_size):
    input=Input(shape=(input_length,))
    embedding_layer_1=emb_layer(input)
    conv1=Conv1D(filters=16, kernel_size=4, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_1)
    batch1=BatchNormalization(scale=False)(conv1)
    pool1=MaxPooling1D(pool_size=2)(batch1)
    flat1=Flatten()(pool1)
    
    embedding_layer_2=emb_layer(input)
    conv2=Conv1D(filters=32, kernel_size=6, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_2)
    batch2=BatchNormalization(scale=False)(conv2)
    pool2=MaxPooling1D(pool_size=2)(batch2)
    flat2=Flatten()(pool2)
    
    embedding_layer_3=emb_layer(input)
    conv3=Conv1D(filters=64, kernel_size=8, padding='causal', activation='relu', kernel_initializer='he_normal')(embedding_layer_3)
    batch3=BatchNormalization(scale=False)(conv3)
    pool3=MaxPooling1D(pool_size=2)(batch3)
    flat3=Flatten()(pool3)
    
    CNN_layer=concatenate([flat1, flat2, flat3])
    
    x=emb_layer(input)
    
    GRU_layer1=GRU(128)(x)
    GRU_layer1=Dropout(0.20)(GRU_layer1)
    
    GRU_layer2=GRU(64)(x)
    GRU_layer2=Dropout(0.10)(GRU_layer2)
    
    GRU_layer3=GRU(32)(x)
    GRU_layer3=Dropout(0.10)(GRU_layer3)
    
    GRU_layer=concatenate([GRU_layer1, GRU_layer2, GRU_layer3])
    
    CNN_GRU_layer=concatenate([GRU_layer, CNN_layer])
    
    dense1=Dense(256, activation='relu')(CNN_GRU_layer)
    outputs=Dense(1, activation='sigmoid')(dense1)
    model=Model(inputs=input, outputs=outputs)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model


# In[13]:


model=define_model(input_length=sent_length, vocab_size=voc_size)


# In[14]:


model.fit(X_train, y_train, epochs=7, validation_split=0.15, batch_size=64, use_multiprocessing=True)


# In[15]:


model.save('/content/drive/MyDrive/CSE-400(Project Thesis)/cnn_gru.h5')


# In[16]:


model.evaluate(X_test, y_test)

