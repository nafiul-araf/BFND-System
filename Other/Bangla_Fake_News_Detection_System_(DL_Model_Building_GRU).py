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


# In[5]:


with open(r'/content/gdrive/MyDrive/AutoEncoders/corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[6]:


df=pd.read_csv(r'/content/gdrive/MyDrive/AutoEncoders/final_data_autoencoder.csv')
df.head()


# In[7]:


y=df['label']


# In[8]:


import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, GRU
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split


# In[9]:


voc_size=150000
onehot_repr=[one_hot(words, voc_size) for words in corpus] 
onehot_repr


# In[10]:


sent_length=300
embedded_docs=pad_sequences(onehot_repr, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[11]:


len(embedded_docs)


# In[12]:


X_final=np.array(embedded_docs)
y_final=np.array(y)

X_train, X_test, y_train, y_test=train_test_split(X_final, y_final, test_size=0.15, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[13]:


embedding_vector_features=500
emb_layer=Embedding(voc_size, embedding_vector_features)
print(emb_layer)


# In[14]:


def GRU_model(input_length, vocab_size):
    input=Input(shape=(input_length,))
    x=emb_layer(input)
    
    GRU_1=GRU(32)(x)
    GRU_1=Dropout(0.20)(GRU_1)
    
    GRU_2=GRU(64)(x)
    
    GRU_3=GRU(128)(x)
    GRU_3=Dropout(0.30)(GRU_3)
    
    
    GRU_layers=concatenate([GRU_1, GRU_2, GRU_3])
    
    dense=Dense(256, activation='relu')(GRU_layers)
    drop=Dropout(0.2)(dense)
    output=Dense(1, activation='sigmoid')(drop)
    model=Model(inputs=input, outputs=output)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    return model


# In[15]:


model=GRU_model(input_length=sent_length, vocab_size=voc_size)


# In[17]:


model.fit(X_train, y_train, epochs=7, validation_data=(X_test, y_test), batch_size=64, use_multiprocessing=True)


# In[18]:


model.evaluate(X_test, y_test)


# In[19]:


model.save(r'/content/gdrive/MyDrive/CSE-400(Project Thesis)/model_GRU.h5')

