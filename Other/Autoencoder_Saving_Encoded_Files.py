#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import warnings
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
warnings.filterwarnings('ignore')


# In[2]:


with open(r'D:\Download\corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer


# In[4]:


hashing=HashingVectorizer(n_features=5000, ngram_range=(1, 3))
X=hashing.fit_transform(corpus).toarray()
len(X)


# In[8]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[9]:


y=df['label'].values


# In[10]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[11]:


K.clear_session()


# In[12]:


X_test


# In[13]:


def norm(features):
  features=features/features.max()

  return features

X_train=norm(X_train)
X_test=norm(X_test)
X_test


# In[15]:


with open('D:\Download\X_train.npy', 'wb') as f:
    np.save(f, X_train)

with open('D:\Download\X_test.npy', 'wb') as f:
    np.save(f, X_test)

with open('D:\Download\y_test.npy', 'wb') as f:
    np.save(f, y_test)

with open('D:\Download\y_train.npy', 'wb') as f:
    np.save(f, y_train)


# In[ ]:




