#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


with open(r'D:\Download\corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[3]:


len(corpus)


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


tfidf=TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X=tfidf.fit_transform(corpus).toarray()


# In[6]:


len(X)


# In[7]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[8]:


y=df['label'].values


# In[9]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[10]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[11]:


clf2=PassiveAggressiveClassifier(max_iter=5000)


# In[12]:


clf2.fit(X_train, y_train)


# In[13]:


clf2.score(X_train, y_train)


# In[14]:


clf2.score(X_test, y_test)


# In[15]:


import joblib
pac_file=r"D:\Download\pac_file.pkl"
joblib.dump(clf2, pac_file)

