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


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


tfidf=TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X=tfidf.fit_transform(corpus).toarray()


# In[5]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[6]:


y=df['label']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[8]:


clf7=DecisionTreeClassifier().fit(X_train, y_train)


# In[9]:


clf7.score(X_train, y_train)


# In[10]:


clf7.score(X_test, y_test)


# In[11]:


import joblib
dt_file=r"D:\Download\dt_file.pkl"
joblib.dump(clf7, dt_file)

