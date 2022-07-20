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
from sklearn.feature_extraction.text import HashingVectorizer


# In[4]:


hashing=HashingVectorizer(n_features=5000, ngram_range=(1, 3))


# In[5]:


X=hashing.fit_transform(corpus).toarray()


# In[6]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[7]:


y=df['label']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[8]:


from xgboost import XGBClassifier


# In[9]:


clf_gpu_xgb=XGBClassifier(learning_rate=1, tree_method='gpu_hist').fit(X_train, y_train)


# In[10]:


clf_gpu_xgb.score(X_train, y_train)


# In[11]:


clf_gpu_xgb.score(X_test, y_test)


# In[12]:


import joblib
xgb_gpu_file=r"D:\Download\xgb_file_hash.pkl"
joblib.dump(clf_gpu_xgb, xgb_gpu_file)

