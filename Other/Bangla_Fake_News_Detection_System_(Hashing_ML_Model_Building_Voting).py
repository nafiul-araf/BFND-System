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
X=hashing.fit_transform(corpus).toarray()


# In[5]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[6]:


y=df['label']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[7]:


from sklearn.ensemble import VotingClassifier


# In[9]:


import joblib

pac_model=joblib.load(r"D:\Download\pac_file_hash.pkl")
print(dir(pac_model))
lr_model=joblib.load(r"D:\Download\lr_file_hash.pkl")
print(dir(lr_model))
dt_model=joblib.load(r"D:\Download\dt_file_hash.pkl")
print(dir(dt_model))
lgbm_model=joblib.load(r"D:\Download\lgbm_file_hash.pkl")
print(dir(lgbm_model))


# In[10]:


clf8=VotingClassifier(estimators=[('PassiveAgressiveClassifier', pac_model), ('LogisticRegression', lr_model), 
                                   ('DecisionTreeClassifier', dt_model), ('LGBM', lgbm_model)], weights = [0.99, 0.96, 0.98, 0.98], 
                       voting='hard')
clf8.fit(X_train, y_train)


# In[11]:


clf8.score(X_train, y_train)


# In[12]:


clf8.score(X_test, y_test)


# In[13]:


import joblib
vt_file=r"D:\Download\vt_file_hash.pkl"
joblib.dump(clf8, vt_file)

