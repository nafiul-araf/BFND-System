#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[15]:


with open(r'D:\Download\corpus_autoencoder.txt', 'rb') as filehandle:
    corpus=pickle.load(filehandle)

corpus[18:21]


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[17]:


tfidf=TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X=tfidf.fit_transform(corpus).toarray()
len(X)


# In[18]:


df=pd.read_csv(r'D:\Download\final_data_autoencoder.csv')
df.head()


# In[19]:


y=df['label'].values


# In[20]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=1)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


clf3=LogisticRegression(solver='saga', max_iter=5000).fit(X_train, y_train)


# In[23]:


clf3.score(X_train, y_train)


# In[24]:


clf3.score(X_test, y_test)


# In[25]:


import joblib
lr_file=r"D:\Download\lr_file_tfidf.pkl"
joblib.dump(clf3, lr_file)

