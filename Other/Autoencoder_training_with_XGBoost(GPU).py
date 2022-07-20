#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


with open(r'D:\Download\X_train_encode.npy', 'rb') as f:
    X_train_encode=np.load(f)

with open(r'D:\Download\X_test_encode.npy', 'rb') as f:
    X_test_encode=np.load(f)

with open(r'D:\Download\y_train.npy', 'rb') as f:
    y_train=np.load(f)

with open(r'D:\Download\y_test.npy', 'rb') as f:
    y_test=np.load(f)

X_train_encode.shape, X_test_encode.shape, y_train.shape, y_test.shape


# In[3]:


from xgboost import XGBClassifier


# In[4]:


clf_xgb=XGBClassifier(learning_rate=1, tree_method='gpu_hist').fit(X_train_encode, y_train)


# In[5]:


clf_xgb.score(X_train_encode, y_train)


# In[6]:


clf_xgb.score(X_test_encode, y_test)


# In[7]:


import joblib
xgb_file=r"D:\Download\xgb_file_encoder.pkl"
joblib.dump(clf_xgb, xgb_file)

