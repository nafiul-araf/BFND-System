#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import numpy as np
import pandas as pd
import pickle
import warnings
import joblib
warnings.filterwarnings('ignore')


# In[3]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_train_encode.npy', 'rb') as f:
    X_train_encode=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/X_test_encode.npy', 'rb') as f:
    X_test_encode=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_train.npy', 'rb') as f:
    y_train=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_test.npy', 'rb') as f:
    y_test=np.load(f)

X_train_encode.shape, X_test_encode.shape, y_train.shape, y_test.shape


# In[4]:


pac_model_encoder=joblib.load(r"/content/drive/MyDrive/AutoEncoders/pac_file_encoder.pkl")
print(dir(pac_model_encoder))
lr_model_encoder=joblib.load(r"/content/drive/MyDrive/AutoEncoders/lr_file_encoder.pkl")
print(dir(lr_model_encoder))
dt_model_encoder=joblib.load(r"/content/drive/MyDrive/AutoEncoders/dt_file_encoder.pkl")
print(dir(dt_model_encoder))
lgbm_model_encoder=joblib.load(r"/content/drive/MyDrive/AutoEncoders/lgbm_file_encoder.pkl")
print(dir(lgbm_model_encoder))


# In[6]:


from sklearn.ensemble import VotingClassifier


# In[7]:


model=VotingClassifier(estimators=[('PassiveAgressiveClassifier', pac_model_encoder), ('LogisticRegression', lr_model_encoder), 
                                   ('DecisionTreeClassifier', dt_model_encoder), ('LGBM', lgbm_model_encoder)], weights = [0.89, 0.97, 0.84, 0.88], 
                       voting='hard')
model.fit(X_train_encode, y_train)


# In[8]:


model.score(X_train_encode, y_train)


# In[9]:


model.score(X_test_encode, y_test)


# In[10]:


import joblib
vote_file=r"/content/drive/MyDrive/AutoEncoders/voting_file_encoder.pkl"
joblib.dump(model, vote_file)

