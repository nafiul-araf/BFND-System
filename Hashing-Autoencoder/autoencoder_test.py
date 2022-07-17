#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow.keras.models import load_model


# In[2]:


with tensorflow.device('/CPU:0'):
    encoder=load_model('D:\Download\encoder_eff.h5')


# In[3]:


#model=load_model('/content/drive/MyDrive/AutoEncoders/encoder_v2.h5')


# In[4]:


from sklearn.datasets import make_classification


# In[5]:


X, y=make_classification(n_samples=85000, n_features=5000)


# In[6]:


X=X/X.max()


# In[7]:


with tensorflow.device('/GPU:0'):
    encoder.evaluate(X, X)


# In[8]:


#model.evaluate(X, X)

