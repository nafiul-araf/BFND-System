#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_train_encode.npy', 'rb') as f:
    X_train_encode=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/X_test_encode.npy', 'rb') as f:
    X_test_encode=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_train.npy', 'rb') as f:
    y_train=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_test.npy', 'rb') as f:
    y_test=np.load(f)

X_train_encode.shape, X_test_encode.shape, y_train.shape, y_test.shape


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


clf_gb=GradientBoostingClassifier().fit(X_train_encode, y_train)


# In[ ]:


clf_gb.score(X_train_encode, y_train)


# In[ ]:


clf_gb.score(X_test_encode, y_test)


# In[ ]:


import joblib
gb_file=r"/content/drive/MyDrive/AutoEncoders/gb_file_encoder.pkl"
joblib.dump(clf_gb, gb_file)

