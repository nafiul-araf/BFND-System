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
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


# In[ ]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_train_encode.npy', 'rb') as f:
    X_train=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/X_test_encode.npy', 'rb') as f:
    X_test=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_train.npy', 'rb') as f:
    y_train=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/y_test.npy', 'rb') as f:
    y_test=np.load(f)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


clf_lr=LogisticRegression(solver='saga', max_iter=5000).fit(X_train, y_train)


# In[ ]:


clf_lr.score(X_train, y_train)


# In[ ]:


clf_lr.score(X_test, y_test)


# In[ ]:


import joblib
lr_file=r"/content/drive/MyDrive/AutoEncoders/lr_file_encoder.pkl"
joblib.dump(clf_lr, lr_file)

