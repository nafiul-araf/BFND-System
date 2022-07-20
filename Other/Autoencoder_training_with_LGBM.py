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


from lightgbm import LGBMClassifier


# In[ ]:


clf_lgbm=LGBMClassifier(learning_rate=0.1).fit(X_train_encode, y_train)


# In[ ]:


clf_lgbm.score(X_train_encode, y_train)


# In[ ]:


clf_lgbm.score(X_test_encode, y_test)


# In[ ]:


import joblib
lgbm_file=r"/content/drive/MyDrive/AutoEncoders/lgbm_file_encoder.pkl"
joblib.dump(clf_lgbm, lgbm_file)

