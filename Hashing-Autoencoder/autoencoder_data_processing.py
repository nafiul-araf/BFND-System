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
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
warnings.filterwarnings('ignore')


# In[ ]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_train.npy', 'rb') as f:
    X_train=np.load(f)

with open(r'/content/drive/MyDrive/AutoEncoders/X_test.npy', 'rb') as f:
    X_test=np.load(f)


# In[ ]:


encoder=load_model('/content/drive/MyDrive/AutoEncoders/encoder_eff.h5')
encoder.summary()


# In[ ]:


X_train_encode=encoder.predict(X_train)
encoder.evaluate(X_train_encode, X_train)


# In[ ]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_train_encode.npy', 'wb') as f:
    np.save(f, X_train_encode)


# In[ ]:


X_test_encode=encoder.predict(X_test)
encoder.evaluate(X_test_encode, X_test)


# In[ ]:


with open(r'/content/drive/MyDrive/AutoEncoders/X_test_encode.npy', 'wb') as f:
    np.save(f, X_test_encode)

