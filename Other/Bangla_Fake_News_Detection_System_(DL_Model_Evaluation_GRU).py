#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[2]:


get_ipython().system('pip install bnlp-toolkit')


# In[3]:


import pandas as pd
import numpy as np
import re
from bnlp.corpus import stopwords
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


model_GRU=load_model(r'/content/gdrive/MyDrive/CSE-400(Project Thesis)/model_GRU.h5')
print(dir(model_GRU))


# In[5]:


label_auth=pd.read_csv(r'/content/gdrive/MyDrive/CSE-400(Project Thesis)/LabeledAuthentic-7K.csv', nrows=3067)
label_fake=pd.read_csv(r'/content/gdrive/MyDrive/CSE-400(Project Thesis)/LabeledFake-1K.csv', nrows=995)
label_auth.drop(['source','relation'], axis=1, inplace=True)
label_fake.drop(['source','relation','F-type'], axis=1, inplace=True)
df=pd.concat([label_auth, label_fake], axis=0)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
df['label']=np.where(df['label']==0.0, 0, 1)
df.tail()
df=df.sample(frac=1).reset_index(drop=True)
print(df.shape)
df.head()


# In[6]:


def preprocess(data):
    """This function is for preprocessing of the news contents. It removes punctuations, English characters and both of Bangla
    and English numerals. It tokenizes all the words and also removes stopwords.
    
    Args: 
        The Entire Dataframe
    Returns:
        Preprocessed news corpuses in a list of lists
    """
    corpus=[]
    for i in range(0, len(data)):
        x=re.sub('[^\u0980-\u09FF]',' ',data['content'][i])
        x=re.sub('[a-zA-Z0-9]+', ' ', x)
        x=re.sub('[০১২৩৪৫৬৭৮৯]+', ' ', x)
        x=x.split()
        x=[w for w in x if w not in set(stopwords)]
        x=' '.join(x)
        corpus.append(x)
    
    return corpus


# In[7]:


corpus=preprocess(df)
corpus[16:20]


# In[8]:


voc_size=150000
onehot_repr=[one_hot(words, voc_size) for words in corpus] 


# In[9]:


sent_length=300
embedded_docs=pad_sequences(onehot_repr, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[10]:


len(embedded_docs)


# In[11]:


y=df['label']

X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape, y_final.shape


# In[12]:


model_GRU.summary()


# In[13]:


prediction_GRU=model_GRU.predict(X_final)
prediction_GRU


# In[14]:


accuracy_score(y, prediction_GRU.round())


# In[15]:


print(classification_report(y, prediction_GRU.round()))


# In[16]:


sns.heatmap(confusion_matrix(y, prediction_GRU.round()), annot=True, fmt='g')
plt.show()

