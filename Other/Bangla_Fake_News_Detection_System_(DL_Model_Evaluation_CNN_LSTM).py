#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import pandas as pd


# In[3]:


label_auth=pd.read_csv(r'/content/drive/MyDrive/CSE-400(Project Thesis)/LabeledAuthentic-7K.csv', nrows=3067)
label_auth.shape


# In[4]:


label_fake=pd.read_csv(r'/content/drive/MyDrive/CSE-400(Project Thesis)/LabeledFake-1K.csv', nrows=995)
label_fake.shape


# In[5]:


label_auth.drop(['source','relation'], axis=1, inplace=True)
label_fake.drop(['source','relation','F-type'], axis=1, inplace=True)
label_auth.shape, label_fake.shape


# In[6]:


df=pd.concat([label_auth, label_fake], axis=0)
df.shape


# In[7]:


df.reset_index(inplace=True)
df.tail()


# In[8]:


df.drop('index', axis=1, inplace=True)


# In[9]:


import numpy as np
df['label']=np.where(df['label']==0.0, 0, 1)
df.tail()


# In[10]:


df=df.sample(frac=1).reset_index(drop=True)
df.shape


# In[11]:


df.head()


# In[12]:


get_ipython().system('pip install bnlp-toolkit')


# In[13]:


import nltk
import re
from bnlp.corpus import stopwords


# In[14]:


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


# In[15]:


corpus_eval=preprocess(df)


# In[16]:


corpus_eval[5:7]


# In[17]:


y=df['label']


# In[18]:


import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# In[19]:


voc_size=150000
onehot_repr=[one_hot(words, voc_size) for words in corpus_eval] 


# In[20]:


sent_length=300
embedded_docs=pad_sequences(onehot_repr, maxlen=sent_length, truncating='post')
print(embedded_docs)


# In[21]:


len(embedded_docs)


# In[22]:


X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape, y_final.shape


# In[23]:


model=load_model(r'/content/drive/MyDrive/CSE-400(Project Thesis)/cnn_lstm.h5')


# In[24]:


model.summary()


# In[25]:


model.evaluate(X_final, y_final)


# In[26]:


prediction=model.predict(X_final)


# In[27]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[28]:


accuracy_score(y_final, prediction.round())


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_final, prediction.round()), annot=True, fmt='g')
plt.show()


# In[30]:


print(classification_report(y_final, prediction.round()))

