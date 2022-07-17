#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(100)


# In[2]:


df=pd.read_csv(r'D:\Download\balance_dataset.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df=df.sample(frac=1).reset_index(drop=True)
df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


remove_rows=12356
drop_indices=np.random.choice(df.index, remove_rows, replace=False)
df=df.drop(drop_indices)
df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df=df.sample(frac=1).reset_index(drop=True)
df.head()


# In[12]:


df.tail()


# In[13]:


df.shape


# In[14]:


import nltk
import re
from bnltk.stemmer import BanglaStemmer
from bnlp.corpus import stopwords


# In[15]:


stem=BanglaStemmer()


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
        x=[stem.stem(w) for w in x if w not in set(stopwords)]
        x=' '.join(x)
        corpus.append(x)
    
    return corpus


# In[16]:


corpus=preprocess(df)


# In[17]:


corpus[10:15]


# In[18]:


type(corpus)


# In[19]:


import pickle

with open(r'D:\Download\corpus_autoencoder.txt', 'wb') as filehandle:
    pickle.dump(corpus, filehandle)


# In[20]:


df.to_csv(r'D:\Download\final_data_autoencoder.csv', index=False)

