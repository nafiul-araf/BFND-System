#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd
import numpy as np
import re
from bnlp.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


lr_model=joblib.load(r"D:\Download\lr_file_tfidf.pkl")
print(dir(lr_model))


# In[3]:


label_auth=pd.read_csv(r'D:\Download\LabeledAuthentic-7K.csv', nrows=3067)
label_fake=pd.read_csv(r'D:\Download\LabeledFake-1K.csv', nrows=995)
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


# In[4]:


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


# In[5]:


corpus=preprocess(df)
corpus[16:20]


# In[6]:


tfidf=TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X=tfidf.fit_transform(corpus).toarray()
y=df['label'].values
X.shape, y.shape


# In[7]:


lr_pred=lr_model.predict(X)
lr_pred


# In[8]:


accuracy_score(y, lr_pred)


# In[9]:


print(classification_report(y, lr_pred))


# In[10]:


sns.heatmap(confusion_matrix(y, lr_pred), annot=True, fmt='g')
plt.show()

