#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd
import numpy as np
import re
from bnlp.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pac_model_encoder=joblib.load(r"D:\Download\pac_file_encoder.pkl")
print(dir(pac_model_encoder))


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


hashing=HashingVectorizer(n_features=5000, ngram_range=(1, 3))
X=hashing.fit_transform(corpus).toarray()
y=df['label'].values
X.shape, y.shape


# In[7]:


from tensorflow.keras.models import load_model
import tensorflow
with tensorflow.device('/CPU:0'):
    encoder=load_model('D:\Download\encoder_eff.h5')


# In[8]:


X_encode=encoder.predict(X)


# In[9]:


X


# In[10]:


X_encode


# In[11]:


pac_pred=pac_model_encoder.predict(X_encode)
pac_pred[0:5]


# In[12]:


accuracy_score(y, pac_pred)


# In[13]:


print(classification_report(y, pac_pred))


# In[14]:


sns.heatmap(confusion_matrix(y, pac_pred), annot=True, fmt='g')
plt.show()


# In[15]:


encoder.evaluate(X, X)

