#!/usr/bin/env python
# coding: utf-8

# ## <span style='color:red'>***Data Loading***</span>

# In[1]:


import pandas as pd

authentic=pd.read_csv(r'D:\Download\Authentic-48K.csv')
fake=pd.read_csv(r'D:\Download\Fake-1K.csv')

df=pd.concat([authentic, fake], axis=0)
df.shape


# In[2]:


df.tail()


# In[3]:


df.reset_index(inplace=True)
df.tail()


# In[4]:


df.drop('index', axis=1, inplace=True)


# In[5]:


import numpy as np
df['target']=np.where(df['label']==0.0, 'Fake', 'Real')
df.tail()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


sns.set(style='darkgrid', rc={'figure.figsize':(15,10)})


# In[8]:


df.isnull().sum()


# #### <span style='color:green'>***Converting the Merged Dataset to CSV File***</span>

# In[9]:


df.to_csv(r'D:\Download\merged_data_model.csv', index=False)


# ## <span style='color:red'>***Total Fake and Real News***</span>

# In[10]:


fig=px.pie(df['target'].value_counts().reset_index().rename(columns={'target': 'count'}), values='count', names='index', width=1200, height=900)

fig.update_traces(textposition='inside', textinfo='percent+label', hole=0.7, marker=dict(colors=['#90afc5','#336b87','#2a3132','#763626'], 
                                                                                           line=dict(color='white', width=2)))

fig.update_layout(annotations=[dict(text='The count of fake and real news', x=0.5, y=0.5, font_size=26, showarrow=False, 
                                    font_family='monospace', font_color='#283655')],
                  showlegend = False)
                  
fig.show()


# ## <span style='color:red'>***Oversampling Technique for Creating the Balanced Dataset***</span>

# In[11]:


df.drop('target', axis=1, inplace=True)
X=df.drop('label', axis=1)
y=df['label']


# In[12]:


from imblearn.over_sampling import RandomOverSampler


# In[13]:


ov_sam=RandomOverSampler()


# In[14]:


X, y=ov_sam.fit_resample(X, y)


# In[15]:


type(X), type(y)


# In[16]:


X.shape, y.shape


# In[17]:


#X=pd.DataFrame(X, columns=['articleID', 'domain', 'date', 'category', 'headline', 'content'])
y=pd.DataFrame(y, columns=['label'])

df=pd.concat([X, y], axis=1)
df.tail()


# ## <span style='color:red'>***Total Fake and Real News in the Balanced Dataset***</span>

# In[18]:


df['target']=np.where(df['label']==0.0, 'Fake', 'Real')
fig=px.pie(df['target'].value_counts().reset_index().rename(columns={'target': 'count'}), values='count', names='index', width=1100, height=900)

fig.update_traces(textposition='inside', textinfo='percent+label', hole=0.7, marker=dict(colors=['#90afc5','#336b87','#2a3132','#763626'], 
                                                                                           line=dict(color='white', width=2)))

fig.update_layout(annotations=[dict(text='The count of fake and real news', x=0.5, y=0.5, font_size=26, showarrow=False, 
                                    font_family='monospace', font_color='#283655')],
                  showlegend = False)
                  
fig.show()


# #### <span style='color:green'>***Converting the Balanced Dataset to CSV File***</span>

# In[19]:


df.to_csv(r'D:\Download\balance_dataset.csv', index=False)


# In[20]:


import session_info
session_info.show()

