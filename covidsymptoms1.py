#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_excel('Covidsymptoms.xlsx')


# In[3]:


dataset


# In[4]:


dataset.columns


# In[5]:


y=dataset['Infected with Covid19']


# In[6]:


X=dataset[['Dry Cough', 'High Fever', 'Sore Throat', 'Difficulty in breathing']]


# In[7]:


import seaborn as sns


# In[8]:


sns.set()


# In[9]:


dataset.isnull()


# In[10]:


sns.heatmap(dataset.isnull())


# In[11]:


sns.heatmap(dataset.corr())


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


model=LogisticRegression()


# In[16]:


model.fit(X_train,y_train)


# In[17]:


y_pred=model.predict(X_test)


# In[18]:


from sklearn.metrics import confusion_matrix


# In[19]:


confusion_matrix(y_test,y_pred)


# In[20]:


y_pred


# In[21]:


y_test.shape


# In[22]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy=accuracy_score(y_test,y_pred)


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


print(classification_report(y_test,y_pred))


# In[36]:


print(accuracy)


# In[ ]:




