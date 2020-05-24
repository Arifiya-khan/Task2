#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd


# In[102]:


dataset=pd.read_excel('Covidsymptoms.xlsx')


# In[103]:


dataset


# In[104]:


dataset.columns


# In[105]:


y=dataset['Infected with Covid19']


# In[106]:


X=dataset[['Dry Cough', 'High Fever', 'Sore Throat', 'Difficulty in breathing']]


# In[107]:


import seaborn as sns


# In[108]:


sns.set()


# In[109]:


dataset.isnull()


# In[110]:


sns.heatmap(dataset.isnull())


# In[111]:


sns.heatmap(dataset.corr())


# In[112]:


from sklearn.model_selection import train_test_split


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[114]:


from sklearn.linear_model import LogisticRegression


# In[115]:


model=LogisticRegression()


# In[116]:


model.fit(X_train,y_train)


# In[117]:


y_pred=model.predict(X_test)


# In[118]:


from sklearn.metrics import confusion_matrix


# In[119]:


confusion_matrix(y_test,y_pred)


# In[120]:


y_pred


# In[121]:


y_test.shape


# In[122]:


from sklearn.metrics import accuracy_score


# In[123]:


accuracy_score(y_test,y_pred)


# In[124]:


from sklearn.metrics import classification_report


# In[125]:


print(classification_report(y_test,y_pred))


# In[ ]:




