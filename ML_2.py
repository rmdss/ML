#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as pt


# In[3]:


import pandas as pd


# In[4]:


mydataset=pd.read_csv(r'E:\MSC\Sem_2\Machine Learning\kaggle\train.csv')


# In[5]:


import sklearn.metrics as sm


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


columns_feature = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare','meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
for each in columns_feature:
    mydataset[each]= mydataset[each].fillna(mydataset[each].mean())


# In[12]:


mydataset = mydataset[['tripid', 'additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare', 'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare',
       'label']]


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


label_encoder_no_of_class = LabelEncoder()


# In[15]:


label_encoder_no_of_class  =label_encoder_no_of_class.fit(mydataset['label'])


# In[16]:


mydataset["label"] = label_encoder_no_of_class.transform(mydataset["label"])


# In[17]:


from sklearn.naive_bayes import GaussianNB


# In[18]:


gnb = GaussianNB()


# In[21]:


_features = mydataset.loc[:,'additional_fare':'fare']


# In[22]:


_labels = mydataset.loc[:,'label']


# In[24]:


model = gnb.fit(_features, _labels)


# In[25]:


test_data=pd.read_csv(r'E:\MSC\Sem_2\Machine Learning\kaggle\test.csv')


# In[26]:


test_data = test_data[['tripid', 'additional_fare', 'duration', 'meter_waiting','meter_waiting_fare',
                       'meter_waiting_till_pickup','pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']]


# In[27]:


test = test_data.loc[:,'additional_fare':'fare']


# In[33]:


prediction = gnb.predict(test)


# In[34]:


prediction_original = list(label_encoder_no_of_class.inverse_transform(prediction))


# In[35]:


prediction_result = pd.read_csv(r'E:\MSC\Sem_2\Machine Learning\kaggle\sample_submission.csv')


# In[36]:


prediction_result['prediction'] = prediction_original


# In[37]:


prediction_result = prediction_result.replace({'prediction': {'correct': 1,'incorrect': 0}})


# In[38]:


prediction_result.to_csv('Prediction_2.csv', index=False)


# In[41]:


prediction_result.to_excel(r'E:\MSC\Sem_2\Machine Learning\kaggle\Pred2.xlsx')


# In[ ]:




