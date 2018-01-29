
# coding: utf-8

# # House price prediction

# Step 1: Import panda libraries and read test and training data

# In[1]:


import pandas as pd
train_data_csv = pd.read_csv('../input/train.csv')
test_data_csv = pd.read_csv('../input/test.csv')


# tep 2: Select target attribute (Attribute that we are going to predict.) and also select attributes on which we are going to apply machine learning algorithm.

# In[2]:


target = train_data_csv.SalePrice
train_data = train_data_csv.drop(['SalePrice'],axis=1).select_dtypes(exclude = ['object'])


# Step 3: Import machine learning libraries

# In[3]:


from sklearn.preprocessing import Imputer


# In[4]:


from xgboost import XGBRegressor


# Step 4: Apply machine learning and create model

# In[5]:


my_model = XGBRegressor(n_estimators = 100,learning_rate = 0.05)
my_model.fit(train_data,target,verbose = False)


# Step 5: Match test data attributes with training data attributes.

# In[6]:


test_data = test_data_csv.select_dtypes(exclude = ['object'])


# Step 6: Apply model built in above steps to predict House price for test data

# In[7]:


predicted_value = my_model.predict(test_data)


# In[8]:


submission = pd.DataFrame({'Id':test_data.Id,'SalePrice':predicted_value})


# Step 7: Writting outcome in a csv file

# In[9]:


submission.to_csv('submission.csv',index=False)

