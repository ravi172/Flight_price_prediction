#!/usr/bin/env python
# coding: utf-8

# #       Flight price prediction using Random Forest Regressor 

# # By- Ravi Kumar Verma & Devendra Yadav

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_df=pd.read_excel('G:/data/flight/Data_Train.xlsx')
test_df=pd.read_excel('G:/data/flight/Test_set.xlsx')


# In[4]:


train_df.head()


# In[5]:


test_df.head()


# In[6]:


big_df=train_df.append(test_df,sort=False)


# In[7]:


big_df.tail()


# In[8]:


big_df.dtypes


# In[9]:


big_df['Date']=big_df['Date_of_Journey'].str.split('/').str[0]
big_df['Month']=big_df['Date_of_Journey'].str.split('/').str[1]
big_df['Year']=big_df['Date_of_Journey'].str.split('/').str[2]


# In[10]:


big_df.head()


# In[11]:


big_df.dtypes


# In[12]:


big_df['Date']=big_df['Date'].astype(int)
big_df['Month']=big_df['Month'].astype(int)
big_df['Year']=big_df['Year'].astype(int)


# In[13]:


big_df.dtypes


# In[14]:


big_df=big_df.drop(['Date_of_Journey'],axis=1)


# In[15]:


big_df.head()


# In[16]:


big_df['Arrival_Time']=big_df['Arrival_Time'].str.split(' ').str[0]


# In[17]:


big_df.head()


# In[18]:


big_df[big_df['Total_Stops'].isnull()]


# In[19]:


big_df['Total_Stops']=big_df['Total_Stops'].fillna('1 stop')


# In[20]:


big_df['Total_Stops']=big_df['Total_Stops'].replace('non-stop','0 stop')


# In[21]:


big_df.head()


# In[22]:


big_df['Stop'] = big_df['Total_Stops'].str.split(' ').str[0]
big_df.head()


# In[23]:



big_df.dtypes


# In[24]:


big_df['Stop']=big_df['Stop'].astype(int)
big_df=big_df.drop(['Total_Stops'],axis=1)


# In[25]:


big_df.head()


# In[26]:


big_df['Arrival_Hour'] = big_df['Arrival_Time'] .str.split(':').str[0]
big_df['Arrival_Minute'] = big_df['Arrival_Time'] .str.split(':').str[1]


# In[27]:


big_df['Arrival_Hour']=big_df['Arrival_Hour'].astype(int)
big_df['Arrival_Minute']=big_df['Arrival_Minute'].astype(int)
big_df=big_df.drop(['Arrival_Time'],axis=1)


# In[28]:


big_df.head()


# In[29]:


big_df['Departure_Hour'] = big_df['Dep_Time'] .str.split(':').str[0]
big_df['Departure_Minute'] = big_df['Dep_Time'] .str.split(':').str[1]


# In[30]:


big_df['Departure_Hour']=big_df['Departure_Hour'].astype(int)
big_df['Departure_Minute']=big_df['Departure_Minute'].astype(int)
big_df=big_df.drop(['Dep_Time'],axis=1)


# In[31]:


big_df.head()


# In[32]:


big_df['Route_1']=big_df['Route'].str.split('→ ').str[0]
big_df['Route_2']=big_df['Route'].str.split('→ ').str[1]
big_df['Route_3']=big_df['Route'].str.split('→ ').str[2]
big_df['Route_4']=big_df['Route'].str.split('→ ').str[3]
big_df['Route_5']=big_df['Route'].str.split('→ ').str[4]


# In[33]:


big_df.head()


# In[34]:


big_df['Price'].fillna((big_df['Price'].mean()),inplace=True)


# In[35]:


big_df['Route_1'].fillna("None",inplace=True)
big_df['Route_2'].fillna("None",inplace=True)
big_df['Route_3'].fillna("None",inplace=True)
big_df['Route_4'].fillna("None",inplace=True)
big_df['Route_5'].fillna("None",inplace=True)


# In[36]:


big_df.head()


# In[37]:


big_df=big_df.drop(['Route'],axis=1)
big_df=big_df.drop(['Duration'],axis=1)


# In[38]:


big_df.head()


# In[39]:


big_df.isnull().sum()


# In[40]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
big_df["Airline"]=encoder.fit_transform(big_df['Airline'])
big_df["Source"]=encoder.fit_transform(big_df['Source'])
big_df["Destination"]=encoder.fit_transform(big_df['Destination'])
big_df["Additional_Info"]=encoder.fit_transform(big_df['Additional_Info'])
big_df["Route_1"]=encoder.fit_transform(big_df['Route_1'])
big_df["Route_2"]=encoder.fit_transform(big_df['Route_2'])
big_df["Route_3"]=encoder.fit_transform(big_df['Route_3'])
big_df["Route_4"]=encoder.fit_transform(big_df['Route_4'])
big_df["Route_5"]=encoder.fit_transform(big_df['Route_5'])


# In[41]:


big_df.head()


# In[42]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[43]:


df_train=big_df[0:10683]
df_test=big_df[10683:]


# In[44]:


X=df_train.drop(['Price'],axis=1)
y=df_train.Price


# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[46]:


model=SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[47]:


model.fit(X_train,y_train)


# In[48]:


model.get_support()


# In[49]:


selected_features=X_train.columns[(model.get_support())]


# In[50]:


selected_features


# In[51]:


X_train=X_train.drop(['Year'],axis=1)


# In[52]:


X_test=X_test.drop(['Year'],axis=1)


# In[54]:


# Random Forest Regressor

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[55]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[56]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[57]:


# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[58]:


rf_random.fit(X_train,y_train)


# In[59]:


y_pred=rf_random.predict(X_test)


# In[67]:


sns.distplot(y_test-y_pred)


# In[65]:


plt.scatter(y_test,y_pred)


# In[75]:


# Performance metrics
errors = abs(y_pred - y_test)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'rupees.')


# Calculate mean absolute percentage error (MAPE)
mape = np.mean(100 * (errors / y_test))
# Calculate and display accuracy
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:




