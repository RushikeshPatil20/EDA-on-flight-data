#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data = pd.read_excel('Data_Train.xlsx')


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data.shape


# In[7]:


train_data.dropna(inplace=True)


# ### EDA

# In[8]:


train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format="%d/%m/%Y").dt.day
train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format="%d/%m/%Y").dt.month


# In[9]:


train_data.head()


# In[10]:


train_data.drop('Date_of_Journey', axis=1, inplace=True)


# - Convert Dep_time in hours and minutes

# In[11]:


train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute


# In[12]:


train_data.drop('Dep_Time', axis=1, inplace=True)


# In[13]:


train_data.head()


# - Coverting Arrival_Time in hours and minutes

# In[14]:


train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute


# In[15]:


train_data.drop('Arrival_Time', axis=1, inplace=True)


# In[16]:


train_data.head()


# - Converting Duration to hours and minutes

# In[17]:


duration = list(train_data['Duration'])


# In[18]:


a = ['20h0m']
int(a[0].split(sep='m')[0].split('h')[-1])


# In[19]:


for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + '0m'
        else:
            duration[i] = '0h' + duration[i]
            
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split('h')[-1]))


# In[20]:


train_data['Duration_hours'] = duration_hours
train_data['Duration_mins'] = duration_mins


# In[21]:


train_data.head()


# In[22]:


train_data.drop('Duration', axis=1, inplace=True)


# In[23]:


train_data.head()


# In[24]:


sns.catplot(x='Airline', y='Price', data=train_data.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3)
plt.show()


# - Categorical data 

# In[25]:


Airline = train_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[26]:


sns.catplot(x='Source', y='Price', data=train_data.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3)
plt.show()


# - Route and Total_stops contains same information so droping Route and Additional_info

# In[27]:


train_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)


# In[28]:


train_data.head()


# - Source and Destination 

# In[29]:


Source = train_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)


# In[30]:


Source.head()


# In[31]:


Destination = train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)


# In[32]:


Destination.head()


# In[33]:


train_data.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)


# In[34]:


train_data.head()


# In[35]:


train_data = pd.concat([train_data,Airline,Source,Destination], axis=1)


# In[36]:


train_data.head()


# In[37]:


train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[38]:


train_data.head()


# ### Test data

# In[39]:


test_data = pd.read_excel('Test_set.xlsx')


# In[40]:


test_data.head()


# In[41]:


test_data.info()


# In[42]:


test_data.isnull().sum()


# - converting Date_of_journey to day and month

# In[43]:


test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey']).dt.day


# In[44]:


test_data['Journey_month'] = pd.to_datetime(test_data['Date_of_Journey']).dt.month


# In[45]:


test_data.head()


# In[46]:


test_data.drop('Date_of_Journey', axis=1, inplace=True)


# In[47]:


test_data.head()


# In[48]:


test_data['Dep_hours'] = pd.to_datetime(test_data['Dep_Time']).dt.hour


# In[49]:


test_data['Dep_min'] = pd.to_datetime(test_data['Dep_Time']).dt.minute


# In[50]:


test_data.drop('Dep_Time', axis=1, inplace=True)


# In[51]:


test_data.head()


# In[52]:


test_data['Arrival_hours'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
test_data['Arrival_mins'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute


# In[53]:


test_data.drop('Arrival_Time', axis=1, inplace=True)


# In[54]:


test_data.head()


# In[55]:


duration = list(test_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + '0m'
        else:
            duration[i] = '0h' + duration[i]
            
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split('h')[-1]))


# In[56]:


test_data['Duration_hours'] = duration_hours
test_data['Duration_mins'] = duration_mins


# In[57]:


test_data.head()


# In[58]:


test_data.drop('Duration', axis=1, inplace=True)


# In[59]:


test_data['Additional_Info'].value_counts()


# In[60]:


test_data.drop(['Additional_Info','Route'], axis=1, inplace=True)


# In[61]:


test_data.head()


# In[62]:


Airline = test_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[63]:


source = test_data[['Source']]
source = pd.get_dummies(source, drop_first=True)
source.head()


# In[64]:


destination = test_data[['Destination']]
destination = pd.get_dummies(destination, drop_first=True)
destination.head()


# In[65]:


test_data.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)


# In[66]:


test_data = pd.concat([test_data, Airline, source, destination], axis=1)


# In[67]:


test_data.head()


# In[68]:


test_data['Total_Stops'].value_counts()


# In[69]:


test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
test_data.head()


# ### Feature Selection

# In[70]:


train_data.shape


# In[71]:


train_data.columns


# In[72]:


X = train_data[['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[73]:


y = train_data['Price']


# In[74]:


plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(), annot=True)
plt.show()


# In[75]:


from sklearn.ensemble import ExtraTreesRegressor


# In[76]:


selction = ExtraTreesRegressor()
selction.fit(X, y)


# In[77]:


print(selction.feature_importances_)


# In[78]:


plt.figure(figsize=(12,8))
feature_importances = pd.Series(selction.feature_importances_, index=X.columns)
feature_importances.nlargest(20).plot(kind='barh')
plt.show()


# ### Fitting model using Random Forest

# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[81]:


from sklearn.ensemble import RandomForestRegressor


# In[82]:


rf_Reg = RandomForestRegressor()
rf_Reg.fit(X_train, y_train)


# In[83]:


y_pred = rf_Reg.predict(X_test)


# In[84]:


rf_Reg.score(X_train, y_train)


# In[85]:


rf_Reg.score(X_test, y_test)


# In[86]:


sns.distplot(y_test-y_pred)
plt.show()


# In[87]:


sns.scatterplot(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[ ]:




