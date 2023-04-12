#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


# In[6]:


car_df=pd.read_csv('D:/Python/car data.csv')


# In[7]:


car_df.head()


# In[8]:


car_df.shape


# In[9]:


car_df.info()


# In[10]:


car_df.describe()


# In[12]:


car_df.Year.value_counts().plot(kind='bar')


# In[14]:


car_df.groupby('Year')['Selling_Price'].mean().plot(kind='bar')


# In[15]:


car_df.Selling_Price.hist()


# In[16]:


print(car_df.Present_Price.corr(car_df.Selling_Price))
plt.scatter(x='Present_Price',y='Selling_Price',data=car_df)


# In[17]:


car_df[(car_df.Present_Price.values>60)]


# In[18]:


car_df[(car_df.Selling_Price.values>25)]


# In[21]:


car_df=car_df[(car_df.Selling_Price.values<26)]


# In[22]:


car_df.Present_Price.hist()


# In[23]:


print(car_df.Present_Price.corr(car_df.Selling_Price))
plt.scatter(x='Present_Price',y='Selling_Price',data=car_df)


# In[24]:


car_df.Kms_Driven.hist()


# In[25]:


print(car_df.Kms_Driven.corr(car_df.Selling_Price))
plt.scatter(x='Kms_Driven',y='Selling_Price',data=car_df)


# In[26]:



car_df=car_df[car_df.Kms_Driven.values<400000]


# In[27]:


print(car_df.Kms_Driven.corr(car_df.Selling_Price))
plt.scatter(x='Kms_Driven',y='Selling_Price',data=car_df)


# In[28]:


car_df.Owner.hist()


# In[29]:


print(car_df.Owner.corr(car_df.Selling_Price))
plt.scatter(x='Owner',y='Selling_Price',data=car_df)


# In[30]:


car_df_New=car_df.drop(['Owner','Car_Name'],axis=1)
#car_df_New=car_df.drop(['Car_Name'],axis=1)
car_df_New.head()


# In[31]:


car_df_New.Fuel_Type.value_counts().plot(kind='bar')


# In[32]:


car_df_New.groupby('Fuel_Type')['Selling_Price'].median().plot(kind='bar')


# In[33]:


car_df_New.Seller_Type.value_counts().plot(kind='bar')


# In[34]:


car_df_New.groupby('Seller_Type')['Selling_Price'].median().plot(kind='bar')


# In[35]:


car_df_New.Transmission.value_counts().plot(kind='bar')


# In[36]:


car_df_New.Transmission.value_counts().plot(kind='bar')


# In[37]:


for feature in car_df_New.columns:
    le = LabelEncoder()
    le.fit(car_df_New[feature])
    car_df_New[feature] = le.transform(car_df_New[feature])
X = car_df_New.drop(['Selling_Price'],axis=1)
Y = car_df_New['Selling_Price']


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)


# In[39]:


lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_Train_pred_LR = lr_model.predict(x_train)
R2_LR = metrics.r2_score(y_train,y_Train_pred_LR)
print('MAE = ', R2_LR)


# In[40]:


y_test_pred_LR = lr_model.predict(x_test)
R2_Test_LR = metrics.r2_score(y_test,y_test_pred_LR)
print('MAE = ', R2_Test_LR)


# In[41]:


plt.hist(y_test_pred_LR, alpha=0.5, label='Predicted Price')
plt.hist(y_test, alpha=0.5, label='Actual Price')
plt.legend(loc='upper right')
plt.title(" Actual Prices vs Predicted Prices LinearRegression")
plt.show()


# In[42]:


plt.scatter(y_test, y_test_pred_LR)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices LinearRegression")
plt.show()


# In[43]:


Lasso_model = Lasso()
Lasso_model.fit(x_train, y_train)
y_Train_pred = Lasso_model.predict(x_train)
R = metrics.r2_score(y_train,y_Train_pred)
print('MAE = ', R)


# In[44]:


y_test_pred = Lasso_model.predict(x_test)
R2_Test_Lasso = metrics.r2_score(y_test,y_test_pred)
print('MAE = ', R2_Test_Lasso)


# In[45]:


plt.hist(y_test_pred, alpha=0.5, label='Predicted Price')
plt.hist(y_test, alpha=0.5, label='Actual Price')
plt.legend(loc='upper right')
plt.title(" Actual Prices vs Predicted Prices Lasso_model")
plt.show()


# In[46]:


plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices Lasso_model")
plt.show()


# In[47]:


print('LinearRegression r2 Test score :',R2_Test_LR)
print('Lasso r2 Test score :',R2_Test_Lasso)


# In[ ]:




