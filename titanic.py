#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the packages
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings("ignore")


# # Loading the dataset

# In[2]:


data = pd.read_csv("titanic.csv")
data.head()


# # Performing EDA

# In[3]:


data.columns


# In[4]:


data.isnull().sum()


# In[5]:


data.drop(columns=['Cabin'],inplace=True)


# In[6]:


data.info()


# In[7]:


data['Survived'].value_counts()


# In[8]:


data['Pclass'].value_counts(ascending =True)


# In[9]:


data['Sex'].value_counts()


# In[10]:


data['Embarked'].value_counts()


# In[11]:


data['Fare'].value_counts()


# In[12]:


data.isnull().sum()


# In[13]:


import matplotlib


# In[14]:


from matplotlib import pyplot as plt


# In[77]:


a = [1,5,8,9,25]
plt.plot(a,'.',color='cyan')
#plt.show()


# In[76]:


x = [1,5,8,9,2]
y = [2,6,8,2,5]
plt.plot(x,y,'.',color='blue')
#plt.show()


# In[17]:


from matplotlib import style


# In[82]:


#to get the list of availble styles
#print(plt.style.available)


# In[81]:


style.use('seaborn-dark-palette')
plt.plot(x,y,color='cyan')
#plt.show()


# In[72]:


#we wiill get the unique value count
data['Pclass'].value_counts().plot(kind="bar")
#plt.show


# In[71]:


data['Pclass'].value_counts().plot(kind="barh") 
#plt.show()


# In[70]:


data['Embarked'].value_counts().plot(kind='bar')
#plt.show()


# In[23]:


lab = data["Survived"].value_counts().keys().tolist()
val = data["Survived"].value_counts().values.tolist()
val


# In[69]:


title = "Survived vs Not-Survived"
plt.pie(val,labels=lab,colors=['cyan','magenta'],autopct = '%1.2f%%') 
plt.axis('equal')
plt.title(title) 
plt.tight_layout()
#plt.show()


# In[25]:


data['Age'].mean()


# In[68]:


plt.hist(data['Age'])
#plt.show()


# In[67]:


counts, edges, bars = plt.hist(data['Age'], color='blue')
plt.bar_label (bars)
#plt.show()


# In[66]:


plt.scatter (x=data [ "Age"], y=data['Fare']) 
plt.title("Age vs Fare") 
#plt.show()


# In[29]:


import seaborn as sns


# In[65]:


sns.boxplot(x="Embarked", y="Age", data=data) #hox plot -->Quartile Distribution
#plt.show()


# In[31]:


data.loc[data['Embarked'].isnull()]


# In[32]:


data['Embarked'].value_counts()


# In[33]:


data.info()


# In[34]:


data['Sex'].value_counts()


# In[35]:


sex = pd.get_dummies(data['Sex'])
sex


# In[36]:


#from plotly.offline import iplot
#import plotly as py 
import cufflinks as cf 
py.offline.init_notebook_mode(connected=True)
#from plotly.figure_factory import create_table 
cf.go_offline() 
#import plotly.io as pio


# In[37]:


data['Age'].mean()


# In[38]:


#filling the age column n-->as majority age group is from 20-30 we fill with mean
data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[39]:


data['Age'].isnull().sum()


# In[40]:


data.isnull().sum()


# In[41]:


data.loc[data['Embarked'].isnull()]


# In[42]:


data['Embarked'].value_counts()


# In[43]:


data['Embarked'].fillna('S',inplace=True)


# In[44]:


data.info()


# In[45]:


sex = pd.get_dummies(data['Sex'])
sex


# In[46]:


pclass= pd.get_dummies(data['Pclass'])
pclass


# In[47]:


embarked = pd.get_dummies(data['Embarked'],drop_first=True)
embarked


# In[48]:


final_data = pd.concat([data,sex,pclass,embarked],axis='columns')
final_data


# In[49]:


#FInally we will drop unnecessary columns
final_data.drop(columns=['PassengerId', 'Sex', 'Name',
                         'Pclass', 
                         'Ticket', 
                         'Embarked'], inplace=True)


# In[50]:


final_data


# In[51]:


final_data.tail()


# In[52]:


final_data.head()


# # Building The Model

# In[53]:


#Training and Testing data
X = final_data.drop('Survived', axis=1)
y = final_data["Survived"]


# In[54]:


X.columns


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


from sklearn.linear_model import LogisticRegression


# In[57]:


#Splitting the data into training and testing set
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=1)                                                


# In[58]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train) #Estimators


# In[59]:


predictions = logmodel.predict(X_test)


# In[60]:


#calaculation of metrics
from sklearn.metrics import plot_confusion_matrix, accuracy_score, confusion_matrix, classification_report, roc_curve 
print (accuracy_score (y_test, predictions)*100)


# In[80]:


#print(confusion_matrix(y_test,predictions))


# In[79]:


#print(classification_report(y_test,predictions))


# In[78]:


#plot_confusion_matrix(logmodel,X_test,y_test,values_format='d')
#plt.show()


# In[64]:


#logmodel.predict([[35,0,0,8.05,0,1,0,0,1,0,1]])


def survive(arr):
    predictions = logmodel.predict(arr)
    return predictions



