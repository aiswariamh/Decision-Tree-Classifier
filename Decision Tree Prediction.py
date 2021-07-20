#!/usr/bin/env python
# coding: utf-8

# ## Prediction Using Decision Tree Classifier
# By Aiswaria Mohan

# In[1]:


# Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the csv file


data=pd.read_csv('iris.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data['Species'].unique()


# There are 3 classes of Species.

# In[5]:


# Checking for any missing values

data.isnull().sum()


# There are no missing values in the data.

# In[6]:


# Split data into independent and dependent variables

y=data['Species']
x=data.drop(['Id','Species'],axis=1)


# In[7]:


# Import and Fit Decision Tree Classifier on data

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x,y)


# In[8]:


# Import libraries for visualizing the image

from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[9]:


# To Visualize the decision tree and its classification

dt_data = StringIO()

export_graphviz(dt,out_file=dt_data, feature_names=x.columns,class_names = data['Species'].unique(),filled=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dt_data.getvalue())  

Image(graph.create_png())


# The figure above illustrates a Decision Tree and it shows the sequence of steps in which the output class for the new set of data will be determined. 

# In[ ]:




