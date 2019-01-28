#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy
from numpy import array
from pyspark.mllib.clustering import KMeans


# Load Data

# In[11]:


data = sc.textFile("test.txt")


# In[12]:


data.collect()


# In[13]:


parsed_data = data.map(lambda line: array([float(x) for x in line.split(' ')]))


# In[14]:


parsed_data.collect()


# In[16]:


clusters = KMeans.train(parsed_data, 2, maxIterations=10)


# In[17]:


clusters.predict(parsed_data).collect()


# In[ ]:




