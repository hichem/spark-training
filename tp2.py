#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello World!!")


# In[2]:


rdd=sc.parallelize([1,2,3,4])


# Multiply Values by 2

# In[3]:


rdd2=rdd.map(lambda x: x*2)


# In[5]:


print(rdd2)


# In[6]:


rdd2.collect()


# Filter Even Values

# In[13]:


rddf=rdd.filter(lambda x: x%2 == 0)


# In[14]:


rddf.collect()


# In[16]:


rdd3=sc.parallelize([1,4,2,2,3])


# In[17]:


rdd3.collect()


# Remove duplicate values (return a distinct list of values)

# In[21]:


rdd4=rdd3.distinct()


# In[22]:


rdd4.collect()


# In[23]:


rdd5=sc.parallelize([1,2,3])


# In[24]:


rdd5.map(lambda x: [x, x+5])


# In[25]:


rdd5.collect()


# In[27]:


rdd6.collect()


# In[31]:


rdd7=rdd5.flatMap(lambda x: [x, x+5])


# In[32]:


rdd7.collect()


# In[33]:


rdd=sc.parallelize([1,2,3])


# In[34]:


rdd.reduce(lambda a,b: a*b)


# In[35]:


rdd.take(2)


# In[36]:


rdd10 = sc.parallelize([5,3,1,2])


# In[37]:


rdd10.takeOrdered(3, lambda x: 1 * x)


# Multiply with -1 invert the order in which elements are sorted

# In[39]:


rdd10.takeOrdered(3, lambda x: -1 * x)


# In[ ]:




