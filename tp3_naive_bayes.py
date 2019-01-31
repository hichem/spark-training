#!/usr/bin/env python
# coding: utf-8

# Load Naive Bayes Classifier from spark ml (not mllib) as we will use data frames

# In[3]:


from pyspark.ml.classification import NaiveBayes


# Load the data

# In[8]:


df = spark.read.load("Iris1.csv", format="csv", sep=",", inferSchema="true", header="true")
df.show()


# Load String indexing module

# In[5]:


from pyspark.ml.feature import StringIndexer


# Initialize the indexer to index the species column

# In[6]:


speciesIndexer = StringIndexer(inputCol='species', outputCol='speciesIndex')


# In[9]:


from pyspark.ml.feature import VectorAssembler


# In[11]:


vectorAssembler = VectorAssembler(inputCols=['petal_width', 'petal_length', 'sepal_width', 'sepal_length'], outputCol='features')


# In[13]:


data = vectorAssembler.transform(df)
data.show(2)


# In[16]:


index_model = speciesIndexer.fit(data)


# In[17]:


data_indexed = index_model.transform(data)
data_indexed.show(2)


# Split the data in 80% for training and 20% for testing. The second argument 0 is to indicate that the split must be equitable among classes

# In[18]:


training_data, test_data = data_indexed.randomSplit([0.8, 0.2], 0.0)


# In[19]:


training_data.count()


# In[20]:


test_data.count()


# Initailize the Naive Bayes model with the features and label columns and set its type to multinomial because there are more than 2 classes

# In[22]:


nb = NaiveBayes().setFeaturesCol('features').setLabelCol('speciesIndex').setModelType('multinomial')


# In[23]:


model = nb.fit(training_data)


# In[24]:


classifications = model.transform(test_data)


# In[26]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[27]:


evaluator = MulticlassClassificationEvaluator(labelCol='speciesIndex', predictionCol='prediction', metricName='accuracy')


# In[28]:


accuracy = evaluator.evaluate(classifications)


# In[29]:


accuracy


# In[ ]:




