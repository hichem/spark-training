#!/usr/bin/env python
# coding: utf-8

# ### Load data

# In[1]:


df = spark.read.load("data/Iris1.csv", format="csv", sep=",", inferSchema="true", header="true")
df.show(2)


# ### Prepare Data

# In[9]:


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
speciesIndexer = StringIndexer(inputCol='species', outputCol='speciesIndex')
vectorAssembler = VectorAssembler(inputCols=['petal_width', 'petal_length', 'sepal_width', 'sepal_length'], outputCol='features')
data = vectorAssembler.transform(df)
data.show(2)
index_model = speciesIndexer.fit(data)
data_indexed = index_model.transform(data)
data_indexed.show(2)
training_data, test_data = data_indexed.randomSplit([0.8, 0.2], 0.0)
print("Training Data Count: %s" % training_data.count())
print("Training Data Count: %s" % test_data.count())


# ### Load Neural Network Library

# In[6]:


from pyspark.ml.classification import MultilayerPerceptronClassifier


# In[7]:


#layers list contains the following
# Number of features
# ...
# Number of neurons by layer
# ...
# Number of classes
#
# In this example we create 2 hidden layers, the first has 5 neurons, the second 4
layers = [4, 5, 4, 3]


# In[10]:


# setBlockSize: provide the size of data to be used for training at each epoch
# setSeed: provide a seed for random generation
nn = MultilayerPerceptronClassifier().setLayers(layers).setLabelCol('speciesIndex').setFeaturesCol('features').setBlockSize(training_data.count()).setSeed(1234)


# In[11]:


model = nn.fit(training_data)


# In[12]:


classifications = model.transform(test_data)


# ### Evaluate Accuracy

# In[13]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol='speciesIndex', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(classifications)
print("Accuracy: %s" % accuracy)


# ### Model with more neurons (10)

# In[18]:


layers = [4, 10, 10, 3]
nn = MultilayerPerceptronClassifier().setLayers(layers).setLabelCol('speciesIndex').setFeaturesCol('features').setBlockSize(training_data.count()).setSeed(1234)
model = nn.fit(training_data)
classifications = model.transform(test_data)
accuracy = evaluator.evaluate(classifications)
print("Accuracy: %s" % accuracy)


# ### Random Forest

# In[19]:


from pyspark.ml.classification import RandomForestClassifier


# In[20]:


rf = RandomForestClassifier().setLabelCol('speciesIndex').setFeaturesCol('features').setNumTrees(40)


# In[21]:


model = rf.fit(training_data)


# In[22]:


classifications = model.transform(test_data)
accuracy = evaluator.evaluate(classifications)
print("Accuracy: %s" % accuracy)


# In[ ]:




