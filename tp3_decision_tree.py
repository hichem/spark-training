#!/usr/bin/env python
# coding: utf-8

# In[4]:


data = sc.textFile("iris num.csv")


# Display 2 elements to make sure data is loaded

# In[5]:


data.take(2)


# In[6]:


from numpy import array


# In[7]:


pdata = data.map(lambda line : array([float(x) for x in line.split(",")]))


# In[8]:


pdata.take(2)


# Prepare the data for spark mllib algorithms. Those algorithms require the data to be labeled input / output by encapsulating it in LabeledPoint objects

# In[9]:


from pyspark.mllib.regression import LabeledPoint


# In[10]:


def parse(l):
    return LabeledPoint(l[4], l[0:4])


# In[11]:


fdata = pdata.map(lambda l: parse(l))


# In[12]:


fdata.take(2)


# Divide the data into training and test

# In[ ]:





# In[53]:


(trainingData, testData) = fdata.randomSplit([0.8, 0.2])


# Use the decision tree classifier to train the model

# In[54]:


from pyspark.mllib.tree import DecisionTree


# In[55]:


model = DecisionTree.trainClassifier(trainingData, numClasses=3, categoricalFeaturesInfo={})


# In[56]:


predictions = model.predict(testData.map(lambda row: row.features))


# Create Confusion Matrix to evaluate the accuracy of the model

# We create a matrix containing the test labels as a first column (real values) and predicted values as second column

# In[57]:


predictionsAndLabels = testData.map(lambda labeledpoint: labeledpoint.label).zip(predictions)


# In[58]:


predictionsAndLabels.collect()


# ## Evaluate the accuracy of the model

# In[59]:


from pyspark.mllib.evaluation import MulticlassMetrics


# In[60]:


metrics = MulticlassMetrics(predictionsAndLabels)


# ### Display Model Accuracy

# In[61]:


accuracy = metrics.accuracy


# In[62]:


print(accuracy)


# In[63]:


confusionMatrix = metrics.confusionMatrix()


# In[64]:


confusionMatrix.toArray()


# ### Display Model Precision

# In[65]:


print("Recall for Class #1", metrics.recall(0))
print("Recall for Class #2", metrics.recall(1))
print("Recall for Class #3", metrics.recall(2))


# In[66]:


print("Precision for Class #1", metrics.precision(0))
print("Precision for Class #2", metrics.precision(1))
print("Precision for Class #3", metrics.precision(2))


# In[71]:


metrics.precision()


# In[72]:


metrics.recall()


# In[73]:


metrics.fMeasure()


# ## Use Spark Dataframe

# In[68]:


df = spark.read.load("Iris1.csv", format="csv", sep=",", inferSchema="true", header="true")


# In[69]:


df.show()


# In[70]:


df.show(2)


# ### Use Spark sqlContext to load the data as a data frame

# In[74]:


df1 = sqlContext.read.format("csv").options(header='true', inferSchema='true').load('Iris1.csv')


# In[75]:


df1.show()


# Print row count

# In[77]:


print('Row count: %s' % df.count())


# Filter the data based on a condition

# In[78]:


df.filter(df['petal_length'] > 6).show()


# In[81]:


df.groupBy(df['species']).count().show()


# Print the first 10 elements

# In[85]:


df.head(10)


# ### Execute SQL Query

# In[86]:


df.registerTempTable('mytable')


# In[87]:


distinct_classes = sqlContext.sql("select distinct species from mytable")
distinct_classes.show()


# In[ ]:




