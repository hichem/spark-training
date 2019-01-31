#!/usr/bin/env python
from pyspark import SparkContext
sc=SparkContext("local[*]",appName="app")
data = sc.textFile("test.txt")
print(data.collect())
from numpy import array
parsedData = data.map(lambda line:array([float(x) for x in line.split(' ')]))
print(parsedData.collect())

