#!/usr/bin/env python
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc=SparkContext("local[*]",appName="app")
sqlContext = SQLContext(sc)

df_mysql = sqlContext.read.format("jdbc").option("url", "jdbc:mysql://localhost/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "customers").option("user", "root").option("cloudera", "password").load()
df_mysql.show()
