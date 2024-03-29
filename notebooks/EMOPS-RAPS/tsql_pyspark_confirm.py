# Databricks notebook source
# DBTITLE 1,Import Modules
from operator import add
from functools import reduce
from itertools import chain
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.mllib.stat import Statistics
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Connect to Microsoft SQL Database
from pyspark import SparkContext, SparkConf, SQLContext

appName = "PySpark SQL Server Example - via JDBC"
master = "local"
conf = SparkConf() \
    .setAppName(appName) \
    .setMaster(master) \
    .set("spark.driver.extraClassPath","sqljdbc_7.2/enu/mssql-jdbc-7.2.1.jre8.jar")
sc = SparkContext.getOrCreate(conf=conf)
sqlContext = SQLContext(sc)
spark = sqlContext.sparkSession

jdbcDF = spark.read.format("jdbc") \
    .option("url", f"jdbc:sqlserver://localhost:1433;databaseName={database}") \
    .option("dbtable", "cats.vw_context_trend_alert_v3_step11") \
    .option("user", user) \
    .option("password", password) \
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
    .load()

microsoftDF.show()

# COMMAND ----------

# DBTITLE 1,Import Preprocessed Data
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

myPySpark = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/FileStore/tables/tmp/gdelt/preprocessed_may2021.csv")
myPySpark = myPySpark.withColumn('EventDate', F.to_date(F.col('EventTimeDate'),"yyyy-MM-dd"))
myPySpark = myPySpark.withColumn('EventRootCode', F.col('EventRootCodeString'))
print((myPySpark.count(), len(myPySpark.columns)))
myPySpark.limit(5).toPandas()

# COMMAND ----------

# select only conflict events 
conflict_codes = ['Investigate','Demand','Disapprove','Reject','Threaten','Reduce Relations',
                  'Exhibit Force Posture','Protest','Coerce','Assault','Fight',
                  'Use of Unconventional Mass Violence']
myPySpark = myPySpark.filter( (F.col('EventRootCode').isin(conflict_codes)) & F.col('EventDate').between(F.lit('2021-01-01'), F.lit('2021-02-13')))

# COMMAND ----------

# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = myPySpark.groupBy('ActionGeo_FullName','EventDate','EventRootCode') \
                         .agg(F.avg('Confidence').alias('avgConfidence'),
                              F.collect_list('GoldsteinScale').alias('GoldsteinList'),
                              F.collect_list('MentionDocTone').alias('ToneList'),
                              F.sum('nArticles').alias('nArticles')) \
                          .withColumn('GoldsteinReportValue', median_udf('GoldsteinList')) \
                          .withColumn('ToneReportValue', median_udf('ToneList')) \
                          .drop('GoldsteinList','ToneList')

# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName','EventDate').orderBy('EventDate')

# get daily distribution of articles for each Event Code string within Window
myPySparkP = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
print((myPySparkP.count(), len(myPySparkP.columns)))
myPySparkP.limit(2).toPandas()

# COMMAND ----------

# Only select data for Jan through Feb 2021  
selectData = myPySparkP.select('ActionGeo_FullName','EventDate','EventRootCode','EventReportValue','GoldsteinReportValue','ToneReportValue','nArticles')
selectData.limit(10).toPandas()

# COMMAND ----------

microsoftDF = selectData

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Compare PySpark Output with Microsoft SQL

# COMMAND ----------

selectData = selectData.withColumnRenamed('ActionGeo_FullName', 'Countries') \
                       .withColumnRenamed('EventReportValue', 'p_EventReportValue') \
                       .withColumnRenamed('GoldsteinReportValue', 'p_GoldsteinReportValue') \
                       .withColumnRenamed('ToneReportValue', 'p_ToneReportValue') \
                        .withColumnRenamed('nArticles', 'p_nArticles')

microsoftDF= microsoftDF.withColumnRenamed('ActionGeo_FullName', 'Countries') \
                        .withColumnRenamed('EventReportValue', 'm_EventReportValue') \
                        .withColumnRenamed('GoldsteinReportValue', 'm_GoldsteinReportValue') \
                        .withColumnRenamed('ToneReportValue', 'm_ToneReportValue') \
                        .withColumnRenamed('nArticles', 'm_nArticles')

# COMMAND ----------

# DBTITLE 1,Merge on Country, Date, Event Code (outter)
col_list=["Countries","EventDate","EventRootCode"]
compareOutput = selectData.join( microsoftDF, col_list, how='full')
compareOutput = compareOutput.withColumn('a_difference', (F.col('p_nArticles') - F.col('m_nArticles')))
compareOutput = compareOutput.withColumn('e_difference', (F.col('p_EventReportValue') - F.col('m_EventReportValue')))
compareOutput = compareOutput.withColumn('g_difference', (F.col('p_GoldsteinReportValue') - F.col('m_GoldsteinReportValue')))
compareOutput = compareOutput.withColumn('t_difference', (F.col('p_ToneReportValue') - F.col('m_ToneReportValue')))
compareOutput.limit(10).toPandas()

# COMMAND ----------

display(compareOutput.select('p_EventReportValue', 'm_EventReportValue', 'e_difference'))

# COMMAND ----------

display(compareOutput.select('p_GoldsteinReportValue', 'm_GoldsteinReportValue', 'g_difference'))

# COMMAND ----------

display(compareOutput.select('p_ToneReportValue', 'm_ToneReportValue', 't_difference'))

# COMMAND ----------

display(compareOutput.select('p_nArticles', 'm_nArticles', 'a_difference'))

# COMMAND ----------

selectData.filter( (F.col('Countries') == 'Afghanistan') & (F.col('EventDate') == F.lit('2021-01-01'))).toPandas()