# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Having established that niether the mean/standard deviation combination nor the median/median absolute deviation combination are viable metricts for the creation of an anomoly detection alert system, the following script seeks to create a flexible clustering model to identify the z parameter cut off criteria for each variable for each period of analysis. 
# MAGIC 
# MAGIC %md 
# MAGIC ### Target Variable #1 – Distribution of Articles by Country by Event Type
# MAGIC - 	Event report value (ERV): 
# MAGIC Calculated as the distribution of articles with respect to an event type category per country per day
# MAGIC -	Event Running Average 1 (ERA1):
# MAGIC Calculated as the rolling **median** of the ERV for 3 days over the previous 12 months
# MAGIC -	Event Running Average 2 (ERA2):
# MAGIC Calculated as the rolling **median** of the ERV for 60 days over the previous 24 months
# MAGIC 
# MAGIC 
# MAGIC ### Target Variable #2 – Medians of Goldstein Score Averages
# MAGIC - 	Goldstein point value (GPV): 
# MAGIC Calculated as the average Goldstein score for all articles with respect to an event type category per country per day
# MAGIC -	Goldstein Running Average (GRA1):
# MAGIC Calculated as the rolling **median** of the GPV for PA13 over the previous 12 months
# MAGIC -	Goldstein Running Average (GRA2):
# MAGIC Calculated as the rolling **median** of the GPV for 60 days over the previous 24 months
# MAGIC 
# MAGIC ### Target Variable #3 – Medians of Tone Score Averages
# MAGIC - 	Tone point value (TPV): 
# MAGIC Calculated as the average Mention Tone for all articles with respect to an event type category per country per day
# MAGIC -	Tone Running Average (TRA1):
# MAGIC Calculated as the rolling **median** of the TPV for PA1 over the previous 12 months
# MAGIC -	Tone Running Average (TRA2):
# MAGIC Calculated as the rolling **median** of the TPV for 60 days over the previous 24 months
# MAGIC 
# MAGIC ### Periods of Analysis
# MAGIC - 1 day
# MAGIC - 3 days
# MAGIC - 60 days 
# MAGIC 
# MAGIC ### Premise of Task
# MAGIC - (1.0) Compute X day *median* for all the data values across all years (2019 to present, result will be X = [x1, x2, x3, ... xn]).
# MAGIC - (2.0) Calculate the difference between X and the *daily median*.
# MAGIC - (3.0) Set a threshold parameter z 
# MAGIC - (4.0) Compare z and *daily median*. If X is greater than or equal to z, alert as an outlier.
# MAGIC - (5.0) Verify z threshold with past (known) data.
# MAGIC 
# MAGIC Sources:
# MAGIC - (1) [Facebook's Prophecy Forcesting Model](https://github.com/facebook/prophet/tree/master/python)
# MAGIC   - (a) [Medium Example](https://towardsdatascience.com/a-quick-start-of-time-series-forecasting-with-a-practical-example-using-fb-prophet-31c4447a2274)
# MAGIC - (2) [Understanding LSTM Recurrent Neural Network models](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

# COMMAND ----------

# DBTITLE 1,Import Modules
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

# MAGIC %md
# MAGIC ### Part 1: Create Target Variables

# COMMAND ----------

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# COMMAND ----------

# DBTITLE 1,Import Preprocessed Data
# The applied options are for CSV files.  
preprocessedGDELT = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/tmp/gdelt/preprocessed.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Data with Confidence of 40% or higher
# create confidence column of more than 
print(preprocessedGDELT.count())
preprocessedGDELTcon40 = preprocessedGDELT.filter(F.col('Confidence') >= 40)
print(preprocessedGDELTcon40.count())
preprocessedGDELTcon40.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Initial Values of Target Variables
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode #,'ActionGeo_Lat','ActionGeo_Long'
targetOutput = preprocessedGDELTcon40.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString') \
                                     .agg(F.avg('Confidence').alias('avgConfidence'),
                                          F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                          F.avg('MentionDocTone').alias('ToneReportValue'),
                                          F.sum('nArticles').alias('nArticles'))
print((targetOutput.count(), len(targetOutput.columns)))
targetOutput.limit(2).toPandas()

# COMMAND ----------

# create a Window, country by date
countriesDaily_window = Window.partitionBy('EventTimeDate', 'ActionGeo_FullName').orderBy('EventTimeDate')

# get daily distribution of articles for each Event Code string within Window
targetOutputPartitioned = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
targetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# verify output
sumERV = targetOutputPartitioned.select('EventTimeDate','ActionGeo_FullName','EventReportValue').groupBy('EventTimeDate', 'ActionGeo_FullName').agg(F.sum('EventReportValue'))
print('Verify all sum(EventReportValue)s are 1')
plt.plot(sumERV.select('sum(EventReportValue)').toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 2:
# MAGIC #### Create Model 

# COMMAND ----------

# DBTITLE 1,Create Rolling Windows for Median
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Create Medians for Output Variables
# Events: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                                 .withColumn('ERV_3d_median', median_udf('ERV_3d_list')) \
                                                 .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('ERV_60d_median', median_udf('ERV_60d_list')) 

# Goldstein: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
                                                 .withColumn('GRV_1d_median', median_udf('GRV_1d_list')) \
                                                 .withColumn('GRV_60d_list', F.collect_list('GoldsteinReportValue').over(rolling60d_window)) \
                                                 .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))

# Tone: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
                                                 .withColumn('TRV_1d_median', median_udf('TRV_1d_list')) \
                                                 .withColumn('TRV_60d_list', F.collect_list('ToneReportValue').over(rolling60d_window)) \
                                                 .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))

# drop extra columns
targetOutputPartitioned = targetOutputPartitioned.drop('ERV_3d_list','ERV_60d_list',
                                                       'GRV_1d_list', 'GRV_60d_list',
                                                       'TRV_1d_list', 'TRV_60d_list')

# verify output data
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(3).toPandas()