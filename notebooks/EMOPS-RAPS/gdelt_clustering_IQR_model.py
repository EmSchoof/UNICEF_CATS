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
# MAGIC - (1) The interquartile range is the best measure of variability for skewed distributions or data sets with outliers. Because it’s based on values that come from the middle half of the distribution, it’s unlikely to be influenced by outliers.

# COMMAND ----------

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

# convert datetime column to dates
preprocessedGDELTcon40 = preprocessedGDELTcon40.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
preprocessedGDELTcon40.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create Target Variables
# MAGIC - Calculate IQR for clusters 
# MAGIC - Calculate IQR and IQR-Outlier for all countries
# MAGIC - Calculate Power for Statistics

# COMMAND ----------

# DBTITLE 1,Create Rolling Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# --- for Creating Metrics ---

# create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName', 'QuadClassString', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'QuadClassString', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'QuadClassString', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Create Initial Report Variables
# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = preprocessedGDELTcon40.groupBy('ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString') \
                                     .agg(F.avg('Confidence').alias('avgConfidence'),
                                          F.collect_list('GoldsteinScale').alias('GoldsteinList'),
                                          F.collect_list('MentionDocTone').alias('ToneList'),
                                          F.sum('nArticles').alias('nArticles')) \
                                      .withColumn('GoldsteinReportValue', median_udf('GoldsteinList')) \
                                      .withColumn('ToneReportValue', median_udf('ToneList')) \
                                      .drop('GoldsteinList','ToneList')

# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName','EventTimeDate').orderBy('EventTimeDate')

# get daily distribution of articles for each Event Code string within Window
targetOutputPartitioned = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate IQR for 12 month and 24 month Windows

# COMMAND ----------

# DBTITLE 1,UDF Functions
lowerQ_udf = F.udf(lambda x: float(np.quantile(x, 0.25)), FloatType())
upperQ_udf = F.udf(lambda x: float(np.quantile(x, 0.75)), FloatType())
IQR_udf = F.udf(lambda lowerQ, upperQ: (upperQ - lowerQ), FloatType())

# COMMAND ----------

# DBTITLE 1,Create IQR Values per Country per Date per Event Code
# Events: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                                 .withColumn('ERV_3d_median', median_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('ERV_60d_median', median_udf('ERV_60d_list'))

# Goldstein: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
                                                 .withColumn('GRV_1d_median', median_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))

# Tone: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
                                                 .withColumn('TRV_1d_median', median_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))
# drop unnecessary columns
targetOutputPartitioned = targetOutputPartitioned.drop('ERV_3d_list','ERV_60d_list','GRV_1d_list',
                                                       'GRV_60d_list','TRV_1d_list','TRV_60d_list') \
                                                 .orderBy('EventTimeDate', ascending=False)

# verify output data
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(3).toPandas()

# COMMAND ----------

# DBTITLE 1,Create IQR Time Windows for 12 and 24 months
# MAGIC %md
# MAGIC #### Since this data only goes back 4 months, the time windows will not be accurate and will inevitably match. However, the formulation will proceed as follows:

# COMMAND ----------

# --- Windows for Evaluation Periods ---

# create a 12 month Window, 12 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling12m_window = Window.partitionBy('ActionGeo_FullName', 'QuadClassString', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(365), 0)

# create a 24 month Window, 24 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling24m_window = Window.partitionBy('ActionGeo_FullName', 'QuadClassString', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(730), 0)

# COMMAND ----------

 # Events: 3d, 60d
targetOutputTimelines = targetOutputPartitioned.withColumn('ERV_3d_12month_list', F.collect_list('ERV_3d_median').over(rolling12m_window)) \
                                               .withColumn('ERV_3d_sampleN',  F.size('ERV_3d_12month_list'))  \
                                               .withColumn('ERV_3d_quantile25', lowerQ_udf('ERV_3d_12month_list'))  \
                                               .withColumn('ERV_3d_quantile75', upperQ_udf('ERV_3d_12month_list'))  \
                                               .withColumn('ERV_3d_IQR', IQR_udf(F.col('ERV_3d_quantile25'), F.col('ERV_3d_quantile75')))  \
                                               .withColumn('ERV_60d_24month_list', F.collect_list('ERV_60d_median').over(rolling24m_window)) \
                                               .withColumn('ERV_60d_sampleN', F.size('ERV_60d_24month_list'))  \
                                               .withColumn('ERV_60d_quantile25', lowerQ_udf('ERV_60d_24month_list'))  \
                                               .withColumn('ERV_60d_quantile75', upperQ_udf('ERV_60d_24month_list')) \
                                               .withColumn('ERV_60d_IQR', IQR_udf(F.col('ERV_60d_quantile25'), F.col('ERV_60d_quantile75')))

# Goldstein: 1d, 60d
targetOutputTimelines = targetOutputTimelines.withColumn('GRV_1d_12month_list', F.collect_list('GRV_1d_median').over(rolling12m_window)) \
                                             .withColumn('GRV_1d_sampleN',  F.size('GRV_1d_12month_list'))  \
                                             .withColumn('GRV_1d_quantile25', lowerQ_udf('GRV_1d_12month_list'))  \
                                             .withColumn('GRV_1d_quantile75', upperQ_udf('GRV_1d_12month_list'))  \
                                             .withColumn('GRV_1d_IQR', IQR_udf(F.col('GRV_1d_quantile25'), F.col('GRV_1d_quantile75')))  \
                                             .withColumn('GRV_60d_24month_list', F.collect_list('GRV_60d_median').over(rolling24m_window)) \
                                             .withColumn('GRV_60d_sampleN', F.size('GRV_60d_24month_list'))  \
                                             .withColumn('GRV_60d_quantile25', lowerQ_udf('GRV_60d_24month_list'))  \
                                             .withColumn('GRV_60d_quantile75', upperQ_udf('GRV_60d_24month_list')) \
                                             .withColumn('GRV_60d_IQR', IQR_udf(F.col('GRV_60d_quantile25'), F.col('GRV_60d_quantile75')))
# Tone: 1d, 60d
targetOutputTimelines = targetOutputTimelines.withColumn('TRV_1d_12month_list', F.collect_list('TRV_1d_median').over(rolling12m_window)) \
                                             .withColumn('TRV_1d_sampleN', F.size('TRV_1d_12month_list'))  \
                                             .withColumn('TRV_1d_quantile25', lowerQ_udf('TRV_1d_12month_list'))  \
                                             .withColumn('TRV_1d_quantile75', upperQ_udf('TRV_1d_12month_list'))  \
                                             .withColumn('TRV_1d_IQR', IQR_udf(F.col('TRV_1d_quantile25'), F.col('TRV_1d_quantile75')))  \
                                             .withColumn('TRV_60d_24month_list', F.collect_list('TRV_60d_median').over(rolling24m_window)) \
                                             .withColumn('TRV_60d_sampleN', F.size('TRV_60d_24month_list'))  \
                                             .withColumn('TRV_60d_quantile25', lowerQ_udf('TRV_60d_24month_list'))  \
                                             .withColumn('TRV_60d_quantile75', upperQ_udf('TRV_60d_24month_list')) \
                                             .withColumn('TRV_60d_IQR', IQR_udf(F.col('TRV_60d_quantile25'), F.col('TRV_60d_quantile75')))

# COMMAND ----------

# verify output data
targetOutputTimelines = targetOutputTimelines.orderBy('EventTimeDate', ascending=False)
print((targetOutputTimelines.count(), len(targetOutputTimelines.columns)))
targetOutputTimelines.limit(3).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Detect Outliers

# COMMAND ----------

def get_upper_outliers(median, upperQ, IQR):
  mild_upper_outlier = upperQ + (IQR*1.5)
  extreme_upper_outlier = upperQ + (IQR*3)
  
  if (median >= mild_upper_outlier) and (median < extreme_upper_outlier):
     return 'mild max outlier'
  elif (median >= extreme_upper_outlier):
    return 'extreme max outlier'
  else:
     return 'not max outlier'

outliers_udf = F.udf(get_upper_outliers, StringType())

# COMMAND ----------

# identify outliers
assessVariableOutliers = targetOutputTimelines.withColumn('ERV_3d_outlier', outliers_udf('ERV_3d_median','ERV_3d_quantile75','ERV_3d_IQR')) \
                                             .withColumn('ERV_60d_outlier', outliers_udf('ERV_60d_median','ERV_60d_quantile75','ERV_60d_IQR')) \
                                             .withColumn('GRV_1d_outlier', outliers_udf('GRV_1d_median','GRV_1d_quantile75','GRV_1d_IQR')) \
                                             .withColumn('GRV_60d_outlier', outliers_udf('GRV_60d_median','GRV_60d_quantile75','GRV_60d_IQR')) \
                                             .withColumn('TRV_1d_outlier', outliers_udf('TRV_1d_median','TRV_1d_quantile75','TRV_1d_IQR')) \
                                             .withColumn('TRV_60d_outlier', outliers_udf('TRV_60d_median','TRV_60d_quantile75','TRV_60d_IQR'))
# verify output data
assessVariableOutliers = assessVariableOutliers.orderBy('EventTimeDate', ascending=False)
print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
assessVariableOutliers.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles',
                              'ERV_3d_outlier','ERV_60d_outlier',
                              'GRV_1d_outlier','GRV_60d_outlier',
                              'TRV_1d_outlier','TRV_60d_outlier'
                             ).limit(20).toPandas()

# COMMAND ----------

assessVariableOutliers.columns

# COMMAND ----------

cols = ['ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString','avgConfidence','nArticles','GoldsteinReportValue','ToneReportValue','EventReportValue','ERV_3d_median','ERV_60d_median','GRV_1d_median','GRV_60d_median','TRV_1d_median','TRV_60d_median','ERV_3d_sampleN','ERV_3d_quantile25','ERV_3d_quantile75','ERV_3d_IQR','ERV_60d_sampleN','ERV_60d_quantile25','ERV_60d_quantile75','ERV_60d_IQR','GRV_1d_sampleN','GRV_1d_quantile25','GRV_1d_quantile75','GRV_1d_IQR','GRV_60d_sampleN','GRV_60d_quantile25','GRV_60d_quantile75','GRV_60d_IQR','TRV_1d_sampleN','TRV_1d_quantile25','TRV_1d_quantile75','TRV_1d_IQR','TRV_60d_sampleN','TRV_60d_quantile25','TRV_60d_quantile75','TRV_60d_IQR','ERV_3d_outlier','ERV_60d_outlier','GRV_1d_outlier','GRV_60d_outlier','TRV_1d_outlier','TRV_60d_outlier']
assessVariableOutliersSelect = assessVariableOutliers.select(cols)

# COMMAND ----------

assessVariableOutliersSelect.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_19april2021.csv')