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
  .load("/FileStore/tables/tmp/gdelt/preprocessed_may2021.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Create QuadClass, EventRooteCode Dataframe
# get unique quadclass: eventcode pairs
quadClassCodes = preprocessedGDELT.select('QuadClassString','EventRootCodeString').dropDuplicates()

# Create distinct list of codes
quadclass = quadClassCodes.select('QuadClassString').rdd.map(lambda r: r[0]).collect()
eventcodes = quadClassCodes.select('EventRootCodeString').rdd.map(lambda r: r[0]).collect()

# Create quadclass: eventcode dictionary
cameo_quadclass_dict = dict(zip(eventcodes, quadclass))
cameo_quadclass_dict

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create Target Variables

# COMMAND ----------

# DBTITLE 1,Create Initial Report Variables
# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = preprocessedGDELT.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString') \
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

# DBTITLE 1,Create Rolling Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# --- for Creating Metrics ---

# # create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
# rolling1d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# # create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
# rolling3d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# # create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
# rolling60d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Calculate Medians per Country per Date per QuadClass and Event Code
# # Events: 3d, 60d
# targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
#                                                  .withColumn('ERV_3d_median', median_udf('ERV_3d_list'))  \
#                                                  .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
#                                                  .withColumn('ERV_60d_median', median_udf('ERV_60d_list'))

# # Goldstein: 1d, 60d
# targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
#                                                  .withColumn('GRV_1d_median', median_udf('GRV_1d_list'))  \
#                                                  .withColumn('GRV_60d_list', F.collect_list('GoldsteinReportValue').over(rolling60d_window)) \
#                                                  .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))

# # Tone: 1d, 60d
# targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
#                                                  .withColumn('TRV_1d_median', median_udf('TRV_1d_list'))  \
#                                                  .withColumn('TRV_60d_list', F.collect_list('ToneReportValue').over(rolling60d_window)) \
#                                                  .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))
# # drop unnecessary columns
# targetOutputPartitioned = targetOutputPartitioned.drop('ERV_3d_list','ERV_60d_list','GRV_1d_list',
#                                                        'GRV_60d_list','TRV_1d_list','TRV_60d_list') \
#                                                  .orderBy('EventTimeDate', ascending=False)

# # verify output data
# print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
# targetOutputPartitioned.limit(3).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create IQR Time Windows for 12 and 24 months
# MAGIC 
# MAGIC #### NOTE: Since this data only goes back 4 months, the time windows will not be accurate and will inevitably match. However, the formulation will proceed as follows:

# COMMAND ----------

# --- Windows for Evaluation Periods ---

# create a 3 month Window, 3 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3m_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(91), 0)

# create a 6 month Window, 6 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling6m_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(182), 0)

# COMMAND ----------

# DBTITLE 1,UDF Functions
lowerQ_udf = F.udf(lambda x: float(np.quantile(x, 0.25)), FloatType())
upperQ_udf = F.udf(lambda x: float(np.quantile(x, 0.75)), FloatType())
IQR_udf = F.udf(lambda lowerQ, upperQ: (upperQ - lowerQ), FloatType())

# COMMAND ----------

# ERV_3d_median // ERV_60d_median
targetOutputTimelines = targetOutputPartitioned.withColumn('ERV_3d_3month_list', F.collect_list('EventReportValue').over(rolling3m_window)) \
                                               .withColumn('ERV_3d_3month_median', median_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_3month_sampleN',  F.size('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_quantile25', lowerQ_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_quantile75', upperQ_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_3month_IQR', IQR_udf(F.col('ERV_3d_quantile25'), F.col('ERV_3d_quantile75')))  \
                                               .withColumn('ERV_60d_6month_list', F.collect_list('EventReportValue').over(rolling6m_window)) \
                                               .withColumn('ERV_60d_6month_median', median_udf('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_6month_sampleN', F.size('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_quantile25', lowerQ_udf('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_quantile75', upperQ_udf('ERV_60d_6month_list')) \
                                               .withColumn('ERV_60d_6month_IQR', IQR_udf(F.col('ERV_60d_quantile25'), F.col('ERV_60d_quantile75')))

# Goldstein: 1d, 60d
# GRV_1d_median // GRV_60d_median
targetOutputTimelines = targetOutputTimelines.withColumn('GRV_1d_3month_list', F.collect_list('GoldsteinReportValue').over(rolling3m_window)) \
                                             .withColumn('GRV_1d_3month_median', median_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_3month_sampleN',  F.size('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_quantile25', lowerQ_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_quantile75', upperQ_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_3month_IQR', IQR_udf(F.col('GRV_1d_quantile25'), F.col('GRV_1d_quantile75')))  \
                                             .withColumn('GRV_60d_6month_list', F.collect_list('GoldsteinReportValue').over(rolling6m_window)) \
                                             .withColumn('GRV_60d_6month_median', median_udf('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_6month_sampleN', F.size('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_quantile25', lowerQ_udf('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_quantile75', upperQ_udf('GRV_60d_6month_list')) \
                                             .withColumn('GRV_60d_6month_IQR', IQR_udf(F.col('GRV_60d_quantile25'), F.col('GRV_60d_quantile75')))
# Tone: 1d, 60d
# TRV_1d_median // TRV_60d_median
targetOutputTimelines = targetOutputTimelines.withColumn('TRV_1d_3month_list', F.collect_list('ToneReportValue').over(rolling3m_window)) \
                                             .withColumn('TRV_1d_3month_median', median_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_3month_sampleN', F.size('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_quantile25', lowerQ_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_quantile75', upperQ_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_3month_IQR', IQR_udf(F.col('TRV_1d_quantile25'), F.col('TRV_1d_quantile75')))  \
                                             .withColumn('TRV_60d_6month_list', F.collect_list('ToneReportValue').over(rolling6m_window)) \
                                             .withColumn('TRV_60d_6month_median', median_udf('TRV_60d_6month_list'))  \
                                             .withColumn('TRV_60d_6month_sampleN', F.size('TRV_60d_6month_list'))  \
                                             .withColumn('TRV_60d_quantile25', lowerQ_udf('TRV_60d_6month_list'))  \
                                             .withColumn('TRV_60d_quantile75', upperQ_udf('TRV_60d_6month_list')) \
                                             .withColumn('TRV_60d_6month_IQR', IQR_udf(F.col('TRV_60d_quantile25'), F.col('TRV_60d_quantile75')))

# COMMAND ----------

# verify output data
targetOutputTimelines = targetOutputTimelines.orderBy('EventTimeDate', ascending=False)
print((targetOutputTimelines.count(), len(targetOutputTimelines.columns)))
targetOutputTimelines.limit(3).toPandas()

# COMMAND ----------

#targetOutputTimelines.select('GRV_60d_24month_list', 'TRV_60d_24month_list').limit(10).toPandas()

# COMMAND ----------

#.withColumn('GRV_60d_24month_list', F.collect_list('GRV_60d_median').over(rolling24m_window))

#.withColumn('TRV_60d_24month_list', F.collect_list('TRV_60d_median').over(rolling24m_window))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Detect Outliers

# COMMAND ----------

def get_outliers(median, upperQ, IQR):
  upper_outlier = upperQ + (IQR*1.5)
  
  if median > upper_outlier:
     return 'outlier (max)'
  else:
     return 'not outlier (max)'
    
def get_lower_outliers(median, lowerQ, IQR):
  lower_outlier = lowerQ - (IQR*1.5)
  
  if median < lower_outlier:
     return 'outlier (min)'
  else:
     return 'not outlier (min)'

upper_outliers_udf = F.udf(get_upper_outliers, StringType())
lower_outliers_udf = F.udf(get_lower_outliers, StringType())

# COMMAND ----------

# identify outliers
assessVariableOutliers = targetOutputTimelines.withColumn('ERV_3m_outlier', upper_outliers_udf('EventReportValue','ERV_3d_quantile75','ERV_3d_3month_IQR')) \
                                             .withColumn('ERV_6m_outlier', upper_outliers_udf('EventReportValue','ERV_60d_quantile75','ERV_60d_6month_IQR')) \
                                             .withColumn('GRV_3m_outlier', lower_outliers_udf('GoldsteinReportValue','GRV_1d_quantile25','GRV_1d_3month_IQR')) \
                                             .withColumn('GRV_6m_outlier', lower_outliers_udf('GoldsteinReportValue','GRV_60d_quantile25','GRV_60d_6month_IQR')) \
                                             .withColumn('TRV_3m_outlier', lower_outliers_udf('ToneReportValue','TRV_1d_quantile25','TRV_1d_3month_IQR')) \
                                             .withColumn('TRV_6m_outlier', lower_outliers_udf('ToneReportValue','TRV_60d_quantile25','TRV_60d_6month_IQR'))
# drop unnecessary columns
assessVariableOutliers = assessVariableOutliers.drop('ERV_3d_3month_list', 'ERV_60d_6month_list',
                                                    'GRV_1d_3month_list', 'GRV_60d_6month_list',
                                                    'TRV_1d_3month_list', 'TRV_60d_6month_list')

# verify output data
assessVariableOutliers = assessVariableOutliers.orderBy('EventTimeDate', ascending=False)
print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
assessVariableOutliers.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles',
                              'ERV_3m_outlier','ERV_6m_outlier',
                              'GRV_3m_outlier','GRV_6m_outlier',
                              'TRV_3m_outlier','TRV_6m_outlier'
                             ).limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Map QuadClass back in to Dataframe
# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_quadclass_dict.items())])
assessVariableOutliers = assessVariableOutliers.withColumn('QuadClassString', mapping_expr[F.col('EventRootCodeString')])

# Confirm accurate output
assessVariableOutliers.select('QuadClassString', 'EventRootCodeString').dropDuplicates()#.show()
assessVariableOutliers.limit(1).toPandas()

# COMMAND ----------

assessVariableOutliers.columns

# COMMAND ----------

# DBTITLE 1,Create Separate DataFrames for Cleaner Output
# # Event Report Value
# evr_cols = ['ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString','EventReportValue',
#             'ERV_3d_median','ERV_3d_12month_median','ERV_3d_12month_sampleN', 'ERV_3d_outlier',
#             'ERV_60d_median','ERV_60d_24month_median','ERV_60d_24month_sampleN','ERV_60d_outlier']

# event_report_value = assessVariableOutliers.select(evr_cols)

# # Rename columns for clean output
# old_erv = event_report_value.schema.names
# new_evr = ['Country','Event Date','Quad Class','CAMEO Root Code','EventReportValue',
#            '3 Day Rolling Median','3 Day Median, Rolling 12month IQR', 'Sample Size, Rolling 12month IQR', '3 Day Median Outlier',
#            '60 Day Rolling Median','60 Day Median, Rolling 24month IQR','Sample Size, Rolling 24month IQR','60 Day Median Outlier']

# event_report_value_csv = reduce(lambda event_report_value, idx: event_report_value.withColumnRenamed(old_erv[idx], new_evr[idx]), range(len(old_erv)), event_report_value)
# event_report_value_csv.limit(5).toPandas()

# COMMAND ----------

#  # Goldstein Report Value
# gold_cols = ['ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString','GoldsteinReportValue',
#             'GRV_1d_median','GRV_1d_12month_median','GRV_1d_12month_sampleN', 'GRV_1d_outlier',
#             'GRV_60d_median','GRV_60d_24month_median','GRV_60d_24month_sampleN','GRV_60d_outlier']

# golstein_report_value = assessVariableOutliers.select(gold_cols)

# # Rename columns for clean output
# old_gold = golstein_report_value.schema.names
# new_gold = ['Country','Event Date','Quad Class','CAMEO Root Code','GoldsteinReportValue',
#            '1 Day Rolling Median','1 Day Median, Rolling 12month IQR', 'Sample Size, Rolling 12month IQR', '1 Day Median Outlier',
#            '60 Day Rolling Median','60 Day Median, Rolling 24month IQR','Sample Size, Rolling 24month IQR','60 Day Median Outlier']

# golstein_report_value_csv = reduce(lambda golstein_report_value, idx: golstein_report_value.withColumnRenamed(old_gold[idx], new_gold[idx]), range(len(old_gold)), golstein_report_value)
# golstein_report_value_csv.limit(5).toPandas()

# COMMAND ----------

#  # Tone Report Value
# tone_cols = ['ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString','ToneReportValue',
#             'TRV_1d_median','TRV_1d_12month_median','TRV_1d_12month_sampleN', 'TRV_1d_outlier',
#             'TRV_60d_median','TRV_60d_24month_median','TRV_60d_24month_sampleN','TRV_60d_outlier']

# tone_report_value = assessVariableOutliers.select(tone_cols)

# # Rename columns for clean output
# old_tone = tone_report_value.schema.names
# new_tone = ['Country','Event Date','Quad Class','CAMEO Root Code','ToneReportValue',
#            '1 Day Rolling Median','1 Day Median, Rolling 12month IQR', 'Sample Size, Rolling 12month IQR', '1 Day Median Outlier',
#            '60 Day Rolling Median','60 Day Median, Rolling 24month IQR','Sample Size, Rolling 24month IQR','60 Day Median Outlier']

# tone_report_value_csv = reduce(lambda tone_report_value, idx: tone_report_value.withColumnRenamed(old_tone[idx], new_tone[idx]), range(len(old_tone)), tone_report_value)
# tone_report_value_csv.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Store as CSV
import os

# TEMPORARY_TARGET="dbfs:/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_20april2021"
# DESIRED_TARGET="dbfs:/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_20april2021.csv"

# assessVariableOutliers.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_TARGET)
# temporary_csv = os.path.join(TEMPORARY_TARGET, dbutils.fs.ls(TEMPORARY_TARGET)[3][1])
# dbutils.fs.cp(temporary_csv, DESIRED_TARGET)

# COMMAND ----------

# 'ERV_3d_median','ERV_60d_median','ERV_3d_3month_median','ERV_3d_3month_sampleN', 'ERV_60d_6month_median','ERV_60d_6month_sampleN',
#'GRV_1d_median','GRV_60d_median','GRV_1d_3month_median','GRV_1d_3month_sampleN','GRV_60d_6month_median','GRV_60d_6month_sampleN','TRV_60d_6month_median','TRV_60d_6month_sampleN',

# COMMAND ----------

cols = ['ActionGeo_FullName','EventTimeDate','QuadClassString','EventRootCodeString','nArticles',
       'EventReportValue','ERV_3m_outlier','ERV_6m_outlier',
       'GoldsteinReportValue','GRV_3m_outlier','GRV_6m_outlier',
       'ToneReportValue','TRV_3m_outlier','TRV_6m_outlier']
assessVariableOutliersSelect = assessVariableOutliers.select(cols)

# select only february and beyond
assessVariableOutliersSelect = assessVariableOutliersSelect.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
powerBI = assessVariableOutliersSelect.filter(F.col('EventTimeDate') >= F.lit('2021-03-01'))
powerBI.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Clean Up Columns for Output
# store to CSV
TEMPORARY_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/modified_IQR_alertsystem_02may2021"
DESIRED_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/modified_IQR_alertsystem_02may2021.csv"

powerBI.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_BI_TARGET)
temporaryPoweBI_csv = os.path.join(TEMPORARY_BI_TARGET, dbutils.fs.ls(TEMPORARY_BI_TARGET)[3][1])
dbutils.fs.cp(temporaryPoweBI_csv, DESIRED_BI_TARGET)