# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### After careful evaluation and verification, it has been established that none of the target variables are normally distributed. Thus, using the standard deviation of these variables would be insufficient for establishing a viable alert system. 
# MAGIC 
# MAGIC *New Methodology*
# MAGIC - (1.0) Compute X day average for all the data values across all years (2019 to present, result will be X = [x1, x2, x3, ... xn]).
# MAGIC - (2.0) Set a threshhold parameter z. // benchmark special parameter (needs to be established per country therefore there is need to come up with a "Ground Truth" value)
# MAGIC - (3.0) For each value in X, compare with z. If X is greater than z, alert.
# MAGIC - (4.0) Verify z threshold with past (known) data.
# MAGIC 
# MAGIC *Execution of Methodology*
# MAGIC - 1 - [create a list of 3 day moving averages from today - 365 days] // compare this list with defined 'z' anomalist behavior to the current 3 day average per EventRootCode
# MAGIC - 2 - [create a list of 60 day moving averages from today - 730 days] // compare this list with defined 'z' anomalist behavior to the current 60 day average per EventRootCode

# COMMAND ----------

# DBTITLE 1,Import Modules
import json
import numpy as np
import pandas as pd
from pyspark.mllib.stat import Statistics
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# COMMAND ----------

# DBTITLE 1,Import ERV Data
eventReportValues = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/tmp/gdelt/erv_confidence40plus.csv") #
print((eventReportValues.count(), len(eventReportValues.columns)))
eventReportValues.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Import GRV Data
# The applied options are for CSV files.  
goldReportValues = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/tmp/gdelt/grv_confidence40plus.csv") #
print((goldReportValues.count(), len(goldReportValues.columns)))
goldReportValues.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Import TRV Data
# The applied options are for CSV files.  
toneReportValues = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/tmp/gdelt/trv_confidence40plus.csv") #
print((toneReportValues.count(), len(toneReportValues.columns)))
toneReportValues.limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prework:
# MAGIC #### Convert String Array Columns to Array Columns

# COMMAND ----------

def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

stripstring_udf = F.udf(parse_embedding_from_string, ArrayType(DoubleType()))

# COMMAND ----------

# ERV
eventReportValues = eventReportValues.withColumn('ERV_3d_list', stripstring_udf(F.col('ERV_3d_list'))) \
                                     .withColumn('ERV_60d_list', stripstring_udf(F.col('ERV_60d_list')))

# GRV
goldReportValues = goldReportValues.withColumn('GRV_1d_list', stripstring_udf(F.col('GRV_1d_list'))) \
                                   .withColumn('GRV_60d_list', stripstring_udf(F.col('GRV_60d_list')))

# TRV
toneReportValues = toneReportValues.withColumn('TRV_1d_list',  stripstring_udf(F.col('TRV_1d_list'))) \
                                   .withColumn('TRV_60d_list',  stripstring_udf(F.col('TRV_60d_list'))) \

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1:
# MAGIC #### Calculate Median Absolute Deviation (MAD) 
# MAGIC 
# MAGIC MAD is the difference between the value and the median.
# MAGIC 
# MAGIC In Spark you can use a SQL expression ‘percentile()’ to calculate any medians or quartiles in a dataframe. ‘percentile()’ expects a column and an array of percentiles to calculate (for median we can provide ‘array(0.5)’ because we want the 50% value ie median) and will return an array of results.
# MAGIC 
# MAGIC Like standard deviation, to use MAD to identify the outliers it needs to be a certain number of MAD’s away. This number is also referred to as the threshold and is defaulted to 3.
# MAGIC [source](https://www.advancinganalytics.co.uk/blog/2020/9/2/identifying-outliers-in-spark-30)

# COMMAND ----------

diff_udf = F.udf(lambda median, arr: [x - median for x in arr],
                   ArrayType(DoubleType()))

MAD_udf = F.udf(lambda arr: [ float(np.median(diff)) for diff in arr],
                   ArrayType(DoubleType()))

# COMMAND ----------

eventReportValuesT = eventReportValues.withColumn('3d_diff', diff_udf(F.col('ERV_3d_median'), F.col('ERV_3d_list')))
eventReportValuesT.limit(10).toPandas()

# COMMAND ----------

MADdf = df.groupby('genre') \
          .agg(F.expr('percentile(duration, array(0.5))')[0].alias('duration_median')) \
          .join(df, "genre", "left") \
          .withColumn("duration_difference_median", F.abs(F.col('duration')-F.col('duration_median'))) \
          .groupby('genre', 'duration_median') \
          .agg(F.expr('percentile(duration_difference_median, array(0.5))')[0].alias('median_absolute_difference'))

outliersremoved = df.join(MADdf, "genre", "left") \
                    .filter(
                           F.abs(F.col("duration")-F.col("duration_median")) <= (F.col("mean_absolute_difference")*3)
                           )

# COMMAND ----------

# MAGIC %md
# MAGIC This method is generally more effective than standard deviation but it suffers from the opposite problem as it can be too aggressive in identifying values as outliers even though they are not really extremely different. For an extreme example: if more than 50% of the data points have the same value, MAD is computed to be 0, so any value different from the residual median is classified as an outlier.

# COMMAND ----------



# COMMAND ----------

