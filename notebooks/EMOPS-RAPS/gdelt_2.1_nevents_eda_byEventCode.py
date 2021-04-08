# Databricks notebook source
# MAGIC %md
# MAGIC ### Calculations – Percentages of Articles by Country by Event Type
# MAGIC 
# MAGIC - 	Event report value (ERV): 
# MAGIC Calculated as the percentage of total articles categorized as belonging to a country that are categorized as matches for an event type
# MAGIC -   Event report sum (ERS):
# MAGIC <strike> Calculated as the number of articles categorized as belonging to a country that are categorized as matches for an event type </strike>
# MAGIC -	Event Running Average 1 (ERA1):
# MAGIC Calculated as the rolling average of the ERV for PA1 over the previous 12 months
# MAGIC -	Event Running Average 2 (ERA2):
# MAGIC Calculated as the rolling average of the ERV for PA2 over the previous 24 months
# MAGIC -	Period of Analysis 1 (PA1): 3 days
# MAGIC -	Period of Analysis 2 (PA2): 60 days 
# MAGIC -	Event spike alert: 
# MAGIC When the *Event Report Value* for a given PA1 (*3 DAYS*) is <strike>one standard deviation</strike>  above ERA1* 
# MAGIC -	Event trend alert: 
# MAGIC when the *Event Report Value* for a given PA2 (*60 DAYS*) is <strike>one standard deviation</strike>  above ERA2*
# MAGIC 
# MAGIC 
# MAGIC *New Methodology*
# MAGIC - (1.0) Compute three day average for all the data values you have across all years. 
# MAGIC - (1.a) Your result will be X = [x1, x2, x3, ... xn]
# MAGIC - (2.0) Set a threshhold parameter z. // benchmark special parameter (needs to be established per country?) come up with a "Ground Truth" value
# MAGIC - (3.0) For each value in X, compare with z 
# MAGIC 
# MAGIC *Execution of Methodology*
# MAGIC - 1 - [create a list of 3 day moving averages from today - 365 days] // compare this list with defined 'z' anomalist behavior to the current 3 day average per EventRootCode
# MAGIC - 2 - [create a list of 60 day moving averages from today - 730 days] // compare this list with defined 'z' anomalist behavior to the current 60 day average per EventRootCode
# MAGIC 
# MAGIC 
# MAGIC Sources:
# MAGIC - (1) [Moving Averaging with Apache Spark](https://www.linkedin.com/pulse/time-series-moving-average-apache-pyspark-laurent-weichberger/)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Statiticians and Machine Learning Engineers
# MAGIC - Objective
# MAGIC - Methodology of Outcome
# MAGIC - Organizing Data to Deliver on that Outcome
# MAGIC - Current Challenges that I am encountering to delivering this outcome

# COMMAND ----------

# DBTITLE 1,Import Modules
from functools import reduce
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.mllib.stat import Statistics
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import scipy.stats as stats
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Import Data with Target Variables
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files.  
preprocessedGDELT = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/tmp/gdelt/events_targetvalues.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Events Data
eventsData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence','EventReportValue','wERA_3d','wERA_60d')
print((eventsData.count(), len(eventsData.columns)))
eventsData.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test for Normal Distribution of % Articles by Country by EventCode

# COMMAND ----------

def get_normal_pval(vars_list):
    if len(vars_list) >= 8:
      k2, p = stats.normaltest(vars_list)
      return float(p)
    else:
      return float('nan')

def if_norm(p):
    alpha = 0.05 # 95% confidence
    if p < alpha: # if norm
      return True
    elif np.isnan(p) == True:
      return False
    else:
      return False

# Create UDF funcs
get_pval_udf = F.udf(lambda vars: get_normal_pval(vars), FloatType())
if_norm_udf = F.udf(lambda p: if_norm(p), BooleanType())

# COMMAND ----------

eventsDataAll = eventsData.select('ActionGeo_FullName', 'EventRootCodeString', 'wERA_3d', 'wERA_60d', 'nArticles') \
                                        .groupBy('ActionGeo_FullName','EventRootCodeString') \
                                        .agg( F.skewness('wERA_3d'),
                                              F.kurtosis('wERA_3d'),
                                              F.stddev('wERA_3d'),
                                              F.variance('wERA_3d'),
                                              F.collect_list('wERA_3d').alias('list_wERA_3d'),
                                              F.skewness('wERA_60d'),
                                              F.kurtosis('wERA_60d'),
                                              F.stddev('wERA_60d'),
                                              F.variance('wERA_60d'),
                                              F.collect_list('wERA_60d').alias('list_wERA_60d'),
                                              F.sum('nArticles').alias('nArticles'),
                                              F.count(F.lit(1)).alias('n_observations')
                                        )

# get p-value and define normalcy
eventsDataAll = eventsDataAll.withColumn('p_value_3d', get_pval_udf(eventsDataAll.list_wERA_3d))
eventsDataAll = eventsDataAll.withColumn('if_normal_3d', if_norm_udf(eventsDataAll.p_value_3d))
eventsDataAll = eventsDataAll.withColumn('p_value_60d', get_pval_udf(eventsDataAll.list_wERA_60d))
eventsDataAll = eventsDataAll.withColumn('if_normal_60d', if_norm_udf(eventsDataAll.p_value_60d))
eventsDataAll.limit(5).toPandas()

# COMMAND ----------

eventsDataAll.select('ActionGeo_FullName','EventRootCodeString', 'if_normal_3d').filter(F.col('if_normal_3d') == False).count()

# COMMAND ----------

eventsDataAll.select('ActionGeo_FullName','EventRootCodeString','if_normal_60d').filter(F.col('if_normal_60d') == False).count()

# COMMAND ----------

