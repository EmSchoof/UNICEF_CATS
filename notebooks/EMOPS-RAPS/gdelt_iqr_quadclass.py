# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Based off of the methodology for the conflict alert system in (Levin, Ali, & Crandall, 2018)[source](http://www.lrec-conf.org/proceedings/lrec2020/workshops/AESPEN2020/pdf/2020.aespen-1.8.pdf) as a precursor for a more indepth predition model, I wanted to explore IQR system at the QuadClass level (particularly focusing on Material Conflict).

# COMMAND ----------

# DBTITLE 1,Import Modules
from dateutil.parser import parse
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
import seaborn as sns

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
  .load("/FileStore/tables/tmp/gdelt/preprocessed.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Convert Date string to Datetime Column
# Convert date-strings to date columns
date_udf = F.udf(lambda d: parse(d), DateType())

preprocessedGDELT = preprocessedGDELT.withColumn('EventTimeDate', date_udf('EventTimeDate'))

# create yearmonth column
preprocessedGDELT = preprocessedGDELT.withColumn('EventYear', F.year('EventTimeDate'))
preprocessedGDELT = preprocessedGDELT.withColumn('EventMonth', F.month('EventTimeDate'))
preprocessedGDELT = preprocessedGDELT.withColumn('YearMonth', F.concat_ws("-", F.col('EventYear'), F.col('EventMonth')))
preprocessedGDELT.limit(2).toPandas()

# COMMAND ----------



# gdelt2021Pandas = targetOutputPartitioned.toPandas()
# gdelt2021Pandas.head()

# COMMAND ----------

# DBTITLE 1,Create Initial Report Variables
# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = preprocessedGDELT.groupBy('ActionGeo_FullName','YearMonth','QuadClassString') \
                                .agg(F.collect_list('GoldsteinScale').alias('GoldsteinList'),
                                     F.collect_list('MentionDocTone').alias('ToneList'),
                                     F.sum('nArticles').alias('nArticles')) \
                                 .withColumn('GoldsteinReportValue', median_udf('GoldsteinList')) \
                                 .withColumn('ToneReportValue', median_udf('ToneList')) \
                                 .drop('GoldsteinList','ToneList')

# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName','YearMonth').orderBy('YearMonth')

# get daily distribution of articles for each Event Code string within Window
targetOutputPartitioned = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC pg 45
# MAGIC 
# MAGIC """
# MAGIC In order to rank the countries in the most appropriate way,
# MAGIC we compute the rate of change between the Q4 of the
# MAGIC current month and the Q4 of the previous one (here called
# MAGIC delta classification). Hence, the rate of change is
# MAGIC 
# MAGIC 𝛥𝑄4 = (𝑄4𝑥 −𝑄4𝑥−1) / 𝑄4𝑥−1
# MAGIC 
# MAGIC where 𝑄4𝑥 is the proportion of the Q4 for the current month and 𝑄4𝑥−1 is the proportion of the Q4 for the previous month. 
# MAGIC 
# MAGIC 
# MAGIC Based on this rate, we rank the countries so as the
# MAGIC country with the highest increase in the Q4 (𝛥𝑄4) will be first and the country with the highest decrease will be the
# MAGIC last.
# MAGIC 
# MAGIC /~~~~/
# MAGIC 
# MAGIC To the initial country ranking, we further add a set alarms (value 0 or 1 if true) that consist of the following parameters:
# MAGIC - Alarm 1: The proportion of the Q4 (Q4x) for the current month is a local max, meaning that the increase is significant and out of the 95% CI that we have calculated for the x-month moving window.
# MAGIC - Alarm 2: The total absolute number of the events mentioned (current values) is a local max.
# MAGIC - Alarm 3: The proportion of the predicted values of the Q4 (Q4x+1) for the next month is a local max.
# MAGIC 
# MAGIC """
# MAGIC 
# MAGIC - 1: 'Verbal Cooperation'
# MAGIC - 2: 'Material Cooperation'
# MAGIC - 3: 'Verbal Conflict'
# MAGIC - 4: 'Material Conflict'

# COMMAND ----------

# explore quadclass 4 for last two months
quadClassChange = targetOutputPartitioned.filter(F.col('QuadClassString') == 'Material Conflict')

# create lag column 
w = Window.partitionBy('ActionGeo_FullName').orderBy('YearMonth')
quadClassChange = quadClassChange.withColumn('lead', F.lag(F.col('EventReportValue'), default=0).over(w))
quadClassChange.limit(10).show()

# COMMAND ----------

df.withColumn('lead', f.lag('rating', 1).over(w)) \
  .withColumn('rating_diff', f.when(f.col('lead').isNotNull(), f.col('rating') - f.col('lead')).otherwise(f.lit(None)))

# COMMAND ----------

# Get month change for QuadClass 4
monthChange_udf = F.udf(lambda m1, m2: (m2 - m1)/m1 , FloatType())
quadClassChange = quadClassChange.withColumn('ERV_monthChange', monthChange_udf(F.col('EventReportValue'), F.col('lead')))
                                                                       
quadClassChange.limit(10).show()

# COMMAND ----------

quadClassChange.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Visualize Boxplots (conflict)
# fig, ax = plt.subplots(figsize=(16, 24))
# sns.boxplot(data=gdelt2021Pandas.loc[ gdelt2021Pandas['Conflict'] == True ],
#             x='nArticles',
#             y='ActionGeo_FullName',
#             ax=ax)

# COMMAND ----------

# fig, ax = plt.subplots(figsize=(16, 24))
# sns.boxplot(data=gdelt2021Pandas.loc[ gdelt2021Pandas['Conflict'] == True ],
#             x='EventReportValue',
#             y='ActionGeo_FullName',
#             ax=ax)

# COMMAND ----------

# fig, ax = plt.subplots(figsize=(16, 24))
# sns.boxplot(data=gdelt2021Pandas,
#             x='GoldsteinReportValue',
#             y='ActionGeo_FullName',
#             ax=ax)

# COMMAND ----------

# fig, ax = plt.subplots(figsize=(16, 24))
# sns.boxplot(data=gdelt2021Pandas,
#             x='ToneReportValue',
#             y='ActionGeo_FullName',
#             ax=ax)

# COMMAND ----------

# DBTITLE 1,Visualize ERV Over Time
for country in countries:
  fig, ax = plt.subplots(figsize=(10, 6))
  gdelt2021.loc[ gdelt2021['ActionGeo_FullName'] == country ].set_index('EventTimeDate').plot(ax=ax)
  ax.set(title="Proportion of Conflict Articles Over Time in " + country,
         xlabel="EventTimeDate",
         ylabel="QuadClass")
  ax.axhline(0, linestyle="dashed", color="black", alpha=0.5)
  plt.show()

# COMMAND ----------

# DBTITLE 1,Store as CSV
#import os

# TEMPORARY_TARGET="dbfs:/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_20april2021"
# DESIRED_TARGET="dbfs:/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_20april2021.csv"

# assessVariableOutliers.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_TARGET)
# temporary_csv = os.path.join(TEMPORARY_TARGET, dbutils.fs.ls(TEMPORARY_TARGET)[3][1])
# dbutils.fs.cp(temporary_csv, DESIRED_TARGET)

# COMMAND ----------

# DBTITLE 1,Clean Up Columns for Output
# convert datetime column to dates
#powerBI = assessVariableOutliers.withColumn('EventMonth', F.month(F.to_timestamp('EventTimeDate', 'yyyy-MM-dd')))

# select only february and beyond
#powerBI = assessVariableOutliers.filter(F.where('EventMonth' == 4)).select(cols)

# store to CSV
#TEMPORARY_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/IQR_alertsystem_29april2021"
#DESIRED_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/IQR_alertsystem_29april2021.csv"

#powerBI.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_BI_TARGET)
#temporaryPoweBI_csv = os.path.join(TEMPORARY_BI_TARGET, dbutils.fs.ls(TEMPORARY_BI_TARGET)[3][1])
#dbutils.fs.cp(temporaryPoweBI_csv, DESIRED_BI_TARGET)

# COMMAND ----------

#powerBI.limit(10).toPandas()