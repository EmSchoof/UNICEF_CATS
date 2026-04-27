# Databricks notebook source
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
preprocessedGDELT.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create Event Report Variable by Conflict/Not

# COMMAND ----------

countries = ['Algeria', 'Angola', 'Bahrain', 'Benin', 'Botswana',
       'Burkina Faso', 'Burundi', 'Cameroon', 'Central African Republic',
       'Chad', 'Democratic Republic of Congo', 'Djibouti', 'Egypt',
       'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia',
       'Ghana', 'Guinea', 'Guinea-Bissau', 'Iran', 'Iraq', 'Israel',
       'Ivory Coast', 'Jordan', 'Kenya', 'Kuwait', 'Lebanon', 'Lesotho',
       'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
       'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Oman',
       'Palestine', 'Republic of Congo', 'Rwanda', 'Saudi Arabia',
       'Senegal', 'Sierra Leone', 'Somalia', 'South Africa',
       'South Sudan', 'Sudan', 'Syria', 'Tanzania', 'Togo', 'Tunisia',
       'Turkey', 'Uganda', 'United Arab Emirates', 'Yemen', 'Zambia',
       'Zimbabwe']

conflict_quads = ['Verbal Conflict', 'Material Conflict']

# COMMAND ----------

# DBTITLE 1,Select specified preprocessed data
gdelt2021 = preprocessedGDELT.filter(F.col('ActionGeo_FullName').isin(countries)) \
                             .filter(F.col('EventTimeDate') >= F.lit('2021-03-01'))

# add conflict, not binary column
gdelt2021 = gdelt2021.withColumn('Conflict', F.when(F.col('QuadClassString').isin(conflict_quads), True).otherwise(False))
gdelt2021.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Initial Report Variables
# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = gdelt2021.groupBy('ActionGeo_FullName','EventTimeDate','Conflict') \
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

gdelt2021Pandas = targetOutputPartitioned.toPandas()
gdelt2021Pandas.head()

# COMMAND ----------

# DBTITLE 1,Visualize Boxplots (conflict)
fig, ax = plt.subplots(figsize=(16, 24))
sns.boxplot(data=gdelt2021Pandas.loc[ gdelt2021Pandas['Conflict'] == True ],
            x='nArticles',
            y='ActionGeo_FullName',
            ax=ax)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16, 24))
sns.boxplot(data=gdelt2021Pandas.loc[ gdelt2021Pandas['Conflict'] == True ],
            x='EventReportValue',
            y='ActionGeo_FullName',
            ax=ax)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16, 24))
sns.boxplot(data=gdelt2021Pandas,
            x='GoldsteinReportValue',
            y='ActionGeo_FullName',
            ax=ax)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16, 24))
sns.boxplot(data=gdelt2021Pandas,
            x='ToneReportValue',
            y='ActionGeo_FullName',
            ax=ax)

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