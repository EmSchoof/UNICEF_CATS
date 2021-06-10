# Databricks notebook source
# MAGIC %md # Query Data from Azure SQL DB

# COMMAND ----------

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

jdbcHostname = "uniictdatdb01.database.windows.net"
jdbcDatabase = "uni-oia-cats-datastgdb"
jdbcPort = 1433

# COMMAND ----------

# DBTITLE 1,Create the JDBC URL and specify connection properties
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)
connectionProperties = {
  "user" : dbutils.preview.secret.get(scope = "emops-secrets", key = "emops-sql-db-user"),
  "password" : dbutils.preview.secret.get(scope = "emops-secrets", key = "emops-sql-db-pwd"),
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# COMMAND ----------

# DBTITLE 1,Get all data from table in SQL DB
table_df = spark.read.jdbc(url=jdbcUrl, table="cats.vw_ContextTrendAlert", properties=connectionProperties)

# COMMAND ----------

table_df = table_df.withColumn('GoldsteinScale', table_df['GoldsteinScale'].cast(IntegerType()))
table_df = table_df.withColumn('MentionDocTone', table_df['MentionDocTone'].cast(IntegerType()))

# COMMAND ----------

display(table_df)

# COMMAND ----------

# DBTITLE 1,Calculate Medians and Target Variables
# create function to calculate median
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = table_df.groupBy('ActionGeo_CountryName','EventDate','EventRootCode') \
                         .agg(F.avg('Confidence').alias('avgConfidence'),
                              F.collect_list('GoldsteinScale').alias('GoldsteinList'),
                              F.collect_list('MentionDocTone').alias('ToneList'),
                              F.count(F.lit(1)).alias('nArticles')) \
                          .withColumn('GoldsteinReportValue', median_udf('GoldsteinList')) \
                          .withColumn('ToneReportValue', median_udf('ToneList')) \
                          .drop('GoldsteinList','ToneList')

# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_CountryName','EventDate').orderBy('EventDate')

# get daily distribution of articles for each Event Code string within Window
targetOutputPartitioned = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Rolling Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# --- for Creating Metrics ---

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_CountryName','EventRootCode').orderBy(F.col('EventDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_CountryName','EventRootCode').orderBy(F.col('EventDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Calculate Medians per Country per Date per QuadClass and Event Code
# Events: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                                 .withColumn('ERV_3d_Median', median_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('ERV_60d_Median', median_udf('ERV_60d_list'))

# Goldstein: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_3d_list', F.collect_list('GoldsteinReportValue').over(rolling3d_window)) \
                                                 .withColumn('GRV_3d_Median', median_udf('GRV_3d_list'))  \
                                                 .withColumn('GRV_60d_list', F.collect_list('GoldsteinReportValue').over(rolling60d_window)) \
                                                 .withColumn('GRV_60d_Median', median_udf('GRV_60d_list'))

# Tone: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_3d_list', F.collect_list('ToneReportValue').over(rolling3d_window)) \
                                                 .withColumn('TRV_3d_Median', median_udf('TRV_3d_list'))  \
                                                 .withColumn('TRV_60d_list', F.collect_list('ToneReportValue').over(rolling60d_window)) \
                                                 .withColumn('TRV_60d_Median', median_udf('TRV_60d_list'))
# drop unnecessary columns
targetOutputPartitioned = targetOutputPartitioned.drop('ERV_3d_list','ERV_60d_list','GRV_3d_list',
                                                       'GRV_60d_list','TRV_3d_list','TRV_60d_list') \
                                                 .orderBy('EventDate', ascending=False)

# verify output data
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(3).toPandas()

# COMMAND ----------

# DBTITLE 1,Create IQR Time Windows for 12 and 24 months
# --- Windows for Evaluation Periods ---

# create a 3 month Window, 12 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3m_window = Window.partitionBy('ActionGeo_CountryName','EventRootCode').orderBy(F.col('EventDate').cast('timestamp').cast('long')).rangeBetween(-days(365), 0)

# create a 6 month Window, 24 months previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling6m_window = Window.partitionBy('ActionGeo_CountryName','EventRootCode').orderBy(F.col('EventDate').cast('timestamp').cast('long')).rangeBetween(-days(730), 0)

# COMMAND ----------

# IQR UDF functions
lowerQ_udf = F.udf(lambda x: float(np.quantile(x, 0.25)), FloatType())
upperQ_udf = F.udf(lambda x: float(np.quantile(x, 0.75)), FloatType())
IQR_udf = F.udf(lambda lowerQ, upperQ: (upperQ - lowerQ), FloatType())

# COMMAND ----------

# Proportion Articles: 3d, 60d
# ERV_3d_Median // ERV_60d_Median
targetOutputTimelines = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('ERV_3d_Median').over(rolling3m_window)) \
                                               .withColumn('ERV_3d_Median', median_udf('ERV_3d_list'))  \
                                               .withColumn('ERV_3d_sampleN',  F.size('ERV_3d_list'))  \
                                               .withColumn('ERV_3d_quantile25', lowerQ_udf('ERV_3d_list'))  \
                                               .withColumn('ERV_3d_quantile75', upperQ_udf('ERV_3d_list'))  \
                                               .withColumn('ERV_3d_IQR', IQR_udf(F.col('ERV_3d_quantile25'), F.col('ERV_3d_quantile75')))  \
                                               .withColumn('ERV_60d_list', F.collect_list('ERV_60d_Median').over(rolling6m_window)) \
                                               .withColumn('ERV_60d_Median', median_udf('ERV_60d_list'))  \
                                               .withColumn('ERV_60d_sampleN', F.size('ERV_60d_list'))  \
                                               .withColumn('ERV_60d_quantile25', lowerQ_udf('ERV_60d_list'))  \
                                               .withColumn('ERV_60d_quantile75', upperQ_udf('ERV_60d_list')) \
                                               .withColumn('ERV_60d_IQR', IQR_udf(F.col('ERV_60d_quantile25'), F.col('ERV_60d_quantile75')))

# Goldstein: 3d, 60d
# GRV_3d_Median // GRV_60d_Median
targetOutputTimelines = targetOutputTimelines.withColumn('GRV_3d_list', F.collect_list('GRV_3d_Median').over(rolling3m_window)) \
                                             .withColumn('GRV_3d_Median', median_udf('GRV_3d_list'))  \
                                             .withColumn('GRV_3d_sampleN',  F.size('GRV_3d_list'))  \
                                             .withColumn('GRV_3d_quantile25', lowerQ_udf('GRV_3d_list'))  \
                                             .withColumn('GRV_3d_quantile75', upperQ_udf('GRV_3d_list'))  \
                                             .withColumn('GRV_3d_IQR', IQR_udf(F.col('GRV_3d_quantile25'), F.col('GRV_3d_quantile75')))  \
                                             .withColumn('GRV_60d_list', F.collect_list('GRV_60d_Median').over(rolling6m_window)) \
                                             .withColumn('GRV_60d_Median', median_udf('GRV_60d_list'))  \
                                             .withColumn('GRV_60d_sampleN', F.size('GRV_60d_list'))  \
                                             .withColumn('GRV_60d_quantile25', lowerQ_udf('GRV_60d_list'))  \
                                             .withColumn('GRV_60d_quantile75', upperQ_udf('GRV_60d_list')) \
                                             .withColumn('GRV_60d_IQR', IQR_udf(F.col('GRV_60d_quantile25'), F.col('GRV_60d_quantile75')))
# Tone: 3d, 60d
# TRV_3d_Median // TRV_60d_Median
targetOutputTimelines = targetOutputTimelines.withColumn('TRV_3d_list', F.collect_list('TRV_3d_Median').over(rolling3m_window)) \
                                             .withColumn('TRV_3d_Median', median_udf('TRV_3d_list'))  \
                                             .withColumn('TRV_3d_sampleN', F.size('TRV_3d_list'))  \
                                             .withColumn('TRV_3d_quantile25', lowerQ_udf('TRV_3d_list'))  \
                                             .withColumn('TRV_3d_quantile75', upperQ_udf('TRV_3d_list'))  \
                                             .withColumn('TRV_3d_IQR', IQR_udf(F.col('TRV_3d_quantile25'), F.col('TRV_3d_quantile75')))  \
                                             .withColumn('TRV_60d_list', F.collect_list('TRV_60d_Median').over(rolling6m_window)) \
                                             .withColumn('TRV_60d_Median', median_udf('TRV_60d_list'))  \
                                             .withColumn('TRV_60d_sampleN', F.size('TRV_60d_list'))  \
                                             .withColumn('TRV_60d_quantile25', lowerQ_udf('TRV_60d_list'))  \
                                             .withColumn('TRV_60d_quantile75', upperQ_udf('TRV_60d_list')) \
                                             .withColumn('TRV_60d_IQR', IQR_udf(F.col('TRV_60d_quantile25'), F.col('TRV_60d_quantile75')))
      
# verify output data
targetOutputTimelines = targetOutputTimelines.orderBy('EventDate', ascending=False)
print((targetOutputTimelines.count(), len(targetOutputTimelines.columns)))
targetOutputTimelines.limit(3).toPandas()

# COMMAND ----------

# DBTITLE 1,Detect Outliers
def get_upper_outliers(median, upperQ, IQR):
  mild_upper_outlier = upperQ + (IQR*1.5)
  extreme_upper_outlier = upperQ + (IQR*3)

  if (median >= mild_upper_outlier) and (median < extreme_upper_outlier):
     return 'mild outlier'
  elif (median >= extreme_upper_outlier):
    return 'extreme outlier'
  else:
     return 'not outlier'

def get_lower_outliers(median, lowerQ, IQR):
  mild_lower_outlier = lowerQ - (IQR*1.5)
  extreme_lower_outlier = lowerQ - (IQR*3)

  if (median <= mild_lower_outlier) and (median > extreme_lower_outlier):
     return 'mild outlier'
  elif (median <= extreme_lower_outlier):
    return 'extreme outlier'
  else:
     return 'not outlier'

max_outliers_udf = F.udf(get_upper_outliers, StringType())
min_outliers_udf = F.udf(get_lower_outliers, StringType())

# COMMAND ----------

# identify outliers
assessVariableOutliers = targetOutputTimelines.withColumn('ERV_3d_MaxOutlier', max_outliers_udf('ERV_3d_Median','ERV_3d_quantile75','ERV_3d_IQR')) \
                                             .withColumn('ERV_60d_MaxOutlier', max_outliers_udf('ERV_60d_Median','ERV_60d_quantile75','ERV_60d_IQR')) \
                                             .withColumn('GRV_3d_MinOutlier', min_outliers_udf('GRV_3d_Median','GRV_3d_quantile25','GRV_3d_IQR')) \
                                             .withColumn('GRV_60d_MinOutlier', min_outliers_udf('GRV_60d_Median','GRV_60d_quantile25','GRV_60d_IQR')) \
                                             .withColumn('TRV_3d_MinOutlier', min_outliers_udf('TRV_3d_Median','TRV_3d_quantile25','TRV_3d_IQR')) \
                                             .withColumn('TRV_60d_MinOutlier', min_outliers_udf('TRV_60d_Median','TRV_60d_quantile25','TRV_60d_IQR'))
# drop unnecessary columns
assessVariableOutliers = assessVariableOutliers.drop('ERV_3d_list', 'ERV_60d_list',
                                                    'GRV_3d_list', 'GRV_60d_list',
                                                    'TRV_3d_list', 'TRV_60d_list')

# verify output data
assessVariableOutliers = assessVariableOutliers.orderBy('EventDate', ascending=False)
print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
assessVariableOutliers.select('ActionGeo_CountryName','EventDate','EventRootCode','nArticles',
                              'ERV_3d_MaxOutlier','ERV_60d_MaxOutlier',
                              'GRV_3d_MinOutlier','GRV_60d_MinOutlier',
                              'TRV_3d_MinOutlier','TRV_60d_MinOutlier'
                             ).limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Add back Regions and Quad Class 
# UNICEF Regions
regionsDF = table_df.select('ActionGeo_CountryName', 'UNICEF_regions').dropDuplicates()
regions_dict = {row['ActionGeo_CountryName']:row['UNICEF_regions'] for row in regionsDF.collect()}
regions_mapping_expr = F.create_map([F.lit(x) for x in chain(*regions_dict.items())])

# Quad Class
quadClass = table_df.select('QuadClass', 'EventRootCode').dropDuplicates()
quadClass_dict = {row['EventRootCode']:row['QuadClass'] for row in quadClass.collect()}
quadClass_mapping_expr = F.create_map([F.lit(x) for x in chain(*quadClass_dict.items())])

# COMMAND ----------

# Map dictionary over df to create string column
assessVariableOutliers = assessVariableOutliers.withColumn('UNICEF_regions', regions_mapping_expr[F.col('ActionGeo_CountryName')])
assessVariableOutliers = assessVariableOutliers.withColumn('QuadClass', quadClass_mapping_expr[F.col('EventRootCode')])
assessVariableOutliers.limit(10).toPandas()

# COMMAND ----------

min_date, max_date = assessVariableOutliers.select(F.min('EventDate'), F.max('EventDate')).first()
min_date, max_date

# COMMAND ----------

# DBTITLE 1,Get Data for PowerBI
cols = ['ActionGeo_CountryName','UNICEF_regions','EventDate','QuadClass','EventRootCode','nArticles',
       'EventReportValue','ERV_3d_Median','ERV_3d_MaxOutlier','ERV_60d_Median','ERV_60d_MaxOutlier',
       'GoldsteinReportValue','GRV_3d_Median','GRV_3d_MinOutlier','GRV_60d_Median','GRV_60d_MinOutlier',
       'ToneReportValue','TRV_3d_Median','TRV_3d_MinOutlier','TRV_60d_Median','TRV_60d_MinOutlier']

assessVariableOutliersSelect = assessVariableOutliers.select(cols)
assessVariableOutliersSelect.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Output to CSV for Data Upload to PowerBI
powerBI = assessVariableOutliersSelect.filter(F.col('EventDate') >= F.lit('2021-02-01'))
powerBI.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/FileStore/tables/tmp/gdelt/msql_cats_dashboard_june2021.csv')