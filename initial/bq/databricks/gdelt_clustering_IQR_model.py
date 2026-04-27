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

# Create quadclass: eventcode dictionary
cameo_quadclass_dict = {row['EventRootCodeString']:row['QuadClassString'] for row in quadClassCodes.collect()}
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

# create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName','EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Calculate Medians per Country per Date per QuadClass and Event Code
# Events: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                                 .withColumn('ERV_3d_median', median_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('ERV_60d_median', median_udf('ERV_60d_list'))

# Goldstein: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
                                                 .withColumn('GRV_1d_median', median_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_60d_list', F.collect_list('GoldsteinReportValue').over(rolling60d_window)) \
                                                 .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))

# Tone: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
                                                 .withColumn('TRV_1d_median', median_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_60d_list', F.collect_list('ToneReportValue').over(rolling60d_window)) \
                                                 .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))
# drop unnecessary columns
targetOutputPartitioned = targetOutputPartitioned.drop('ERV_3d_list','ERV_60d_list','GRV_1d_list',
                                                       'GRV_60d_list','TRV_1d_list','TRV_60d_list') \
                                                 .orderBy('EventTimeDate', ascending=False)

# verify output data
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(3).toPandas()

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

# Proportion Articles: 3d, 60d
# ERV_3d_median // ERV_60d_median
targetOutputTimelines = targetOutputPartitioned.withColumn('ERV_3d_3month_list', F.collect_list('ERV_3d_median').over(rolling3m_window)) \
                                               .withColumn('ERV_3d_3month_median', median_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_3month_sampleN',  F.size('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_quantile25', lowerQ_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_quantile75', upperQ_udf('ERV_3d_3month_list'))  \
                                               .withColumn('ERV_3d_3month_IQR', IQR_udf(F.col('ERV_3d_quantile25'), F.col('ERV_3d_quantile75')))  \
                                               .withColumn('ERV_60d_6month_list', F.collect_list('ERV_60d_median').over(rolling6m_window)) \
                                               .withColumn('ERV_60d_6month_median', median_udf('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_6month_sampleN', F.size('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_quantile25', lowerQ_udf('ERV_60d_6month_list'))  \
                                               .withColumn('ERV_60d_quantile75', upperQ_udf('ERV_60d_6month_list')) \
                                               .withColumn('ERV_60d_6month_IQR', IQR_udf(F.col('ERV_60d_quantile25'), F.col('ERV_60d_quantile75')))

# Goldstein: 1d, 60d
# GRV_1d_median // GRV_60d_median
targetOutputTimelines = targetOutputTimelines.withColumn('GRV_1d_3month_list', F.collect_list('GRV_1d_median').over(rolling3m_window)) \
                                             .withColumn('GRV_1d_3month_median', median_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_3month_sampleN',  F.size('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_quantile25', lowerQ_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_quantile75', upperQ_udf('GRV_1d_3month_list'))  \
                                             .withColumn('GRV_1d_3month_IQR', IQR_udf(F.col('GRV_1d_quantile25'), F.col('GRV_1d_quantile75')))  \
                                             .withColumn('GRV_60d_6month_list', F.collect_list('GRV_60d_median').over(rolling6m_window)) \
                                             .withColumn('GRV_60d_6month_median', median_udf('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_6month_sampleN', F.size('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_quantile25', lowerQ_udf('GRV_60d_6month_list'))  \
                                             .withColumn('GRV_60d_quantile75', upperQ_udf('GRV_60d_6month_list')) \
                                             .withColumn('GRV_60d_6month_IQR', IQR_udf(F.col('GRV_60d_quantile25'), F.col('GRV_60d_quantile75')))
# Tone: 1d, 60d
# TRV_1d_median // TRV_60d_median
targetOutputTimelines = targetOutputTimelines.withColumn('TRV_1d_3month_list', F.collect_list('TRV_1d_median').over(rolling3m_window)) \
                                             .withColumn('TRV_1d_3month_median', median_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_3month_sampleN', F.size('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_quantile25', lowerQ_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_quantile75', upperQ_udf('TRV_1d_3month_list'))  \
                                             .withColumn('TRV_1d_3month_IQR', IQR_udf(F.col('TRV_1d_quantile25'), F.col('TRV_1d_quantile75')))  \
                                             .withColumn('TRV_60d_6month_list', F.collect_list('TRV_60d_median').over(rolling6m_window)) \
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

# MAGIC %md
# MAGIC #### Detect Outliers

# COMMAND ----------

def get_max_outliers(median, upperQ, IQR):
  upper_limit = upperQ + (IQR*1.5)
  
  if median > upper_limit:
     return 'outlier (max)'
  else:
     return 'not outlier (max)'
    
def get_min_outliers(median, lowerQ, IQR):
  lower_limit = lowerQ - (IQR*1.5)
  
  if median < lower_limit:
     return 'outlier (min)'
  else:
     return 'not outlier (min)'

max_outliers_udf = F.udf(get_max_outliers, StringType())
min_outliers_udf = F.udf(get_min_outliers, StringType())

# COMMAND ----------

# identify outliers
assessVariableOutliers = targetOutputTimelines.withColumn('ERV_3d_outlier', max_outliers_udf('ERV_3d_median','ERV_3d_quantile75','ERV_3d_3month_IQR')) \
                                             .withColumn('ERV_60d_outlier', max_outliers_udf('ERV_60d_median','ERV_60d_quantile75','ERV_60d_6month_IQR')) \
                                             .withColumn('GRV_1d_outlier', min_outliers_udf('GRV_1d_median','GRV_1d_quantile25','GRV_1d_3month_IQR')) \
                                             .withColumn('GRV_60d_outlier', min_outliers_udf('GRV_60d_median','GRV_60d_quantile25','GRV_60d_6month_IQR')) \
                                             .withColumn('TRV_1d_outlier', min_outliers_udf('TRV_1d_median','TRV_1d_quantile25','TRV_1d_3month_IQR')) \
                                             .withColumn('TRV_60d_outlier', min_outliers_udf('TRV_60d_median','TRV_60d_quantile25','TRV_60d_6month_IQR'))
# drop unnecessary columns
assessVariableOutliers = assessVariableOutliers.drop('ERV_3d_month_list', 'ERV_60d_month_list',
                                                    'GRV_1d_month_list', 'GRV_60d_month_list',
                                                    'TRV_1d_month_list', 'TRV_60d_month_list')

# verify output data
assessVariableOutliers = assessVariableOutliers.orderBy('EventTimeDate', ascending=False)
print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
assessVariableOutliers.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles',
                              'ERV_3d_outlier','ERV_60d_outlier',
                              'GRV_1d_outlier','GRV_60d_outlier',
                              'TRV_1d_outlier','TRV_60d_outlier'
                             ).limit(2).toPandas()

# COMMAND ----------

assessVariableOutliers.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString',
                              'GRV_1d_median','GRV_1d_quantile25','GRV_1d_3month_IQR','GRV_1d_outlier'
                             ).filter(F.col('GRV_1d_outlier') == 'outlier (min)').limit(50).toPandas()

# COMMAND ----------

# DBTITLE 1,Map QuadClass back in to Dataframe
# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_quadclass_dict.items())])
assessVariableOutliers = assessVariableOutliers.withColumn('QuadClassString', mapping_expr[F.col('EventRootCodeString')])

# Confirm accurate output
assessVariableOutliers.select('QuadClassString', 'EventRootCodeString').dropDuplicates()#.show()
assessVariableOutliers.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Map UNICEF Regions in to Dataframe
# source: country column
unicef_countries = ["Afghanistan","Angola","Anguilla","Albania","United Arab Emirates","Argentina","Armenia","Antigua and Barbuda","Azerbaijan","Burundi","Benin","Burkina Faso","Bangladesh","Bulgaria","Bahrain","Bosnia-Herzegovina","Belarus","Belize","Bolivia","Brazil","Barbados","Bhutan","Botswana","Central African Republic","Chile","China","Côte d'Ivoire","Cameroon","DRC","ROC","Colombia","Comoros","Cape Verde","Costa Rica","Cuba","Djibouti","Dominica","Dominican Republic","Algeria","Ecuador","Egypt, Arab Rep.","Eritrea","Western Sahara","Ethiopia","Pacific Islands (Fiji)","Micronesia","Gabon","Georgia","Ghana","Guinea Conakry","Gambia, The","Guinea-Bissau","Equatorial Guinea","Grenada","Guatemala","Guyana","Honduras","Croatia","Haiti","Indonesia","India","Iran","Iraq","Jamaica","Jordan","Kazakhstan","Kenya","Kyrgyzstan","Cambodia","Kiribati","Saint Kitts and Nevis","Kuwait","Laos","Lebanon","Liberia","Libya","Saint Lucia","Sri Lanka","Lesotho","Morocco","Moldova","Madagascar","Maldives","Mexico","Marshall Islands","Macedonia","Mali","Myanmar","Montenegro","Mongolia","Mozambique","Mauritania","Montserrat","Malawi","Malaysia","Namibia","Niger","Nigeria","Nicaragua","Nepal ","Nauru","Oman","Pakistan","Panama","Peru","Philippines","Palau","Papua New Guinea","Korea, North","Paraguay","Palestine","Qatar","Kosovo","Romania","Rwanda","Saudi Arabia","Sudan","Senegal","Solomon Islands","Sierra Leone","El Salvador","Somalia","Serbia","South Sudan","Sao Tome and Principe","Suriname","Eswatini","Syria","Turks and Caicos","Chad","Togo","Thailand","Tajikistan","Tokelau","Turkmenistan","Timor-Leste","Tonga","Trinidad and Tobago", "Tunisia","Turkey","Tuvalu",
"Tanzania","Uganda","Ukraine","Uruguay","Uzbekistan","Saint Vincent and the Grenadines","Venezuela","British Virgin Islands","Vietnam","Vanuatu","Samoa","Yemen","South Africa","Zambia","Zimbabwe"]

# source: unicef region column
unicef_region_ordered = ["ROSA", "ESARO", "LACRO", "ECARO", "MENARO", "LACRO", "ECARO", "LACRO", "ECARO", "ESARO", "WCARO", "WCARO", "ROSA", "ECARO", "MENARO", "WCARO", "ECARO", "LACRO", "LACRO", "LACRO", "LACRO", "ROSA", "ESARO", "WCARO", "LACRO", "EAPRO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "ESARO", "WCARO", "LACRO", "LACRO", "MENARO", "LACRO", "LACRO", "MENARO", "LACRO", "MENARO", "ESARO", "MENARO", "ESARO", "EAPRO", "EAPRO", "WCARO", "ECARO", "WCARO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "LACRO", "LACRO", "LACRO", "ECARO", "LACRO", "EAPRO", "ROSA", "MENARO", "MENARO", "LACRO", "MENARO", "ECARO", "ESARO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "EAPRO", "MENARO", "WCARO", "MENARO", "LACRO", "ROSA", "ESARO", "MENARO", "ECARO", "ESARO", "ROSA", "LACRO", "EAPRO", "ECARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ESARO", "WCARO", "LACRO", "ESARO", "EAPRO", "ESARO", "WCARO", "WCARO", "LACRO", "ROSA", "EAPRO", "MENARO", "ROSA", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "EAPRO", "LACRO", "MENARO", "MENARO", "ECARO", "ECARO", "ESARO", "MENARO", "MENARO", "WCARO", "EAPRO", "WCARO", "LACRO", "ESARO", "ECARO", "ESARO", "WCARO", "LACRO", "ESARO", "MENARO", "LACRO", "WCARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "ECARO", "EAPRO", "ESARO", "ESARO", "ECARO", "LACRO", "ECARO", "LACRO", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "MENARO", "ESARO", "ESARO", "ESARO"]

# Create Country: Country Region dictionary
country_cluster_dict = dict(zip(unicef_countries, unicef_region_ordered))

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*country_cluster_dict.items())])
assessVariableOutliers = assessVariableOutliers.withColumn('UNICEF_regions', mapping_expr[F.col('ActionGeo_FullName')])
assessVariableOutliers.limit(1).toPandas()

# COMMAND ----------

assessVariableOutliers.columns

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

cols = ['ActionGeo_FullName','UNICEF_regions','EventTimeDate','QuadClassString','EventRootCodeString','nArticles',
       'EventReportValue','ERV_3d_median','ERV_3d_outlier','ERV_60d_median','ERV_60d_outlier',
       'GoldsteinReportValue','GRV_1d_median','GRV_1d_outlier','GRV_60d_median','GRV_60d_outlier',
       'ToneReportValue','TRV_1d_median','TRV_1d_outlier','TRV_60d_median','TRV_60d_outlier']
assessVariableOutliersSelect = assessVariableOutliers.select(cols)

# select only february and beyond
assessVariableOutliersSelect = assessVariableOutliersSelect.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
powerBI = assessVariableOutliersSelect.filter(F.col('EventTimeDate') >= F.lit('2021-03-01'))
powerBI.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Clean Up Columns for Output
# store to CSV
TEMPORARY_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/shorter_IQR_alertsystem_regions_10may2021"
DESIRED_BI_TARGET="dbfs:/Filestore/tables/tmp/gdelt/shorter_IQR_alertsystem_regions_10may2021.csv"

powerBI.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_BI_TARGET)
temporaryPoweBI_csv = os.path.join(TEMPORARY_BI_TARGET, dbutils.fs.ls(TEMPORARY_BI_TARGET)[3][1])
dbutils.fs.cp(temporaryPoweBI_csv, DESIRED_BI_TARGET)