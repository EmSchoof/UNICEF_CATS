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
# MAGIC #### Count Event Articles by Country

# COMMAND ----------

countCountryArticles = preprocessedGDELTcon40.select('ActionGeo_FullName','nArticles').groupBy('ActionGeo_FullName').sum().orderBy('sum(nArticles)')
countCountryArticles.show(), countCountryArticles.tail()

# COMMAND ----------

# DBTITLE 1,Calculate Minimum Sample Size for Statistically Comparing Medians and IQRs
from math import sqrt
from statsmodels.stats.power import TTestIndPower
  
#calculation of effect size
# size of samples in pilot study
n1, n2 = 4, 4
  
# variance of samples in pilot study
s1, s2 = 5**2, 5**2
  
# calculate the pooled standard deviation 
# (Cohen's d)
s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
  
# means of the samples
u1, u2 = 90, 85
  
# calculate the effect size
d = (u1 - u2) / s
print(f'Effect size: {d}')
  
# factors for power analysis
alpha = 0.05
power = 0.8
  
# perform power analysis to find sample size 
# for given effect
obj = TTestIndPower()
n = obj.solve_power(effect_size=d, alpha=alpha, power=power, 
                    ratio=1, alternative='two-sided')
  
print('Sample size/Number needed in each group: {:.3f}'.format(n))

# COMMAND ----------

# MAGIC %md 
# MAGIC Since the Central Limit Theorem 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create Country Clusters (Based on UNICEF's "IOS Codes and Regions")

# COMMAND ----------

# source: country column
unicef_countries = ["Afghanistan","Angola","Anguilla","Albania","United Arab Emirates","Argentina","Armenia","Antigua and Barbuda","Azerbaijan","Burundi","Benin","Burkina Faso","Bangladesh","Bulgaria","Bahrain","Bosnia-Herzegovina","Belarus","Belize","Bolivia","Brazil","Barbados","Bhutan","Botswana","Central African Republic","Chile","China","Côte d'Ivoire","Cameroon","DRC","ROC","Colombia","Comoros","Cape Verde","Costa Rica","Cuba","Djibouti","Dominica","Dominican Republic","Algeria","Ecuador","Egypt, Arab Rep.","Eritrea","Western Sahara","Ethiopia","Pacific Islands (Fiji)","Micronesia","Gabon","Georgia","Ghana","Guinea Conakry","Gambia, The","Guinea-Bissau","Equatorial Guinea","Grenada","Guatemala","Guyana","Honduras","Croatia","Haiti","Indonesia","India","Iran","Iraq","Jamaica","Jordan","Kazakhstan","Kenya","Kyrgyzstan","Cambodia","Kiribati","Saint Kitts and Nevis","Kuwait","Laos","Lebanon","Liberia","Libya","Saint Lucia","Sri Lanka","Lesotho","Morocco","Moldova","Madagascar","Maldives","Mexico","Marshall Islands","Macedonia","Mali","Myanmar","Montenegro","Mongolia","Mozambique","Mauritania","Montserrat","Malawi","Malaysia","Namibia","Niger","Nigeria","Nicaragua","Nepal ","Nauru","Oman","Pakistan","Panama","Peru","Philippines","Palau","Papua New Guinea","Korea, North","Paraguay","Palestine","Qatar","Kosovo","Romania","Rwanda","Saudi Arabia","Sudan","Senegal","Solomon Islands","Sierra Leone","El Salvador","Somalia","Serbia","South Sudan","Sao Tome and Principe","Suriname","Eswatini","Syria","Turks and Caicos","Chad","Togo","Thailand","Tajikistan","Tokelau","Turkmenistan","Timor-Leste","Tonga","Trinidad and Tobago", "Tunisia","Turkey","Tuvalu",
"Tanzania","Uganda","Ukraine","Uruguay","Uzbekistan","Saint Vincent and the Grenadines","Venezuela","British Virgin Islands","Vietnam","Vanuatu","Samoa","Yemen","South Africa","Zambia","Zimbabwe"]

# source: unicef region column
unicef_region_ordered = ["ROSA", "ESARO", "LACRO", "ECARO", "MENARO", "LACRO", "ECARO", "LACRO", "ECARO", "ESARO", "WCARO", "WCARO", "ROSA", "ECARO", "MENARO", "WCARO", "ECARO", "LACRO", "LACRO", "LACRO", "LACRO", "ROSA", "ESARO", "WCARO", "LACRO", "EAPRO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "ESARO", "WCARO", "LACRO", "LACRO", "MENARO", "LACRO", "LACRO", "MENARO", "LACRO", "MENARO", "ESARO", "MENARO", "ESARO", "EAPRO", "EAPRO", "WCARO", "ECARO", "WCARO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "LACRO", "LACRO", "LACRO", "ECARO", "LACRO", "EAPRO", "ROSA", "MENARO", "MENARO", "LACRO", "MENARO", "ECARO", "ESARO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "EAPRO", "MENARO", "WCARO", "MENARO", "LACRO", "ROSA", "ESARO", "MENARO", "ECARO", "ESARO", "ROSA", "LACRO", "EAPRO", "ECARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ESARO", "WCARO", "LACRO", "ESARO", "EAPRO", "ESARO", "WCARO", "WCARO", "LACRO", "ROSA", "EAPRO", "MENARO", "ROSA", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "EAPRO", "LACRO", "MENARO", "MENARO", "ECARO", "ECARO", "ESARO", "MENARO", "MENARO", "WCARO", "EAPRO", "WCARO", "LACRO", "ESARO", "ECARO", "ESARO", "WCARO", "LACRO", "ESARO", "MENARO", "LACRO", "WCARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "ECARO", "EAPRO", "ESARO", "ESARO", "ECARO", "LACRO", "ECARO", "LACRO", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "MENARO", "ESARO", "ESARO", "ESARO"]


# Create Country: Country Region dictionary
country_cluster_dict = dict(zip(unicef_countries, unicef_region_ordered))
country_cluster_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*country_cluster_dict.items())])
clusteredCountries = preprocessedGDELTcon40.withColumn('UNICEF_regions', mapping_expr[F.col('ActionGeo_FullName')])
clusteredCountries.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Separate Data for Countries with and without a Cluster
# with
countriesWithCluster = clusteredCountries.filter(~F.col('UNICEF_regions').isNull())
print(countriesWithCluster.count())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create Target Variables
# MAGIC - Calculate IQR for clusters 
# MAGIC - Calculate IQR and IQR-Outlier for all countries
# MAGIC - Calculate Power for Statistics

# COMMAND ----------

# DBTITLE 1,UDF Functions
sampleN_udf = F.udf(lambda arr: len(arr), FloatType())
lowerQ_udf = F.udf(lambda x: float(np.quantile(x, 0.25)), FloatType())
median_udf = F.udf(lambda x: float(np.quantile(x, 0.5)), FloatType())
upperQ_udf = F.udf(lambda x: float(np.quantile(x, 0.75)), FloatType())
IQR_udf = F.udf(lambda lowerQ, upperQ: (upperQ - lowerQ), FloatType())
quantileDeviation_udf = F.udf(lambda IQR: IQR/2, FloatType())

# COMMAND ----------

# DBTITLE 1,Create Rolling Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### No Cluster (regular protocol)

# COMMAND ----------

F.skewness('wERA_3d'),
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

# COMMAND ----------

# DBTITLE 1,Create Initial Report Variables
# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName','EventTimeDate').orderBy('EventTimeDate')

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
targetOutput = clusteredCountries.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString') \
                                     .agg(F.avg('Confidence').alias('avgConfidence'),
                                          F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                          F.avg('MentionDocTone').alias('ToneReportValue'),
                                          F.sum('nArticles').alias('nArticles'))

# get daily distribution of articles for each Event Code string within Window
targetOutputPartitioned = targetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create IQR Values per Country per Date per Event Code
# Events: 3d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                                 .withColumn('ERV_3d_sampleN', sampleN_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_3d_quantile25', lowerQ_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_3d_median', median_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_3d_quantile75', upperQ_udf('ERV_3d_list'))  \
                                                 .withColumn('ERV_3d_IQR', IQR_udf(F.col('ERV_3d_quantile25'), F.col('ERV_3d_quantile75')))  \
                                                 .withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('ERV_60d_sampleN', sampleN_udf('ERV_60d_list'))  \
                                                 .withColumn('ERV_60d_quantile25', lowerQ_udf('ERV_60d_list'))  \
                                                 .withColumn('ERV_60d_median', median_udf('ERV_60d_list'))  \
                                                 .withColumn('ERV_60d_quantile75', upperQ_udf('ERV_60d_list')) \
                                                 .withColumn('ERV_60d_IQR', IQR_udf(F.col('ERV_60d_quantile25'), F.col('ERV_60d_quantile75')))

# Goldstein: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
                                                 .withColumn('GRV_1d_sampleN', sampleN_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_1d_quantile25', lowerQ_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_1d_median', median_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_1d_quantile75', upperQ_udf('GRV_1d_list'))  \
                                                 .withColumn('GRV_1d_IQR', IQR_udf(F.col('GRV_1d_quantile25'), F.col('GRV_1d_quantile75')))  \
                                                 .withColumn('GRV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('GRV_60d_sampleN', sampleN_udf('GRV_60d_list'))  \
                                                 .withColumn('GRV_60d_quantile25', lowerQ_udf('GRV_60d_list'))  \
                                                 .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))  \
                                                 .withColumn('GRV_60d_quantile75', upperQ_udf('GRV_60d_list')) \
                                                 .withColumn('GRV_60d_IQR', IQR_udf(F.col('GRV_60d_quantile25'), F.col('GRV_60d_quantile75')))

# Tone: 1d, 60d
targetOutputPartitioned = targetOutputPartitioned.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
                                                 .withColumn('TRV_1d_sampleN', sampleN_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_1d_quantile25', lowerQ_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_1d_median', median_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_1d_quantile75', upperQ_udf('TRV_1d_list'))  \
                                                 .withColumn('TRV_1d_IQR', IQR_udf(F.col('TRV_1d_quantile25'), F.col('TRV_1d_quantile75')))  \
                                                 .withColumn('TRV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                                 .withColumn('TRV_60d_sampleN', sampleN_udf('TRV_60d_list'))  \
                                                 .withColumn('TRV_60d_quantile25', lowerQ_udf('TRV_60d_list'))  \
                                                 .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))  \
                                                 .withColumn('TRV_60d_quantile75', upperQ_udf('TRV_60d_list')) \
                                                 .withColumn('TRV_60d_IQR', IQR_udf(F.col('TRV_60d_quantile25'), F.col('TRV_60d_quantile75')))
# select output columns
targetOutputPartitioned = targetOutputPartitioned.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','avgConfidence','nArticles',
                                                         'EventReportValue','ERV_3d_list','ERV_3d_sampleN','ERV_3d_median','ERV_3d_IQR',
                                                         'ERV_60d_list','ERV_60d_sampleN','ERV_60d_median','ERV_60d_IQR',
                                                         'GoldsteinReportValue','GRV_1d_list','GRV_1d_sampleN','GRV_1d_median','GRV_1d_IQR',
                                                         'GRV_60d_list','GRV_60d_sampleN','GRV_60d_median','GRV_60d_IQR',
                                                         'ToneReportValue','TRV_1d_list','TRV_1d_sampleN','TRV_1d_median','TRV_1d_IQR',
                                                         'TRV_60d_list','TRV_60d_sampleN','TRV_60d_median','TRV_60d_IQR')

# verify output data
print((targetOutputPartitioned.count(), len(targetOutputPartitioned.columns)))
targetOutputPartitioned.limit(3).toPandas()

# COMMAND ----------

# DBTITLE 1,Add Power for Non-Normally Distributed Variables
# MAGIC %md
# MAGIC #### [Power analysis](https://webpower.psychstat.org/wiki/_media/grant/du-zhang-yuan-2017.pdf) 
# MAGIC 
# MAGIC - Power Analysis is widely used for sample size determination (e.g., Cohen，1988). With appropriate power analysis, an adequate but not “too large” sample size is determined to detect an existing effect.
# MAGIC - The **Monte Carlo simulation** is a method can flexibly take into account non-normality in one-sample t-test, two-sample t-test, and paired t-test, and unequal variances in two-sample t-test.

# COMMAND ----------

targetOutputPartitioned.columns

# COMMAND ----------

T = targetOutputPartitioned.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString', 'ERV_3d_median').withColumn('ERV_3d_variance', F.variance('ERV_3d_median'))


# COMMAND ----------

# DBTITLE 1,Clean Up Dataframe and Add Variance
select_cols =['ActionGeo_FullName','EventTimeDate','EventRootCodeString','avgConfidence','nArticles','EventReportValue','ERV_3d_median','ERV_3d_qDeviation', 'ERV_60d_median','ERV_60d_qDeviation','GoldsteinReportValue','GRV_1d_median','GRV_1d_qDeviation','GRV_60d_median','GRV_60d_qDeviation','ToneReportValue', 'TRV_1d_median','TRV_1d_qDeviation','TRV_60d_median','TRV_60d_qDeviation']

targetOutputStatsMetrics = targetOutputPartitioned.select(select_cols) \
                                                  .groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString') \
                                                  .agg( 
                                                        F.count(F.lit(1)).alias('n_observations')
                                                  )

# COMMAND ----------

# MAGIC %md
# MAGIC #### If Country in Cluster, replace Country IQR Values with Cluster IQR values (for date, eventcode)

# COMMAND ----------

# MAGIC %md 
# MAGIC # define columns for replacement
# MAGIC cols_to_update = ['ERV_3d_qDeviation','ERV_60d_qDeviation','ERV_3d_median','ERV_60d_median',
# MAGIC                   'GRV_1d_qDeviation','GRV_60d_qDeviation','GRV_1d_median','GRV_60d_median',
# MAGIC                   'TRV_1d_qDeviation', 'TRV_60d_qDeviation','TRV_1d_median', 'TRV_60d_median']
# MAGIC 
# MAGIC # replace Median and Quantile Deviation values for countries with a cluster
# MAGIC targetOutputPartitioned = targetOutputPartitioned.withColumn('id', F.monotonically_increasing_id())
# MAGIC countryClustersUpdate = targetOutputPartitioned.alias('a') \
# MAGIC                         .join(clustersQRs_exploded.alias('b'), on=['ActionGeo_FullName','EventTimeDate','EventRootCodeString'], how='left') \
# MAGIC                         .select(
# MAGIC                             *[
# MAGIC                                 [ F.when(~F.isnull(F.col('b.UNICEF_regions')), F.col('b.{}'.format(c))
# MAGIC                                     ).otherwise(F.col('a.{}'.format(c))).alias(c)
# MAGIC                                     for c in cols_to_update
# MAGIC                                 ]
# MAGIC                             ]
# MAGIC                         ) \
# MAGIC                         .withColumn('id', F.monotonically_increasing_id())
# MAGIC 
# MAGIC # merge dataframes
# MAGIC #all_cols = targetOutputPartitioned.drop(*cols_to_update).columns
# MAGIC assessVariableOutliers = targetOutputPartitioned.select(*all_cols).join(countryClustersUpdate, on='id', how='outer').drop('id')
# MAGIC 
# MAGIC # verify output data
# MAGIC print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
# MAGIC assessVariableOutliers.limit(3).toPandas()

# COMMAND ----------

#assessVariableOutliers.limit(20).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Detect Outliers

# COMMAND ----------

# define outlier UDF function
outliers_udf = F.udf(lambda val, median, qdev: 'normal' if np.abs(val - median) >= (qdev*2.2) else 'outlier', StringType())

# identify outliers
assessVariableOutliers = targetOutputPartitioned.withColumn('ERV_3d_outlier', outliers_udf(F.col('EventReportValue'), F.col('ERV_3d_median'), F.col('ERV_3d_qDeviation'))) \
                                                 .withColumn('ERV_60d_outlier', outliers_udf(F.col('EventReportValue'), F.col('ERV_60d_median'), F.col('ERV_60d_qDeviation'))) \
                                                 .withColumn('GRV_1d_outlier', outliers_udf(F.col('GoldsteinReportValue'), F.col('GRV_1d_median'), F.col('GRV_1d_qDeviation'))) \
                                                 .withColumn('GRV_60d_outlier', outliers_udf(F.col('GoldsteinReportValue'), F.col('GRV_60d_median'), F.col('GRV_60d_qDeviation'))) \
                                                 .withColumn('TRV_1d_outlier', outliers_udf(F.col('ToneReportValue'), F.col('TRV_1d_median'), F.col('TRV_1d_qDeviation'))) \
                                                 .withColumn('TRV_60d_outlier', outliers_udf(F.col('ToneReportValue'), F.col('TRV_60d_median'), F.col('TRV_60d_qDeviation')))
# verify output data
print((assessVariableOutliers.count(), len(assessVariableOutliers.columns)))
assessVariableOutliers.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles',
                              'ERV_3d_outlier','ERV_60d_outlier',
                              'GRV_1d_outlier','GRV_60d_outlier',
                              'TRV_1d_outlier','TRV_60d_outlier'
                             ).limit(20).toPandas()

# COMMAND ----------

assessVariableOutliers.printSchema()

# COMMAND ----------

#assessVariableOutliers = assessVariableOutliers.withColumn('UNICEF_regions', F.when(F.col('UNICEF_regions') == F.lit(None), '').otherwise(F.col('UNICEF_regions')))
assessVariableOutliers.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/ALL_IQR_alertsystem_16april2021.csv')

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Explore Output

# COMMAND ----------

# Test output
AFG = assessVariableOutliers.filter(F.col('ActionGeo_FullName') == 'Afghanistan')
AFG.limit(20).toPandas()

# COMMAND ----------

groupCols = ['EventRootCodeString',
             'ERV_3d_outlier','ERV_60d_outlier']
AFG_E = AFG.select(groupCols) \
           .groupBy(groupCols) \
           .count() \
           .orderBy('EventRootCodeString') \
           .toPandas()
display(AFG_E)

# COMMAND ----------

groupCols = ['EventRootCodeString',
             'GRV_1d_outlier','GRV_60d_outlier']
AFG_G = AFG.select(groupCols) \
           .groupBy(groupCols) \
           .count() \
           .orderBy('EventRootCodeString') \
           .toPandas()
display(AFG_G)

# COMMAND ----------

groupCols = ['EventRootCodeString',
             'TRV_1d_outlier','TRV_60d_outlier']
AFG_T = AFG.select(groupCols) \
           .groupBy(groupCols) \
           .count() \
           .orderBy('EventRootCodeString') \
           .toPandas()
display(AFG_T)

# COMMAND ----------

AFG.na.fill("").write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/AFG_MAD_alertsystem_14april2021.csv')