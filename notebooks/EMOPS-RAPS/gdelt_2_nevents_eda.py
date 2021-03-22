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
# MAGIC When the *Event Report Value* for a given PA1 (*3 DAYS*) is one standard deviation above ERA1* 
# MAGIC -	Event trend alert: 
# MAGIC when the *Event Report Value* for a given PA2 (*60 DAYS*) is one standard deviation above ERA2*
# MAGIC 
# MAGIC 
# MAGIC Sources:
# MAGIC - (1) [Moving Averaging with Apache Spark](https://www.linkedin.com/pulse/time-series-moving-average-apache-pyspark-laurent-weichberger/)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the sake of data storage, the *Weighted Average* of the Target Variables will be assessed, since the average numerical value per global event id per country per date was created in the previous preprocessing process.

# COMMAND ----------

# DBTITLE 1,Import Modules
from functools import reduce
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from pyspark.mllib.stat import Statistics
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

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
  .load("/Filestore/tables/tmp/gdelt/targetvalues.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Events Data
eventsData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence',
                                          'EventReportValue','weightedERA_3d','weightedERA_60d')

print((eventsData.count(), len(eventsData.columns)))
eventsData.limit(2).toPandas()

# COMMAND ----------

display(eventsData)

# COMMAND ----------

display(eventsData)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Assess the Difference Between 3d and 30d Rolling Averages of the Event Report Value (ERV - % Articles)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (16,8)
import pylab as pl
import seaborn as sns
import scipy.stats as stats
# Seed the random number generator
np.random.seed(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ERV

# COMMAND ----------

print('Number of Nulls: ', eventsData.filter(F.col('EventReportValue').isNull()).count())
eventsData.select('EventReportValue').describe().show()

# COMMAND ----------

quantile = eventsData.approxQuantile(['EventReportValue'], [0.25, 0.5, 0.75], 0)
quantile_25 = quantile[0][0]
quantile_50 = quantile[0][1]
quantile_75 = quantile[0][2]
print('quantile_25: '+str(quantile_25))
print('quantile_50: '+str(quantile_50))
print('quantile_75: '+str(quantile_75))

# COMMAND ----------

dailyERV = eventsData.select('EventReportValue').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.2, .9)})

# add boxplot
sns.boxplot(dailyERV, ax=boxplot)
boxplot.set(xlabel='') # Remove x-axis label from boxplot

# add histogram and normal curve
fit = stats.norm.pdf(dailyERV, np.mean(dailyERV), np.std(dailyERV))
pl.plot(dailyERV, fit, '-o')
pl.hist(dailyERV, density=True, alpha=0.5, bins=20)

# label axis 
pl.xlabel('EventReportValue')
pl.ylabel('Probability Density Function')
pl.title('EventReportValue Distribution')

# how plot and print mean and std sample information
plt.show()
'The sample(n=' + str(len(dailyERV)) + ') population mean difference of averages is ' + str(round(np.mean(dailyERV), 2)) + ' with a standard deviation of ' + str(round(np.std(dailyERV), 2)) + '.'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 3D

# COMMAND ----------

print('Number of Nulls: ', eventsData.filter(F.col('weightedERA_3d').isNull()).count())
eventsData.select('weightedERA_3d').describe().show()

# COMMAND ----------

quantile = eventsData.approxQuantile(['weightedERA_3d'], [0.25, 0.5, 0.75], 0)
quantile_25 = quantile[0][0]
quantile_50 = quantile[0][1]
quantile_75 = quantile[0][2]
print('quantile_25: '+str(quantile_25))
print('quantile_50: '+str(quantile_50))
print('quantile_75: '+str(quantile_75))

# COMMAND ----------

weightedERA_3 = eventsData.select('weightedERA_3d').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.2, .9)})

# add boxplot
sns.boxplot(weightedERA_3, ax=boxplot)
boxplot.set(xlabel='') # Remove x-axis label from boxplot

# add histogram and normal curve
fit = stats.norm.pdf(weightedERA_3, np.mean(weightedERA_3), np.std(weightedERA_3))
pl.plot(weightedERA_3, fit, '-o')
pl.hist(weightedERA_3, density=True, alpha=0.5, bins=20)

# label axis 
pl.xlabel('ERV 3d Rolling Averages')
pl.ylabel('Probability Density Function')
pl.title('ERV 3d Rolling Weighted Averages Distribution')

# how plot and print mean and std sample information
plt.show()
'The sample(n=' + str(len(weightedERA_3)) + ') population mean difference of averages is ' + str(round(np.mean(weightedERA_3), 2)) + ' with a standard deviation of ' + str(round(np.std(weightedERA_3), 2)) + '.'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 60D

# COMMAND ----------

print('Number of Nulls: ', eventsData.filter(F.col('weightedERA_60d').isNull()).count())
eventsData.select('weightedERA_60d').describe().show()

# COMMAND ----------

quantile = eventsData.approxQuantile(['weightedERA_60d'], [0.25, 0.5, 0.75], 0)
quantile_25 = quantile[0][0]
quantile_50 = quantile[0][1]
quantile_75 = quantile[0][2]
print('quantile_25: '+str(quantile_25))
print('quantile_50: '+str(quantile_50))
print('quantile_75: '+str(quantile_75))

# COMMAND ----------

weightedERA_60 = eventsData.select('weightedERA_60d').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.2, .9)})

# add boxplot
sns.boxplot(weightedERA_60, ax=boxplot)
boxplot.set(xlabel='') # Remove x-axis label from boxplot

# add histogram and normal curve
fit = stats.norm.pdf(weightedERA_60, np.mean(weightedERA_60), np.std(weightedERA_60))
pl.plot(weightedERA_60, fit, '-o')
pl.hist(weightedERA_60, density=True, alpha=0.5, bins=20)

# label axis 
pl.xlabel('ERV 60d Rolling Averages')
pl.ylabel('Probability Density Function')
pl.title('ERV 60d Rolling Weighted Averages Distribution')

# how plot and print mean and std sample information
plt.show()
'The sample(n=' + str(len(weightedERA_60d)) + ') population mean difference of averages is ' + str(round(np.mean(weightedERA_60d), 2)) + ' with a standard deviation of ' + str(round(np.std(weightedERA_60d), 2)) + '.'

# COMMAND ----------

# DBTITLE 1,Assess Normalcy
# create theoretical dataset with Normal Distribution 
cdf_mean = np.mean(avgDiffs)
cdf_std = np.std(avgDiffs)

# Simulate a random sample with the same distribution and size of 1,000,000
cdf_samples = np.random.normal(cdf_mean, cdf_std, size=1000000)

# COMMAND ----------

# DBTITLE 1,Calculate Empirical Cumulative Distribution Function (ECDF)
# create edcf function
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

# Compute the CDFs
x_sample, y_sample = ecdf(avgDiffs)
x_norm, y_norm = ecdf(cdf_samples)

# COMMAND ----------

# DBTITLE 1,Plot ECDF
# Plot both ECDFs on same the same figure
fig = plt.plot(x_sample, y_sample, marker='.', linestyle='none', alpha=0.5)
fig = plt.plot(x_norm, y_norm, marker='.', linestyle='none', alpha=0.5)

# Label figure
fig = plt.xlabel('Difference Between ERV 3d and 30d Rolling Averages')
fig = plt.ylabel('CDF')
fig = plt.legend(('Sample Population', 'Theoretical Norm'))
fig = plt.title('Distribution of Difference Between ERV 3d and 30d Rolling Averages')

# Save plots
plt.show()

# COMMAND ----------

# create the split list ranging from 0 to  21, interval of 0.5
split_list = [float(i) for i in np.arange(0,21,0.5)]

# initialize buketizer
bucketizer = Bucketizer(splits=split_list,inputCol='difference',outputCol='buckets')

# transform
df_buck = bucketizer.setHandleInvalid('keep').transform(rollingERAs.select('difference').dropna())

# the "buckets" column gives the bucket rank, not the acctual bucket value(range), 
# use dictionary to match bucket rank and bucket value
bucket_names = dict(zip([float(i) for i in range(len(split_list[1:]))],split_list[1:]))

# user defined function to update the data frame with the bucket value
udf_foo = F.udf(lambda x: bucket_names[x], DoubleType())
#bins = df_buck.withColumn('bins', udf_foo('buckets')).groupBy('bins').count().sort('bins').toPandas()

# COMMAND ----------

