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
from pyspark.ml.feature import Bucketizer
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
  .load("/Filestore/tables/tmp/gdelt/preprocessed.csv")

# COMMAND ----------

# DBTITLE 1,Verify Unique on Global Event IDs
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.agg(F.countDistinct(F.col("GLOBALEVENTID")).alias("nEvents")).show()
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Only Conflict Events
conflictEvents = preprocessedGDELT.filter(F.col('QuadClassString').isin('Verbal Conflict', 'Material Conflict'))
print((conflictEvents.count(), len(conflictEvents.columns)))
conflictEvents.limit(10).toPandas()

# COMMAND ----------

display(rollingERAs)

# COMMAND ----------

display(rollingERAs)

# COMMAND ----------

# DBTITLE 1,Assess Value Correlations In Dataset
def plot_corr_matrix(correlations,attr,fig_no):
    fig=plt.figure(fig_no, figsize=(16,10))
    ax=fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticklabels(['']+attr)
    ax.set_yticklabels(['']+attr)
    cax=ax.matshow(correlations,vmax=1,vmin=-1)
    fig.colorbar(cax)
    plt.show()
    
# select variables to check correlation
df_features = rollingERAs.select('avgConfidence','avgTone','avgGoldstein','nArticles','EventReportValue','ERA_3d','ERA_30d','difference') 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
plot_corr_matrix(corr_mat, df_features.columns, 234)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Assess the Difference Between 3d and 30d Rolling Averages of the Event Report Value (ERV - % Articles)

# COMMAND ----------

plt.rcParams["figure.figsize"] = (16,8)
import pylab as pl
import seaborn as sns
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF

# COMMAND ----------

avgDiffs = rollingERAs.select('difference').rdd.flatMap(lambda x: x).collect()
#avgDiffs

# COMMAND ----------

print('Number of Nulls: ', rollingERAs.filter(F.col('difference').isNull()).count())
rollingERAs.select('difference').describe().show()

# COMMAND ----------

quantile = rollingERAs.approxQuantile(['difference'], [0.25, 0.5, 0.75], 0)
quantile_25 = quantile[0][0]
quantile_50 = quantile[0][1]
quantile_75 = quantile[0][2]
print('quantile_25: '+str(quantile_25))
print('quantile_50: '+str(quantile_50))
print('quantile_75: '+str(quantile_75))

# COMMAND ----------

# DBTITLE 1,Plot the Distribution of the Difference Between Two Averages
# create a figure with two plots
fig, (boxplot, histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.2, .9)})

# add boxplot
sns.boxplot(avgDiffs, ax=boxplot)
boxplot.set(xlabel='') # Remove x-axis label from boxplot

# add histogram and normal curve
fit = stats.norm.pdf(avgDiffs, np.mean(avgDiffs), np.std(avgDiffs))
pl.plot(avgDiffs, fit, '-o')
#pl.hist(avgDiffs, density=True, alpha=0.5, bins=20)

# label axis 
pl.xlabel('Difference Between ERV 3d and 30d Rolling Averages')
pl.ylabel('Probability Density Function')
pl.title('Difference Distribution Associated with ERV 3d and 30d Rolling Averages')

# how plot and print mean and std sample information
plt.show()
'The sample(n=' + str(len(avgDiffs)) + ') population mean difference of averages is ' + str(round(np.mean(avgDiffs), 2)) + ' with a standard deviation of ' + str(round(np.std(avgDiffs), 2)) + '.'

# COMMAND ----------

# DBTITLE 1,Assess Normalcy
# Seed the random number generator
np.random.seed(15)

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

