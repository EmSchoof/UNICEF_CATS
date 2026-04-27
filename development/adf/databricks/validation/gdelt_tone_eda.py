# Databricks notebook source
# MAGIC %md
# MAGIC ### Calculations – Percentages of Tone Scores
# MAGIC 
# MAGIC - 	Tone point value (TPV): 
# MAGIC Calculated as the average Mention Tone for all articles tagged as associated to the country
# MAGIC -	Tone Running Average (TRA1):
# MAGIC Calculated as the rolling average of the TPV for PA1 over the previous 12 months
# MAGIC -	Tone Running Average (TRA2):
# MAGIC Calculated as the rolling average of the TPV for PA2 over the previous 24 months
# MAGIC -	Tone spike alert: 
# MAGIC When the *Tone Point Value* for a given PA1 (*1 DAYS*) is one standard deviation above TRA1* 
# MAGIC -	Tone trend alert: 
# MAGIC when the *Tone Point Value* for a given PA2 (*60 DAYS*) is one standard deviation above TRA2*

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
  .load("/Filestore/tables/tmp/gdelt/gold_tone_targetvalues_confidence40plus.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Tone Data
toneData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','nArticles','avgConfidence',
                                          'ToneReportValue','wTRA_1d','wTRA_60d','if_conflict')

print((toneData.count(), len(toneData.columns)))
toneData.limit(2).toPandas()

# COMMAND ----------

datesDF = toneData.select('EventTimeDate')
min_date, max_date = datesDF.select(F.min('EventTimeDate'),F.max('EventTimeDate')).first()
min_date, max_date

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Plot

# COMMAND ----------

def get_quantiles(df, col):
    quantile = df.approxQuantile([col], [0.25, 0.5, 0.75], 0)
    quantile_25 = quantile[0][0]
    quantile_50 = quantile[0][1]
    quantile_75 = quantile[0][2]
    print('quantile_25: '+str(quantile_25))
    print('quantile_50: '+str(quantile_50))
    print('quantile_75: '+str(quantile_75))

def plot_boxplot(var_list, title):
  sns.boxplot(var_list)
  plt.xlabel(title)
  plt.ylabel('Probability Density Function')
  plt.title('Distribution of ' + title)
  plt.show()
  'The sample(n=' + str(len(var_list)) + ') population mean is ' + str(round(np.mean(var_list), 2)) + ' with a standard deviation of ' + str(round(np.std(var_list), 2)) + '.'
  
def plot_dist(df, col):
    sample_df = df.select(col).sample(False, 0.5, 42)
    pandas_df = sample_df.toPandas()
    sns.distplot(pandas_df)
    plt.xlabel(col) 
    plt.show()
    
def normal_dist_proof(vars_list, title):
    k2, p = stats.normaltest(vars_list)
    alpha = 0.05 # 95% confidence
    print("p = {}".format(p))
    print("n = " + str(len(vars_list)))
    if p < alpha: # if norm
      print(title + " IS normally distributed.") 
    else:
      print(title + " *IS NOT* normally distributed.")

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

def plot_ecdf(vals_list, title): 
      cdf_mean = np.mean(vals_list)
      cdf_std = np.std(vals_list)
      cdf_samples = np.random.normal(cdf_mean, cdf_std, size=1000000)
      x_sample, y_sample = ecdf(vals_list)
      x_norm, y_norm = ecdf(cdf_samples)
      fig = plt.plot(x_sample, y_sample, marker='.', linestyle='none', alpha=0.5)
      fig = plt.plot(x_norm, y_norm, marker='.', linestyle='none', alpha=0.5)
      plt.xlabel(title)
      fig = plt.ylabel('CDF')
      fig = plt.legend(('Sample Population', 'Theoretical Norm'), loc='lower right')
      fig = plt.title('Variable ECDF Distribution Compared to Statistical Norm')
      plt.show()

# COMMAND ----------

def eda_funcs(df, country, col, conflict=True): 
  
  if conflict == True:
      name = 'Conflict'
      df1 = df.filter((F.col('ActionGeo_FullName') == country) & (F.col('if_conflict') == True))
      list_vals = df.select(col).rdd.flatMap(lambda x: x).collect()
  else:
      name = 'NonConflict'  
      df1 = df.filter((F.col('ActionGeo_FullName') == country) & (F.col('if_conflict') != True))
      list_vals = df.select(col).rdd.flatMap(lambda x: x).collect()
  
  print('Get ' + name + ' Quantiles for ' + col)
  get_quantiles(df1, col)
  plot_boxplot(list_vals, col)
  plot_dist(df1, col)
  plot_ecdf(list_vals,  name + ' ' +  col)
  normal_dist_proof(list_vals,  name + ' ' +  col)
  return list_vals

# COMMAND ----------

# MAGIC %md
# MAGIC ### TRV Weighted Average, 1D

# COMMAND ----------

# MAGIC %md
# MAGIC Afghanistan

# COMMAND ----------

afg_trv_conflict = eda_funcs(df=toneData, country='Afghanistan', col='wTRA_1d', conflict=True)

# COMMAND ----------

afg_trv_nonconflict = eda_funcs(df=toneData, country='Afghanistan', col='wTRA_1d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Somalia

# COMMAND ----------

som_trv_conflict = eda_funcs(df=toneData, country='Somalia', col='wTRA_1d', conflict=True)

# COMMAND ----------

som_trv_nonconflict = eda_funcs(df=toneData, country='Somalia', col='wTRA_1d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TRV Weighted Average, 60D

# COMMAND ----------

# MAGIC %md
# MAGIC Afghanistan

# COMMAND ----------

afg_trv60d_conflict = eda_funcs(df=toneData, country='Afghanistan', col='wTRA_60d', conflict=True)

# COMMAND ----------

afg_trv60d_nonconflict = eda_funcs(df=toneData, country='Afghanistan', col='wTRA_60d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Somalia

# COMMAND ----------

som_trv60d_conflict = eda_funcs(df=toneData, country='Somalia', col='wTRA_60d', conflict=True)

# COMMAND ----------

som_trv60d_nonconflict = eda_funcs(df=toneData, country='Somalia', col='wTRA_60d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore Tone distribution per country 
# MAGIC [source](https://syamkakarla.medium.com/intermediate-guide-to-pyspark-pyspark-sql-functions-with-examples-7eec883b5eaa)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Skewness**: a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.
# MAGIC 
# MAGIC **Kurtosis**: a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails or outliers. Data sets with low kurtosis tend to have light tails or a lack of outliers.
# MAGIC 
# MAGIC **Standard Deviation**: a statistical measure of the dispersion of the data relative to its mean. It is calculated with the square root of the variance. A low standard deviation indicates that the values tend to be close to the mean of the dataset, while a high standard deviation indicates that the values are spread out over a wider range.
# MAGIC 
# MAGIC **Variance**: a measure of variability. It is calculated by taking the average of squared deviations from the mean. Variance tells you the degree of spread in your data set. The more spread the data, the larger the variance is in relation to the mean.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test for Normal Distribution of Tone by Country without distinguishing between if_conflict

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

toneDataAll = toneData.select('ActionGeo_FullName', 'wTRA_1d', 'wTRA_60d', 'nArticles') \
                                        .groupBy('ActionGeo_FullName') \
                                        .agg( F.skewness('wTRA_1d'),
                                              F.kurtosis('wTRA_1d'),
                                              F.stddev('wTRA_1d'),
                                              F.variance('wTRA_1d'),
                                              F.collect_list('wTRA_1d').alias('list_wTRA_1d'),
                                              F.skewness('wTRA_60d'),
                                              F.kurtosis('wTRA_60d'),
                                              F.stddev('wTRA_60d'),
                                              F.variance('wTRA_60d'),
                                              F.collect_list('wTRA_60d').alias('list_wTRA_60d'),
                                              F.sum('nArticles').alias('nArticles'),
                                              F.count(F.lit(1)).alias('n_observations')
                                        )

# get p-value and define normalcy
toneDataAll = toneDataAll.withColumn('p_value_1d', get_pval_udf(toneDataAll.list_wTRA_1d))
toneDataAll = toneDataAll.withColumn('if_normal_1d', if_norm_udf(toneDataAll.p_value_1d))
toneDataAll = toneDataAll.withColumn('p_value_60d', get_pval_udf(toneDataAll.list_wTRA_60d))
toneDataAll = toneDataAll.withColumn('if_normal_60d', if_norm_udf(toneDataAll.p_value_60d))
toneDataAll.limit(5).toPandas()

# COMMAND ----------

toneDataAll.select('ActionGeo_FullName', 'if_normal_1d').filter(F.col('if_normal_1d') == False).count()

# COMMAND ----------

toneDataAll.select('ActionGeo_FullName', 'if_normal_60d').filter(F.col('if_normal_60d') == False).count()