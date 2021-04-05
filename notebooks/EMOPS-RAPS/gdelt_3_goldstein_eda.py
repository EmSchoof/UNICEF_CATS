# Databricks notebook source
# MAGIC %md
# MAGIC ### Calculations – Percentages of Goldstein Scores
# MAGIC 
# MAGIC - 	Goldstein point value (GPV): 
# MAGIC Calculated as the average Goldstein score for all articles tagged as associated to the country
# MAGIC -	Goldstein Running Average (GRA1):
# MAGIC Calculated as the rolling average of the GPV for PA1 over the previous 12 months
# MAGIC -	Goldstein Running Average (GRA2):
# MAGIC Calculated as the rolling average of the GPV for PA2 over the previous 24 months
# MAGIC -	Goldstein spike alert: 
# MAGIC When the *Goldstein Point Value* for a given PA1 (*1 DAYS*) is one standard deviation above GRA1* 
# MAGIC -	Goldstein trend alert: 
# MAGIC when the *Goldstein Point Value* for a given PA2 (*60 DAYS*) is one standard deviation above GRA2*

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
  .load("/Filestore/tables/tmp/gdelt/gold_tone_targetvalues.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Goldstein Data
goldsteinData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','nArticles','avgConfidence',
                                          'GoldsteinReportValue','wGRA_1d','wGRA_60d', 'if_conflict')

print((goldsteinData.count(), len(goldsteinData.columns)))
goldsteinData.limit(2).toPandas()

# COMMAND ----------

# It's a best practice to sample data from your Spark df into pandas
sub_goldsteinData = goldsteinData#.sample(withReplacement=False, fraction=0.5, seed=42)

# separate into conflict vs not 
goldsteinDataConflict = sub_goldsteinData.filter(F.col('if_conflict') == True)
print((goldsteinDataConflict.count(), len(goldsteinDataConflict.columns)))
goldsteinDataNonConflict = sub_goldsteinData.filter(F.col('if_conflict') != True)
print((goldsteinDataNonConflict.count(), len(goldsteinDataNonConflict.columns)))

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
# MAGIC ### GRV Weighted Average, 1D

# COMMAND ----------

# MAGIC %md
# MAGIC Afghanistan

# COMMAND ----------

afg_grv_conflict = eda_funcs(df=goldsteinData, country='Afghanistan', col='wGRA_1d', conflict=True)

# COMMAND ----------

afg_grv_nonconflict = eda_funcs(df=goldsteinData, country='Afghanistan', col='wGRA_1d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Somalia

# COMMAND ----------

som_grv_conflict = eda_funcs(df=goldsteinData, country='Somalia', col='wGRA_1d', conflict=True)

# COMMAND ----------

som_grv_nonconflict = eda_funcs(df=goldsteinData, country='Somalia', col='wGRA_1d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted GRA 60D

# COMMAND ----------

# MAGIC %md
# MAGIC Afghanistan

# COMMAND ----------

afg_grv60d_conflict = eda_funcs(df=goldsteinData, country='Afghanistan', col='wGRA_60d', conflict=True)

# COMMAND ----------

afg_grv60d_nonconflict = eda_funcs(df=goldsteinData, country='Afghanistan', col='wGRA_60d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Somalia

# COMMAND ----------

som_grv60d_nonconflict = eda_funcs(df=goldsteinData, country='Somalia', col='wGRA_60d', conflict=False)

# COMMAND ----------

som_grv60d_conflict = eda_funcs(df=goldsteinData, country='Somalia', col='wGRA_60d', conflict=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Initial Conclusion:
# MAGIC All versions of the GoldsteinReportValue (Daily and Rolling Averages) in both Conflict and Non-Conflict events seems to be *generally* normally distribution. As mentioned earlier, the Central Limit Theorem (CLT) states that, in many situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed. More exploration needs to be performed in order to determine if this is sufficient for the purpose of the alert system, or if, like the EventReportValue, a non-parametirc statistics approach would be more accurate.

# COMMAND ----------

