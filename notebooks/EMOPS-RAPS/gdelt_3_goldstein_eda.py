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
# MAGIC When the *Goldstein Point Value* for a given PA1 (*3 DAYS*) is one standard deviation above GRA1* 
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
  .load("/Filestore/tables/tmp/gdelt/targetvalues.csv")
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Goldstein Data
goldsteinData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence',
                                          'GoldsteinReportValue','weightedGRA_3d','weightedGRA_60d')

print((goldsteinData.count(), len(goldsteinData.columns)))
goldsteinData.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Explore Values in Conflict vs Not Situations
# create conflict column
conflict_events = ['DEMAND','DISAPPROVE','PROTEST','REJECT','THREATEN','ASSAULT','COERCE','ENGAGE IN UNCONVENTIONAL MASS VIOLENCE','EXHIBIT MILITARY POSTURE','FIGHT','REDUCE RELATIONS']
goldsteinData = goldsteinData.withColumn('if_conflict', F.when(F.col('EventRootCodeString').isin(conflict_events), True).otherwise(False))

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

def plot_boxplot(var_list, title):
  
  # Add boxplot for var
  sns.boxplot(var_list)

  # Label axis 
  plt.xlabel(title)
  plt.ylabel('Probability Density Function')
  plt.title('Distribution of ' + title)

  # Show plot and add print mean and std sample information
  plt.show()
  'The sample(n=' + str(len(var_list)) + ') population mean is ' + str(round(np.mean(var_list), 2)) + ' with a standard deviation of ' + str(round(np.std(var_list), 2)) + '.'

# COMMAND ----------

def get_quantiles(df, col):
    quantile = df.approxQuantile([col], [0.25, 0.5, 0.75], 0)
    quantile_25 = quantile[0][0]
    quantile_50 = quantile[0][1]
    quantile_75 = quantile[0][2]
    print('quantile_25: '+str(quantile_25))
    print('quantile_50: '+str(quantile_50))
    print('quantile_75: '+str(quantile_75))

# COMMAND ----------

# DBTITLE 1,Calculate Empirical Cumulative Distribution Function (ECDF)
# create edcf function
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

# COMMAND ----------

# DBTITLE 1,Plot ECDF
def plot_ecdf(vals_list, title):
  
      # create theoretical dataset with Normal Distribution 
      cdf_mean = np.mean(vals_list)
      cdf_std = np.std(vals_list)

      # Simulate a random sample with the same distribution and size of 1,000,000
      cdf_samples = np.random.normal(cdf_mean, cdf_std, size=1000000)

      # Compute the CDFs
      x_sample, y_sample = ecdf(vals_list)
      x_norm, y_norm = ecdf(cdf_samples)

      # Plot both ECDFs on same the same figure
      fig = plt.plot(x_sample, y_sample, marker='.', linestyle='none', alpha=0.5)
      fig = plt.plot(x_norm, y_norm, marker='.', linestyle='none', alpha=0.5)

      # Label figure
      plt.xlabel(title)
      fig = plt.ylabel('CDF')
      fig = plt.legend(('Sample Population', 'Theoretical Norm'), loc='lower right')
      fig = plt.title('Variable ECDF Distribution Compared to Statistical Norm')

      # Save plots
      plt.show()

# COMMAND ----------

def plot_dist(df, col):
  
    # Plot distribution of a features
    # Select a single column and sample and convert to pandas
    sample_df = df.select(col).sample(False, 0.5, 42)
    pandas_df = sample_df.toPandas()

    # Plot distribution of pandas_df and display plot
    sns.distplot(pandas_df)
    plt.xlabel(col) 
    plt.show()

# COMMAND ----------

def conflict_eda_funcs(col): 
  
  # get lists
  list_conflict = goldsteinDataConflict.select(col).rdd.flatMap(lambda x: x).collect()
  
  # get quantiles
  print('Get Conflict Quantiles for ' + col)
  get_quantiles(goldsteinDataConflict, col)
  plot_boxplot(list_conflict, col)
  
  # plot dist
  plot_dist(goldsteinDataConflict, col)
  
  # plot ecdf
  plot_ecdf(list_conflict, 'CONFLICT ' + col)
  
  return list_conflict

# COMMAND ----------

def nonconflict_eda_funcs(col): 
  
  # get lists
  list_not = goldsteinDataNonConflict.select(col).rdd.flatMap(lambda x: x).collect()
  
  # get quantiles
  print('Get Non-Conflict Quantiles for ' + col)
  get_quantiles(goldsteinDataNonConflict, col)
  plot_boxplot(list_not, col)
  
  # plot dist
  plot_dist(goldsteinDataNonConflict, col)
 
  # plot ecdf
  plot_ecdf(list_not, 'NON-CONFLICT ' + col)
  
  return list_not

# COMMAND ----------

# MAGIC %md
# MAGIC ### GRV

# COMMAND ----------

grv_conflict = conflict_eda_funcs('GoldsteinReportValue')

# COMMAND ----------

grv_nonconflict = nonconflict_eda_funcs('GoldsteinReportValue')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted GRA 3D

# COMMAND ----------

weightedGRA_3d_conflict = conflict_eda_funcs('weightedGRA_3d')

# COMMAND ----------

 weightedGRA_3d_nonconflict = nonconflict_eda_funcs('weightedGRA_3d')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted GRA 60D

# COMMAND ----------

weightedGRA_60d_conflict = conflict_eda_funcs('weightedGRA_60d')

# COMMAND ----------

weightedGRA_60d_nonconflict = nonconflict_eda_funcs('weightedGRA_60d')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Initial Conclusion:
# MAGIC All versions of the GoldsteinReportValue (Daily and Rolling Averages) in both Conflict and Non-Conflict events seems to be *loosely* normally distribution. As mentioned earlier, the Central Limit Theorem (CLT) states that, in many situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed. More exploration needs to be performed in order to determine if this is sufficient for the purpose of the alert system, or if, like the EventReportValue, a non-parametirc statistics approach would be more accurate.
# MAGIC - [Non-Parametric Statistics](http://erecursos.uacj.mx/bitstream/handle/20.500.11961/2064/Gibbons%2C%202003.pdf?sequence=14&isAllowed=y)

# COMMAND ----------

