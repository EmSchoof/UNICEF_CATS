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

# DBTITLE 1,Select Events Data
eventsData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence',
                                          'EventReportValue','weightedERA_3d','weightedERA_60d')

print((eventsData.count(), len(eventsData.columns)))
eventsData.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Explore Values in Conflict vs Not Situations
# create conflict column
conflict_events = ['DEMAND','DISAPPROVE','PROTEST','REJECT','THREATEN','ASSAULT','COERCE','ENGAGE IN UNCONVENTIONAL MASS VIOLENCE','EXHIBIT MILITARY POSTURE','FIGHT','REDUCE RELATIONS']
eventsData = eventsData.withColumn('if_conflict', F.when(F.col('EventRootCodeString').isin(conflict_events), True).otherwise(False))

# COMMAND ----------

# It's a best practice to sample data from your Spark df into pandas
sub_eventsData = eventsData.sample(withReplacement=False, fraction=0.5, seed=42)

# separate into conflict vs not 
eventsDataConflict = sub_eventsData.filter(F.col('if_conflict') == True)
print((eventsDataConflict.count(), len(eventsDataConflict.columns)))
eventsDataNonConflict = sub_eventsData.filter(F.col('if_conflict') != True)
print((eventsDataNonConflict.count(), len(eventsDataNonConflict.columns)))

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
  list_conflict = eventsData.select(col).rdd.flatMap(lambda x: x).collect()
  
  # get quantiles
  print('Get Conflict Quantiles for ' + col)
  get_quantiles(eventsDataConflict, col)
  plot_boxplot(list_conflict, col)
  
  # plot dist
  plot_dist(eventsDataConflict, col)
  
  # plot ecdf
  plot_ecdf(list_conflict, 'CONFLICT ' + col)

# COMMAND ----------

def nonconflict_eda_funcs(col): 
  
  # get lists
  list_not = eventsDataNonConflict.select(col).rdd.flatMap(lambda x: x).collect()
  
  # get quantiles
  print('Get Non-Conflict Quantiles for ' + col)
  get_quantiles(eventsDataNonConflict, col)
  plot_boxplot(list_not, col)
  
  # plot dist
  plot_dist(eventsDataNonConflict, col)
 
  # plot ecdf
  plot_ecdf(list_not, 'NON-CONFLICT ' + col)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ERV

# COMMAND ----------

 conflict_eda_funcs('EventReportValue')

# COMMAND ----------

nonconflict_eda_funcs('EventReportValue')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 3D

# COMMAND ----------

 conflict_eda_funcs('weightedERA_3d')

# COMMAND ----------

 nonconflict_eda_funcs('weightedERA_3d')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 60D

# COMMAND ----------

 conflict_eda_funcs('weightedERA_60d')

# COMMAND ----------

nonconflict_eda_funcs('weightedERA_60d')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Initial Conclusion:
# MAGIC All versions of the EventReportValue (Daily and Rolling Averages) in both Conflict and Non-Conflict events have a skewed-right distribution. This generally means that the mean and median are both greater than the mode, and the mean is greater than median. In order to account for these inconsistencies, the following steps with log-transform these variables
# MAGIC [source](https://towardsdatascience.com/skewed-data-a-problem-to-your-statistical-model-9a6b5bb74e37)