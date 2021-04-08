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
# MAGIC When the *Event Report Value* for a given PA1 (*3 DAYS*) is <strike>one standard deviation</strike>  above ERA1* 
# MAGIC -	Event trend alert: 
# MAGIC when the *Event Report Value* for a given PA2 (*60 DAYS*) is <strike>one standard deviation</strike>  above ERA2*
# MAGIC 
# MAGIC 
# MAGIC *New Methodology*
# MAGIC - (1.0) Compute three day average for all the data values you have across all years. 
# MAGIC - (1.a) Your result will be X = [x1, x2, x3, ... xn]
# MAGIC - (2.0) Set a threshhold parameter z. // benchmark special parameter (needs to be established per country?) come up with a "Ground Truth" value
# MAGIC - (3.0) For each value in X, compare with z 
# MAGIC 
# MAGIC *Execution of Methodology*
# MAGIC - 1 - [create a list of 3 day moving averages from today - 365 days] // compare this list with defined 'z' anomalist behavior to the current 3 day average per EventRootCode
# MAGIC - 2 - [create a list of 60 day moving averages from today - 730 days] // compare this list with defined 'z' anomalist behavior to the current 60 day average per EventRootCode
# MAGIC 
# MAGIC 
# MAGIC Sources:
# MAGIC - (1) [Moving Averaging with Apache Spark](https://www.linkedin.com/pulse/time-series-moving-average-apache-pyspark-laurent-weichberger/)
# MAGIC 
# MAGIC Statiticians and Machine Learning Engineers
# MAGIC - Objective
# MAGIC - Methodology of Outcome
# MAGIC - Organizing Data to Deliver on that Outcome
# MAGIC - Current Challenges that I am encountering to delivering this outcome

# COMMAND ----------

# DBTITLE 1,Import Modules
from functools import reduce
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
eventsData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence','EventReportValue','wERA_3d','wERA_60d')
print((eventsData.count(), len(eventsData.columns)))
eventsData.limit(2).toPandas()

# COMMAND ----------

datesDF = eventsData.select('EventTimeDate')
min_date, max_date = datesDF.select(F.min('EventTimeDate'),F.max('EventTimeDate')).first()
min_date, max_date

# COMMAND ----------

# DBTITLE 1,Explore Values in Conflict vs Not Situations
# create conflict column
conflict_events = ['DEMAND','DISAPPROVE','PROTEST','REJECT','THREATEN','ASSAULT','COERCE','ENGAGE IN UNCONVENTIONAL MASS VIOLENCE','EXHIBIT MILITARY POSTURE','FIGHT','REDUCE RELATIONS']
eventsData = eventsData.withColumn('if_conflict', F.when(F.col('EventRootCodeString').isin(conflict_events), True).otherwise(False))

# COMMAND ----------

eventsData.limit(10).toPandas()

# COMMAND ----------

# It's a best practice to sample data from your Spark df into pandas #.sample(withReplacement=False, fraction=0.5, seed=42)
explore_countries = ['Afghanistan','Myanmar','Somalia','Guinea']
sub_eventsData = eventsData.filter(F.col('ActionGeo_FullName').isin(explore_countries))

# separate into conflict vs not 
eventsDataConflict = sub_eventsData.filter(F.col('if_conflict') == True)
print((eventsDataConflict.count(), len(eventsDataConflict.columns)))
eventsDataNonConflict = sub_eventsData.filter(F.col('if_conflict') != True)
print((eventsDataNonConflict.count(), len(eventsDataNonConflict.columns)))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Plot

# COMMAND ----------

# DBTITLE 0,Calculate Empirical Cumulative Distribution Function (ECDF)
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
      list_vals = df1.select(col).rdd.flatMap(lambda x: x).collect()
  else:
      name = 'NonConflict'  
      df1 = df.filter((F.col('ActionGeo_FullName') == country) & (F.col('if_conflict') != True))
      list_vals = df1.select(col).rdd.flatMap(lambda x: x).collect()
  
  print('Get ' + name + ' Quantiles for ' + col)
  get_quantiles(df1, col)
  plot_boxplot(list_vals, col)
  plot_dist(df1, col)
  plot_ecdf(list_vals,  name + ' ' +  col)
  normal_dist_proof(list_vals,  name + ' ' +  col)
  return list_vals 

# COMMAND ----------

# MAGIC %md
# MAGIC ### ERV

# COMMAND ----------

afg_erv_conflict = eda_funcs(df=eventsData, country='Afghanistan', col='EventReportValue', conflict=True) #eventcode='',

# COMMAND ----------

afg_erv_nonconflict = eda_funcs(df=eventsData, country='Afghanistan', col='EventReportValue', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### wERA 3D

# COMMAND ----------

afg_erv3d_conflict = eda_funcs(df=eventsData, country='Afghanistan', col='wERA_3d', conflict=True)

# COMMAND ----------

afg_erv3d_nonconflict = eda_funcs(df=eventsData, country='Afghanistan', col='wERA_3d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### wERA 60D

# COMMAND ----------

afg_erv60d_conflict = eda_funcs(df=eventsData, country='Afghanistan', col='wERA_60d', conflict=True)

# COMMAND ----------

afg_erv60d_nonconflict = eda_funcs(df=eventsData, country='Afghanistan', col='wERA_60d', conflict=False)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Initial Conclusion:
# MAGIC All versions of the EventReportValue (Daily and Rolling Averages) in both Conflict and Non-Conflict events have a skewed-right distribution. This generally means that the mean and median are both greater than the mode, and the mean is greater than median. While the Central Limit Theorem (CLT) states that, in many situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed, non-parametirc statistics are more accurate to perform on data that does not meet the criteria of normal distribution. Since the purpose of this assessment is to be able to detect a statistical change in the percent Articles (EventReportValue) rolling averages that could indicate an increasing trend in conflict events within a given country, a non-parametric test, like Kruskai-Wallis, should replace the proposed parametric standard deviation, which is a form of descriptive statistics that is reliant on normal distribution.
# MAGIC - [Medium: Skewed Data](https://towardsdatascience.com/skewed-data-a-problem-to-your-statistical-model-9a6b5bb74e37)
# MAGIC - [Non-Parametric Statistics](http://erecursos.uacj.mx/bitstream/handle/20.500.11961/2064/Gibbons%2C%202003.pdf?sequence=14&isAllowed=y)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC (A) As mentioned above, the Kruskal-Wallis H-test is the non-parametric version of ANOVA tests, which the null hypothesis that the population median of all of the groups are equal. The test works on 2 or more independent samples, which may have different sizes, and makes the assumption that H has a chi square distribution. Since the rejection the null hypothesis does not indicate which of the groups differs, a post hoc comparisons between the groups is required to determine which groups are different. [source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
# MAGIC 
# MAGIC 
# MAGIC #### Assumptions for Kruskal-Wallis H-test:
# MAGIC - *Assumption #1*: The dependent variable is a continuous or ordinal variable
# MAGIC   - EventReportValue et.al is a percentage, which is a type of continuous variable measured between 0 and 1
# MAGIC - *Assumption #2*: The independent variable should consist of 2 or more categorical, independent groups.
# MAGIC   - When grouping at the country level, there are 4 categorical groups for QuadClass, which can be subdivided in to 20 independent categories of EventRootCodes. 
# MAGIC   - However, for the initial iteration of this analysis, all data will be grouped into 2 categorical, independent groups: Conflict vs Non-Conflict.
# MAGIC - *Assumption #3*: There is independence of observations, which means that there is no relationship between the observations in each group or between the groups themselves.
# MAGIC   - When grouping at the country level, each global event and respective calculated EventReportValue is entirely independents of other entries.
# MAGIC - *Assumption #4*: The distribution of each dependent variable categorical group has been explored; if the distributions are similar the median will be assessed, else the mean.
# MAGIC   - As discussed, above, all versions of the EventReportValue (Daily and Rolling Averages) in both Conflict and Non-Conflict events have a skewed-right distribution.
# MAGIC 
# MAGIC [source](https://statistics.laerd.com/spss-tutorials/kruskal-wallis-h-test-using-spss-statistics.php)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC (B) Another option is to transform the data in order create a normally-distributed variable. Since this variable is a percentage, one post suggested transforming the data via the arcsine method [source](https://www.researchgate.net/post/Should_I_do_log_transform_percentage_data_of_cell_subsets_measured_with_FACS/54b682a5d2fd645a788b4686/citation/download). Arcsine, or angular, transformation is the preferred variable transformation for multivariant problems where both 0% and 100% are viable options. [source](http://strata.uga.edu/8370/rtips/proportions.html#:~:text=The%20arcsine%20transformation%20(also%20called,square%20root%20of%20the%20proportion.&text=Multiplying%20by%20two%20makes%20the,scale%20stop%20at%20pi%2F2.)

# COMMAND ----------

# ERV_3d: Apply Kruskal-Wallis H-test
stats.kruskal(afg_erv_conflict, afg_erv_nonconflict)

# COMMAND ----------

# ERV_3d: Apply Kruskal-Wallis H-test
stats.kruskal(afg_erv3d_conflict, afg_erv3d_nonconflict)

# COMMAND ----------

# ERV_3d: Apply Kruskal-Wallis H-test
stats.kruskal(afg_erv60d_conflict, afg_erv60d_nonconflict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify for Non-Normal Distribution for multiple Countries

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

eventsDataAll = eventsData.select('ActionGeo_FullName', 'wERA_3d', 'wERA_60d', 'nArticles') \
                                        .groupBy('ActionGeo_FullName') \
                                        .agg( F.skewness('wERA_3d'),
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
                                        )

# get p-value and define normalcy
eventsDataAll = eventsDataAll.withColumn('p_value_3d', get_pval_udf(eventsDataAll.list_wERA_3d))
eventsDataAll = eventsDataAll.withColumn('if_normal_3d', if_norm_udf(eventsDataAll.p_value_3d))
eventsDataAll = eventsDataAll.withColumn('p_value_60d', get_pval_udf(eventsDataAll.list_wERA_60d))
eventsDataAll = eventsDataAll.withColumn('if_normal_60d', if_norm_udf(eventsDataAll.p_value_60d))
eventsDataAll.limit(5).toPandas()

# COMMAND ----------

eventsDataAll.select('ActionGeo_FullName','n_observations', 'if_normal_3d').filter(F.col('if_normal_3d') == False).count()

# COMMAND ----------

eventsDataAll.select('ActionGeo_FullName','n_observations','if_normal_60d').filter(F.col('if_normal_60d') == False).count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore ERV by Country by Event Code

# COMMAND ----------

eventsCodeData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence','EventReportValue','wERA_3d','wERA_60d')
print((eventsCodeData.count(), len(eventsCodeData.columns)))
eventsCodeData.limit(2).toPandas()

# COMMAND ----------

eventsCodeDataAll = eventsCodeData.select('ActionGeo_FullName', 'EventRootCodeString', 'wERA_3d', 'wERA_60d', 'nArticles') \
                                        .groupBy('ActionGeo_FullName', 'EventRootCodeString') \
                                        .agg( F.skewness('wERA_3d'),
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
                                        )

# get p-value and define normalcy
eventsCodeDataAll = eventsCodeDataAll.withColumn('p_value_3d', get_pval_udf(eventsCodeDataAll.list_wERA_3d))
eventsCodeDataAll = eventsCodeDataAll.withColumn('if_normal_3d', if_norm_udf(eventsCodeDataAll.p_value_3d))
eventsCodeDataAll = eventsCodeDataAll.withColumn('p_value_60d', get_pval_udf(eventsCodeDataAll.list_wERA_60d))
eventsCodeDataAll = eventsCodeDataAll.withColumn('if_normal_60d', if_norm_udf(eventsCodeDataAll.p_value_60d))
eventsCodeDataAll.limit(5).toPandas()

# COMMAND ----------

eventsCodeDataAll.select('ActionGeo_FullName','n_observations', 'if_normal_3d').filter(F.col('if_normal_3d') == False).count()

# COMMAND ----------

eventsCodeDataAll.select('ActionGeo_FullName','n_observations','if_normal_60d').filter(F.col('if_normal_60d') == False).count()