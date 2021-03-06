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
# MAGIC Sources:
# MAGIC - (1) [Moving Averaging with Apache Spark](https://www.linkedin.com/pulse/time-series-moving-average-apache-pyspark-laurent-weichberger/)

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

# filter on flagged countries from Horizan Scan
conflictEventsHorizonCountries = preprocessedGDELT.filter(F.col('ActionGeo_FullName').isin('Afghanistan','Guinea','Myanmar','Somalia'))
print((conflictEventsHorizonCountries.count(), len(conflictEventsHorizonCountries.columns)))
conflictEventsHorizonCountries.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Events Data
eventsData = preprocessedGDELT.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence',
                                          'EventReportValue','ERA_3d','ERA_60d','weightedERA_3d','weightedERA_60d')

print((eventsData.count(), len(eventsData.columns)))
eventsData.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Explore Values in Conflict vs Not Situations
# create conflict column
conflict_events = ['DEMAND','DISAPPROVE','PROTEST','REJECT','THREATEN','ASSAULT','COERCE','ENGAGE IN UNCONVENTIONAL MASS VIOLENCE','EXHIBIT MILITARY POSTURE','FIGHT','REDUCE RELATIONS']
eventsData = eventsData.withColumn('if_conflict', F.when(F.col('EventRootCodeString').isin(conflict_events), True).otherwise(False))

# COMMAND ----------

# DBTITLE 1,Separate by Country
AFG = eventsData.filter((F.col('ActionGeo_FullName') == 'Afghanistan'))
MMR = eventsData.filter((F.col('ActionGeo_FullName') == 'Myanmar'))
SOM = eventsData.filter((F.col('ActionGeo_FullName') == 'Somalia'))
GIN = eventsData.filter((F.col('ActionGeo_FullName') == 'Guinea'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### AFG

# COMMAND ----------

#AFG = AFG.toPandas()

plt.subplots(figsize=(16,8))
plt.plot(AFG['EventReportValue'], label='Daily Return')
plt.plot(AFG['ERA_3d'], label='Rolling 3d Mean')
plt.plot(AFG['weightedERA_3d'], label = 'Rolling Weighted 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------

AFG['EventReportValueExpanding'] = AFG['EventReportValue'].expanding().mean()
AFG['EventReportValueRolling3d'] = AFG['EventReportValue'].rolling(3).mean()

plt.subplots(figsize=(16,8))
plt.plot(AFG['EventReportValue'], label='Daily Return')
plt.plot(AFG['EventReportValueExpanding'], label='Expanding')
plt.plot(AFG['EventReportValueRolling3d'], label = 'Rolling 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MMR

# COMMAND ----------

MMR = MMR.toPandas()
plt.subplots(figsize=(16,8))
plt.plot(MMR['EventReportValue'], label='Daily Return')
plt.plot(MMR['ERA_3d'], label='Rolling 3d Mean')
plt.plot(MMR['weightedERA_3d'], label = 'Rolling Weighted 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------

MMR['EventReportValueExpanding'] = MMR['EventReportValue'].expanding().mean()
MMR['EventReportValueRolling3d'] = MMR['EventReportValue'].rolling(3).mean()

plt.subplots(figsize=(16,8))
plt.plot(MMR['EventReportValue'], label='Daily Return')
plt.plot(MMR['EventReportValueExpanding'], label='Expanding')
plt.plot(MMR['EventReportValueRolling3d'], label = 'Rolling 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SOM

# COMMAND ----------

SOM = SOM.toPandas()
plt.subplots(figsize=(16,8))
plt.plot(SOM['EventReportValue'], label='Daily Return')
plt.plot(SOM['ERA_3d'], label='Rolling 3d Mean')
plt.plot(SOM['weightedERA_3d'], label = 'Rolling Weighted 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------

SOM['EventReportValueExpanding'] = SOM['EventReportValue'].expanding().mean()
SOM['EventReportValueRolling3d'] = SOM['EventReportValue'].rolling(3).mean()

plt.subplots(figsize=(16,8))
plt.plot(SOM['EventReportValue'], label='Daily Return')
plt.plot(SOM['EventReportValueExpanding'], label='Expanding')
plt.plot(SOM['EventReportValueRolling3d'], label = 'Rolling 3d Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

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

def eda_funcs(col, df): 
  
  # get lists
  list_conflict = df.select(col).rdd.flatMap(lambda x: x).collect()
  
  # get quantiles
  print('Get Conflict Quantiles for ' + col)
  get_quantiles(df, col)
  plot_boxplot(list_conflict, col)
  
  # plot dist
  plot_dist(df, col)
  
  # plot ecdf
  plot_ecdf(list_conflict, 'CONFLICT ' + col)
  
  return list_conflict

# COMMAND ----------

last20d = eventsDataNonConflict.filter(F.col('EventTimeDate') >= '2021-03-02').select('ActionGeo_FullName','EventTimeDate','weightedERA_3d').toPandas()
last20d.head()

# COMMAND ----------

plt.subplots(figsize=(8,6))
plt.plot(stock_df['daily_return'], label='Daily Return')
plt.plot(stock_df['expand_mean'], label='Expanding Mean')
plt.plot(stock_df['roll_mean_3'], label = 'Rolling Mean')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.show()

# COMMAND ----------



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
# MAGIC ### ERV

# COMMAND ----------

erv_conflict = eda_funcs('EventReportValue', eventsDataConflict)

# COMMAND ----------

erv_nonconflict = eda_funcs('EventReportValue', eventsDataNonConflict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ERA 3D

# COMMAND ----------

ERA_3d_conflict = eda_funcs('ERA_3d', eventsDataConflict)

# COMMAND ----------

 ERA_3d_nonconflict = eda_funcs('ERA_3d', eventsDataNonConflict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ERA 60D

# COMMAND ----------

ERA_60d_conflict = eda_funcs('ERA_60d', eventsDataConflict)

# COMMAND ----------

ERA_60d_nonconflict = eda_funcs('ERA_60d', eventsDataNonConflict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 3D

# COMMAND ----------

weightedERA_3d_conflict = eda_funcs('weightedERA_3d', eventsDataConflict)

# COMMAND ----------

 weightedERA_3d_nonconflict = eda_funcs('weightedERA_3d', eventsDataNonConflict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weighted ERA 60D

# COMMAND ----------

weightedERA_60d_conflict = eda_funcs('weightedERA_60d', eventsDataConflict)

# COMMAND ----------

weightedERA_60d_nonconflict = eda_funcs('weightedERA_60d', eventsDataNonConflict)

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

# MAGIC %md
# MAGIC 
# MAGIC #### Option A - Apply Kruskal-Wallis H-test

# COMMAND ----------

# ERV
stats.kruskal(erv_conflict, erv_nonconflict)

# COMMAND ----------

# ERA_3d
stats.kruskal(weightedERA_3d_conflict, weightedERA_3d_nonconflict)

# COMMAND ----------

# ERA_60d
stats.kruskal(weightedERA_60d_conflict, weightedERA_60d_nonconflict)

# COMMAND ----------

# ERA_60d
stats.kruskal(weightedERA_3d_conflict, weightedERA_3d_nonconflict, weightedERA_60d_conflict, weightedERA_60d_nonconflict)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Preliminary Conclusion**: While this isn't the final output comparrison, this method does not appear to be incredibly informative. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Option B - Transform Data with Arcsine

# COMMAND ----------

arcsine = F.udf(lambda x: float(np.arcsin(x)), FloatType())
spark.udf.register("arcsine", arcsine)

eventsDataNonConflict.createOrReplaceTempView('test')
test_list = spark.sql("select arcsine(weightedERA_3d) from test")#.select('arcsine(weightedERA_3d)').rdd.map(lambda x: x[0])
# = eventsDataNonConflict.withColumn('arcsine_weightedERA_3d', test_list)

test = eventsDataNonConflict.join(test_list)
test_list = eda_funcs('arcsine(weightedERA_3d)', test)

# COMMAND ----------

test.limit(10).toPandas()

# COMMAND ----------

eventsDataConflict // eventsDataNonConflict

# COMMAND ----------

test_list = eda_funcs('arcsine(weightedERA_3d)', test)

# COMMAND ----------

arcsine = F.udf(lambda x: float(np.arcsin(x)), FloatType())
spark.udf.register("arcsine", arcsine)
eventsDataNonConflict.createOrReplaceTempView('test')

# COMMAND ----------

test_list = spark.sql("select arcsine(EventReportValue) from test")
test_list.limit(10).toPandas()

# COMMAND ----------

test_list2 = eda_funcs('arcsine(EventReportValue)', test_list)