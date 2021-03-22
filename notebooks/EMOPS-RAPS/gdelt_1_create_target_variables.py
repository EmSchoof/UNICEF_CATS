# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Periods of Analysis
# MAGIC - Period of Analysis 1 (PA1): 3 days
# MAGIC - Period of Analysis 2 (PA2): 60 days 
# MAGIC 
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
# MAGIC -	Event spike alert: 
# MAGIC When the *Event Report Value* for a given PA1 (*3 DAYS*) is one standard deviation above ERA1* 
# MAGIC -	Event trend alert: 
# MAGIC when the *Event Report Value* for a given PA2 (*60 DAYS*) is one standard deviation above ERA2*
# MAGIC 
# MAGIC 
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
# MAGIC 
# MAGIC 
# MAGIC ### Calculations – Percentages of Tone Scores
# MAGIC 
# MAGIC - 	Tone point value (TPV): 
# MAGIC Calculated as the average Mention Tone for all articles tagged as associated to the country
# MAGIC -	Goldstein Running Average (TRA1):
# MAGIC Calculated as the rolling average of the TPV for PA1 over the previous 12 months
# MAGIC -	Goldstein Running Average (TRA2):
# MAGIC Calculated as the rolling average of the TPV for PA2 over the previous 24 months
# MAGIC -	Tone spike alert: 
# MAGIC When the *Tone Point Value* for a given PA1 (*3 DAYS*) is one standard deviation above ERA1* 
# MAGIC -	Tone trend alert: 
# MAGIC when the *Tone Point Value* for a given PA2 (*60 DAYS*) is one standard deviation above ERA2*
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

# COMMAND ----------

# DBTITLE 1,Import Preprocessed Data
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
preprocessedGDELT.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Convert Event Date Column from Timestamp to Date
preprocessedGDELT = preprocessedGDELT.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### (1) Event report value (ERV)
# MAGIC Calculated as the percentage of total articles categorized as belonging to a country that are categorized as matches for an event type
# MAGIC 
# MAGIC #### (2) Goldstein report value (GRV)
# MAGIC Calculated as the average Goldstein score for all articles tagged as associated to the country
# MAGIC 
# MAGIC #### (3) Tone report value (TRV)
# MAGIC Calculated as the average Mention tone for all articles tagged as associated to the country (within 30 days of event)

# COMMAND ----------

# DBTITLE 1,Get Target Variables
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode 
gdeltTargetOutput = preprocessedGDELT.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').agg(F.avg('Confidence').alias('avgConfidence'),
                                                                                                      F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                                                                                      F.avg('MentionDocTone').alias('ToneReportValue'),
                                                                                                      F.sum('nArticles').alias('nArticles')
                                                                                                     ).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((gdeltTargetOutput.count(), len(gdeltTargetOutput.columns)))
gdeltTargetOutput.select('nArticles').describe().show()
gdeltTargetOutput.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Event Report Value (ERV)
# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName', 'EventTimeDate').orderBy('EventTimeDate')

# get daily percent of articles for each Event Code string within Window
gdeltTargetOutputPartitioned = gdeltTargetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
gdeltTargetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# verify output
AFG_01feb2021 = gdeltTargetOutputPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-01'))
AFG_02feb2021 = gdeltTargetOutputPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-02'))
AFG_03feb2021 = gdeltTargetOutputPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-03'))
print('Event Report Values for One Country Per Day Should Sum to 100% (or 1)')
print(AFG_01feb2021.select(F.sum('EventReportValue')).collect()[0][0])
print(AFG_02feb2021.select(F.sum('EventReportValue')).collect()[0][0])
print(AFG_03feb2021.select(F.sum('EventReportValue')).collect()[0][0])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Create Running Average (*RA1) and (*RA2): Calculated as the rolling average of the Values for PA1 and PA2

# COMMAND ----------

# DBTITLE 1,Create Rolling Average Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Calculate Rolling Average (RA1/RA2)
# get 3d/60d average of the Event Report Value (ERV) within Window
rollingERAs1 = gdeltTargetOutputPartitioned.withColumn('ERA_3d', F.avg('EventReportValue').over(rolling3d_window)) 
rollingERAs = rollingERAs1.withColumn('ERA_60d', F.avg('EventReportValue').over(rolling60d_window))

# get 3d/60d average of the Golstein Report Value (GRV) within Window
rollingGRVs1 = rollingERAs.withColumn('GRA_3d', F.avg('GoldsteinReportValue').over(rolling3d_window)) 
rollingGRVs = rollingGRVs1.withColumn('GRA_60d', F.avg('GoldsteinReportValue').over(rolling60d_window))

# get 3d/60d average of the Tone Report Value (TRV) within Window
rollingTRVs = rollingGRVs.withColumn('TRA_3d', F.avg('ToneReportValue').over(rolling3d_window)) 
rollingAvgsAll = rollingTRVs.withColumn('TRA_60d', F.avg('ToneReportValue').over(rolling60d_window))
rollingAvgsAll.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Weighted Averages of Rolling Values
def weighted_avg(original_avg, sample_n):
  return np.sum(original_avg * sample_n) / np.sum(sample_n)


weightedAvg = F.udf(lambda col: F.sum(F.col(col) * F.col('nArticles')) / F.sum('nArticles'))

# COMMAND ----------

# get 3d WEIGHTED average of the Event Report Value (ERV) within Country Window
weightedERA1 = rollingAvgsAll.withColumn('weightedERA_3d_num', F.sum(F.col('ERA_3d') * F.col('nArticles')).over(rolling3d_window))
weightedERA1 = weightedERA1.withColumn('weightedERA_3d_dem', F.sum('nArticles').over(rolling3d_window))
weightedERA1 = weightedERA1.withColumn('weightedERA_3d', F.col('weightedERA_3d_num')/F.col('weightedERA_3d_dem'))
# 60d
weightedERA2 = weightedERA1.withColumn('weightedERA_60d_num', F.sum(F.col('ERA_60d') * F.col('nArticles')).over(rolling60d_window))
weightedERA2 = weightedERA2.withColumn('weightedERA_60d_dem', F.sum('nArticles').over(rolling3d_window))
weightedERA2 = weightedERA2.withColumn('weightedERA_60d', F.col('weightedERA_60d_num')/F.col('weightedERA_60d_dem'))

# get 3d WEIGHTED average of the Goldstein Report Value (GRV) within Country Window
weightedGRA1 = weightedERA2.withColumn('weightedGRA_3d_num', F.sum(F.col('GRA_3d') * F.col('nArticles')).over(rolling3d_window))
weightedGRA1 = weightedGRA1.withColumn('weightedGRA_3d_dem', F.sum('nArticles').over(rolling3d_window))
weightedGRA1 = weightedGRA1.withColumn('weightedGRA_3d', F.col('weightedGRA_3d_num')/F.col('weightedGRA_3d_dem'))
# 60d
weightedGRA2 = weightedGRA1.withColumn('weightedGRA_60d_num', F.sum(F.col('GRA_60d') * F.col('nArticles')).over(rolling60d_window))
weightedGRA2 = weightedGRA2.withColumn('weightedGRA_60d_dem', F.sum('nArticles').over(rolling3d_window))
weightedGRA2 = weightedGRA2.withColumn('weightedGRA_60d', F.col('weightedGRA_60d_num')/F.col('weightedGRA_60d_dem'))

# get 3d/60d WEIGHTED average of the Tone Report Value (GRV) within Country Window
weightedTRA1 = weightedGRA2.withColumn('weightedTRA_3d_num', F.sum(F.col('TRA_3d') * F.col('nArticles')).over(rolling3d_window))
weightedTRA1 = weightedTRA1.withColumn('weightedTRA_3d_dem', F.sum('nArticles').over(rolling3d_window))
weightedTRA1 = weightedTRA1.withColumn('weightedTRA_3d', F.col('weightedTRA_3d_num')/F.col('weightedTRA_3d_dem'))
# 60d
weightedTRA2 = weightedTRA1.withColumn('weightedTRA_60d_num', F.sum(F.col('TRA_60d') * F.col('nArticles')).over(rolling60d_window))
weightedTRA2 = weightedTRA2.withColumn('weightedTRA_60d_dem', F.sum('nArticles').over(rolling3d_window))
weightedRollingAvgs = weightedTRA2.withColumn('weightedTRA_60d', F.col('weightedTRA_60d_num')/F.col('weightedTRA_60d_dem'))
weightedRollingAvgs.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Output Data
targetValueOutput = weightedRollingAvgs.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles','avgConfidence',
                                          'GoldsteinReportValue','ToneReportValue','EventReportValue','weightedERA_3d','weightedERA_60d',
                                          'weightedGRA_3d','weightedGRA_60d','weightedTRA_3d','weightedTRA_60d')

print((targetValueOutput.count(), len(targetValueOutput.columns)))
targetValueOutput.limit(2).toPandas()

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

# COMMAND ----------

# DBTITLE 1,EventReportValue
# select variables to check correlation
df_features = targetValueOutput.select('avgConfidence','EventReportValue','weightedERA_3d','weightedERA_60d') 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
plot_corr_matrix(corr_mat, df_features.columns, 234)

# COMMAND ----------

# DBTITLE 1,GoldsteinReportValue
# select variables to check correlation
df_features = targetValueOutput.select('avgConfidence','GoldsteinReportValue','weightedGRA_3d','weightedGRA_60d') 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
plot_corr_matrix(corr_mat, df_features.columns, 234)

# COMMAND ----------

# DBTITLE 1,ToneReportValue
# select variables to check correlation
df_features = targetValueOutput.select('avgConfidence','ToneReportValue','weightedTRA_3d','weightedTRA_60d') 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
plot_corr_matrix(corr_mat, df_features.columns, 234)

# COMMAND ----------

# DBTITLE 1,Overall
# select variables to check correlation
df_features = targetValueOutput.select('weightedERA_3d','weightedERA_60d','weightedGRA_3d','weightedGRA_60d','weightedTRA_3d','weightedTRA_60d') 

# create RDD table for correlation calculation
rdd_table = df_features.rdd.map(lambda row: row[0:])

# get the correlation matrix
corr_mat=Statistics.corr(rdd_table, method="pearson")
plot_corr_matrix(corr_mat, df_features.columns, 234)

# COMMAND ----------

# DBTITLE 1,Save Target Data as CSV
targetValueOutput.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/targetvalues.csv')