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
# MAGIC ### Calculations – Percentages of Goldstein Scores
# MAGIC 
# MAGIC - 	Goldstein point value (GPV): 
# MAGIC Calculated as the average Goldstein score for all articles tagged as associated to the country
# MAGIC -	Goldstein Running Average (GRA1):
# MAGIC Calculated as the rolling average of the GPV for PA1 over the previous 12 months
# MAGIC -	Goldstein Running Average (GRA2):
# MAGIC Calculated as the rolling average of the GPV for PA2 over the previous 24 months
# MAGIC -	Period of Analysis 1 (PA1): 3 days
# MAGIC -	Period of Analysis 2 (PA2): 60 days 
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
# MAGIC -	Period of Analysis 1 (PA1): 3 days
# MAGIC -	Period of Analysis 2 (PA2): 60 days 
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
from pyspark.ml.feature import Bucketizer
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

# DBTITLE 1,Select Only Conflict Events
conflictEvents = preprocessedGDELT.filter(F.col('QuadClassString').isin('Verbal Conflict', 'Material Conflict'))
print((conflictEvents.count(), len(conflictEvents.columns)))
conflictEvents.limit(2).toPandas()

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
gdeltTargetOutput = conflictEvents.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').agg(F.avg('Confidence').alias('avgConfidence'),
                                                                                                      F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                                                                                      F.avg('MentionDocTone').alias('ToneReportValue'),
                                                                                                      F.sum('nArticles').alias('nArticles')
                                                                                                     ).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((nEventsDaily.count(), len(nEventsDaily.columns)))
nEventsDaily.select('nArticles').describe().show()
nEventsDaily.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Report Value (ERV)
# create a Window, country by date
countriesDaily_window = Window.partitionBy('ActionGeo_FullName').orderBy('EventTimeDate')

# get daily percent of articles for each Event Code string within Window
nEventsDailyPartitioned = nEventsDaily.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
nEventsDailyPartitioned.limit(2).toPandas()

# COMMAND ----------

# verify output
AFG_01feb2021 = nEventsDailyPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-01'))
print('All Report Values for One Country Per Day Should Sum to 100% (or 1)')
print(AFG_01feb2021.select(F.sum('EventReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('GoldsteinReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('ToneReportValue')).collect()[0][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2) Goldstein report value (GRV)
# MAGIC #### Calculated as the average Goldstein score for all articles tagged as associated to the country

# COMMAND ----------

# DBTITLE 1,Create Column to Count Number of Daily Articles by Country by EventRootCode
nEventsDaily = conflictEvents.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').agg(F.sum('nArticles').alias('nArticles')).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((nEventsDaily.count(), len(nEventsDaily.columns)))
nEventsDaily.select('nArticles').describe().show()
nEventsDaily.limit(3).toPandas()

# COMMAND ----------

# verify output
AFG_01feb2021 = nEventsDailyPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-01'))
print('All Report Values for One Country Per Day Should Sum to 100% (or 1)')
print(AFG_01feb2021.select(F.sum('EventReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('GoldsteinReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('ToneReportValue')).collect()[0][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3) Tone report value (TRV)
# MAGIC #### Calculated as the average Mention Tone for all articles tagged as associated to the country

# COMMAND ----------

# DBTITLE 1,Create Column to Count Number of Daily Articles by Country by EventRootCode
nEventsDaily = conflictEvents.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').agg(F.sum('nArticles').alias('nArticles')).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((nEventsDaily.count(), len(nEventsDaily.columns)))
nEventsDaily.select('nArticles').describe().show()
nEventsDaily.limit(3).toPandas()

# COMMAND ----------

# verify output
AFG_01feb2021 = nEventsDailyPartitioned.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-01'))
print('All Report Values for One Country Per Day Should Sum to 100% (or 1)')
print(AFG_01feb2021.select(F.sum('EventReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('GoldsteinReportValue')).collect()[0][0])
print(AFG_01feb2021.select(F.sum('ToneReportValue')).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Create Rolling Average Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 30 day Window, 30 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling30d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(30), 0)

# create a 3 day Window, 30 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# COMMAND ----------

# DBTITLE 1,Calculate ERV Rolling Average (ERA1, modified to 30 days)
# get 30d average of the Event Report Value (ERV) within Window
rollingERAs = nEventsDailyPartitioned.withColumn('ERA_30d', F.avg('EventReportValue').over(rolling30d_window)) 
rollingERAs = rollingERAs.withColumn('ERA_3d', F.avg('EventReportValue').over(rolling3d_window))
rollingERAs = rollingERAs.withColumn('difference', F.col('ERA_30d') - F.col('ERA_3d'))
rollingERAs.limit(10).toPandas()

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