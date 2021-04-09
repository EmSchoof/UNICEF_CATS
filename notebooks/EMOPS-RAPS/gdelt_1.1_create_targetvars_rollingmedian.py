# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC ### Periods of Analysis
# MAGIC - Period of Analysis 1 (PA1): 3 days
# MAGIC - Period of Analysis 2 (PA2): 60 days 
# MAGIC - Period of Analysis 3 (PA2): 1 days 
# MAGIC 
# MAGIC ### Premise of Task
# MAGIC - Anomaly detection of target variables
# MAGIC 
# MAGIC ### Calculations – Distribution of Articles by Country by Event Type
# MAGIC 
# MAGIC - 	Event report value (ERV): 
# MAGIC Calculated as the distribution of articles with respect to an event type category per country per day
# MAGIC -   Event report sum (ERS):
# MAGIC <strike> Calculated as the number of articles categorized as belonging to a country that are categorized as matches for an event type </strike>
# MAGIC -	Event Running Average 1 (ERA1):
# MAGIC Calculated as the rolling average of the ERV for 3 days over the previous 12 months
# MAGIC -	Event Running Average 2 (ERA2):
# MAGIC Calculated as the rolling average of the ERV for 60 days over the previous 24 months
# MAGIC 
# MAGIC 
# MAGIC ### Calculations – Averages of Goldstein Scores
# MAGIC 
# MAGIC - 	Goldstein point value (GPV): 
# MAGIC Calculated as the average Goldstein score for all articles tagged as associated to the country
# MAGIC -	Goldstein Running Average (GRA1):
# MAGIC Calculated as the rolling average of the GPV for PA13 over the previous 12 months
# MAGIC -	Goldstein Running Average (GRA2):
# MAGIC Calculated as the rolling average of the GPV for 60 days over the previous 24 months
# MAGIC 
# MAGIC 
# MAGIC ### Calculations – Averages of Tone Scores
# MAGIC 
# MAGIC - 	Tone point value (TPV): 
# MAGIC Calculated as the average Mention Tone for all articles tagged as associated to the country
# MAGIC -	Tone Running Average (TRA1):
# MAGIC Calculated as the rolling average of the TPV for PA1 over the previous 12 months
# MAGIC -	Tone Running Average (TRA2):
# MAGIC Calculated as the rolling average of the TPV for 60 days over the previous 24 months
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
preprocessedGDELT = preprocessedGDELT.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.agg(F.countDistinct(F.col("GLOBALEVENTID")).alias("nEvents")).show()
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Add Conflict 'flag' to data
# create conflict column
conflict_events = ['DEMAND','DISAPPROVE','PROTEST','REJECT','THREATEN','ASSAULT','COERCE','ENGAGE IN UNCONVENTIONAL MASS VIOLENCE','EXHIBIT MILITARY POSTURE','FIGHT','REDUCE RELATIONS']
preprocessedGDELT = preprocessedGDELT.withColumn('if_conflict', F.when(F.col('EventRootCodeString').isin(conflict_events), True).otherwise(False))
preprocessedGDELT.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Data with Confidence of 40% or higher
# create confidence column of more than 
print(preprocessedGDELT.count())
preprocessedGDELTT = preprocessedGDELT.filter(F.col('Confidence') >= 40)
print(preprocessedGDELTT.count())
preprocessedGDELTT.limit(2).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define function for median

# COMMAND ----------

median_udf = udf(lambda x: float(np.median(x)), FloatType())

# COMMAND ----------

# MAGIC %md
# MAGIC #### (1) Event report value (ERV)
# MAGIC Calculated as the distribution of articles with respect to an event type category per country per day

# COMMAND ----------

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode 
ervOutput = preprocessedGDELTT.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString', 'if_conflict').agg(F.avg('Confidence').alias('avgConfidence'),
                                                                                                      F.sum('nArticles').alias('nArticles')
                                                                                                     ).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((ervOutput.count(), len(ervOutput.columns)))
ervOutput.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Event Report Value (ERV)
# create a Window, country by date
countriesDaily_window = Window.partitionBy('EventTimeDate', 'ActionGeo_FullName').orderBy('EventTimeDate')

# get daily distribution of articles for each Event Code string within Window
ervOutputPartitioned = ervOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
ervOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# verify output
sumERV = ervOutputPartitioned.select('EventTimeDate','ActionGeo_FullName','EventReportValue').groupBy('EventTimeDate', 'ActionGeo_FullName').agg(F.sum('EventReportValue'))
print('Verify all sum(EventReportValue)s are 1')
plt.plot(sumERV.select('sum(EventReportValue)').toPandas())

# COMMAND ----------

# DBTITLE 1,Create Rolling Median Windows (ERV)
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Event Report Value (ERV) within Country Window
# get rolling 3d median
ervOutputMedian1 = ervOutputPartitioned.withColumn('ERV_3d_list', F.collect_list('EventReportValue').over(rolling3d_window)) \
                                        .withColumn('ERV_3d_median', median_udf('ERV_3d_list'))

# get rolling 60d median
ervOutputMedians = ervOutputMedian1.withColumn('ERV_60d_list', F.collect_list('EventReportValue').over(rolling60d_window)) \
                                        .withColumn('ERV_60d_median', median_udf('ERV_60d_list'))

# verify output data
print((ervOutputMedians.count(), len(ervOutputMedians.columns)))
ervOutputMedians.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### (2) Goldstein report value (GRV)
# MAGIC Calculated as the average Goldstein score for all articles tagged as associated to the country
# MAGIC 
# MAGIC #### (3) Tone report value (TRV)
# MAGIC Calculated as the average Mention tone for all articles tagged as associated to the country (within 30 days of event)

# COMMAND ----------

# DBTITLE 1,Get Target Variables: TRV and GRV
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode 
goldToneOutput = preprocessedGDELT.groupBy('ActionGeo_FullName','EventTimeDate', 'if_conflict').agg(F.avg('Confidence').alias('avgConfidence'),
                                                                                                      F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                                                                                      F.avg('MentionDocTone').alias('ToneReportValue'),
                                                                                                      F.sum('nArticles').alias('nArticles')
                                                                                                     ).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((goldToneOutput.count(), len(goldToneOutput.columns)))
goldToneOutput.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Rolling Average Windows
# create a 1 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName', 'if_conflict').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 60 day Window, 60 days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window2 = Window.partitionBy('ActionGeo_FullName', 'if_conflict').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Goldstein Report Value (GRV) within Country Window
# get rolling 1d median
goldsteinOutputMedian1 = goldToneOutput.withColumn('GRV_1d_list', F.collect_list('GoldsteinReportValue').over(rolling1d_window)) \
                                        .withColumn('GRV_1d_median', median_udf('GRV_1d_list'))

# get rolling 60d median
goldsteinOutputMedians = goldsteinOutputMedian1.withColumn('GRV_60d_list', F.collect_list('GoldsteinReportValue').over(rolling60d_window2)) \
                                        .withColumn('GRV_60d_median', median_udf('GRV_60d_list'))

# verify output data
print((goldsteinOutputMedians.count(), len(goldsteinOutputMedians.columns)))
goldsteinOutputMedians.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Tone Report Value (TRV) within Country Window
# get rolling 1d median
toneOutputMedian1 = goldToneOutput.withColumn('TRV_1d_list', F.collect_list('ToneReportValue').over(rolling1d_window)) \
                                        .withColumn('TRV_1d_median', median_udf('TRV_1d_list'))

# get rolling 60d median
toneOutputMedians = toneOutputMedian1.withColumn('TRV_60d_list', F.collect_list('ToneReportValue').over(rolling60d_window2)) \
                                        .withColumn('TRV_60d_median', median_udf('TRV_60d_list'))

# verify output data
print((toneOutputMedians.count(), len(toneOutputMedians.columns)))
toneOutputMedians.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Save Target Data as CSV
#ervOutputMedians.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/erv_confidence40plus.csv')
#goldsteinOutputMedians.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/grv_confidence40plus.csv')
#toneOutputMedians.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/trv_confidence40plus.csv')