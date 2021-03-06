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
# MAGIC #### (1) Event report value (ERV)
# MAGIC Calculated as the distribution of articles with respect to an event type category per country per day

# COMMAND ----------

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode 
gdeltTargetOutput = preprocessedGDELTT.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString', 'if_conflict').agg(F.avg('Confidence').alias('avgConfidence'),
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
countriesDaily_window = Window.partitionBy('EventTimeDate', 'ActionGeo_FullName').orderBy('EventTimeDate')

# get daily distribution of articles for each Event Code string within Window
gdeltTargetOutputPartitioned = gdeltTargetOutput.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(countriesDaily_window))
gdeltTargetOutputPartitioned.limit(2).toPandas()

# COMMAND ----------

# verify output
sumERV = gdeltTargetOutputPartitioned.select('EventTimeDate','ActionGeo_FullName','EventReportValue').groupBy('EventTimeDate', 'ActionGeo_FullName').agg(F.sum('EventReportValue'))
print('Verify all sum(EventReportValue)s are 1')
plt.plot(sumERV.select('sum(EventReportValue)').toPandas())

# COMMAND ----------

# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode 
# def addRow(pdf):
    # if  pdf.loc[~ pdf['EventRootCodeString'].isin(event_codes)]:
       #  return pdf.append(pd.Series([country, date, event_code, 0.0, 0.0, 0.0, 0], index=gdeltTargetOutput.columns), ignore_index=True)


# gdeltTargetOutputModified = gdeltTargetOutput.groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').apply(addRow).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
# print((gdeltTargetOutputModified.count(), len(gdeltTargetOutputModified.columns)))
# gdeltTargetOutputModified.limit(2).toPandas()

# COMMAND ----------

#gdeltTargetOutput.createOrReplaceTempView("targets")
#countryList = [x['ActionGeo_FullName'] for x in sqlContext.sql("select ActionGeo_FullName, ActionGeo_FullName, EventTimeDate from targets").rdd.collect()]



#def addRow(pdf):
   # if  pdf.loc[~ pdf['EventRootCodeString'].isin(event_codes)]:
       # return pdf.append(pd.Series([country, date, event_code, 0.0, 0.0, 0.0, 0], index=gdeltTargetOutput.columns), ignore_index=True)

#df.groupBy("id").apply(addRow).show()

# COMMAND ----------

# DBTITLE 0,Verify Event Code is Present per Day (*slower code*)
#event_codes = ['MAKE PUBLIC STATEMENT', 'APPEAL', 'EXPRESS INTENT TO COOPERATE', 'CONSULT', 'ENGAGE IN DIPLOMATIC COOPERATION', 'ENGAGE IN MATERIAL COOPERATION', 'PROVIDE AID', 'YIELD', 'INVESTIGATE', 'DEMAND', 'DISAPPROVE', 'REJECT', 'THREATEN', 'PROTEST', 'EXHIBIT MILITARY POSTURE', 'REDUCE RELATIONS', 'COERCE', 'ASSAULT', 'FIGHT', 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE']

##
#test = gdeltTargetOutput
##

#blank_list = [[country, date, event_code, 0.0, 0.0, 0.0, 0]]
#blank_df = spark.createDataFrame(blank_list, gdeltTargetOutput.schema)

#test.createOrReplaceTempView("targets")
#countryList = [x['ActionGeo_FullName'] for x in sqlContext.sql("select ActionGeo_FullName from targets").rdd.collect()]

#for country in countryList:
  #country_df = test.filter(F.col('ActionGeo_FullName') == country)
  #country_df.createOrReplaceTempView("country")
  #dateList = [x['EventTimeDate'] for x in sqlContext.sql("select EventTimeDate from country").rdd.collect()]
  
  #for date in dateList:
    #date_df = country_df.filter(F.col('EventTimeDate') == date)
    
    #for event_code in event_codes:
      #if date_df.filter(F.col('EventRootCodeString') == event_code) == True:
        #next
      #else:
        # append row to dataframe
       # blank_list = [[country, date, event_code, 0.0, 0.0, 0.0, 0]]
       # b_df = spark.createDataFrame(blank_list, gdeltTargetOutput.schema)
       # blank_df = test.union(b_df)

# COMMAND ----------

# DBTITLE 1,Create Rolling Average Windows
# function to calculate number of seconds from number of days
days = lambda i: i * 86400

# create a 3 day Window, 3 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling3d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(3), 0)

# create a 60 day Window, 60 days days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window = Window.partitionBy('ActionGeo_FullName', 'EventRootCodeString').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Event Report Value (ERV) within Country Window
# get WEIGHTED 3d average
weightedERA1 = gdeltTargetOutputPartitioned.withColumn('wERA_3d_num', F.sum(F.col('EventReportValue') * F.col('nArticles')).over(rolling3d_window))
weightedERA1 = weightedERA1.withColumn('wERA_3d_dem', F.sum('nArticles').over(rolling3d_window))
weightedERA1 = weightedERA1.withColumn('wERA_3d', F.col('wERA_3d_num')/F.col('wERA_3d_dem'))

# get WEIGHTED 60d average
weightedERA2 = weightedERA1.withColumn('wERA_60d_num', F.sum(F.col('EventReportValue') * F.col('nArticles')).over(rolling60d_window))
weightedERA2 = weightedERA2.withColumn('wERA_60d_dem', F.sum('nArticles').over(rolling60d_window))
weightedERA2 = weightedERA2.withColumn('wERA_60d', F.col('wERA_60d_num')/F.col('wERA_60d_dem'))

# drop extra columns
targetValueOutput = weightedERA2.drop('wERA_3d_num', 'wERA_3d_dem', 'wERA_60d_num', 'wERA_60d_dem')

# verify output data
print((targetValueOutput.count(), len(targetValueOutput.columns)))
targetValueOutput.limit(2).toPandas()

# COMMAND ----------

targetValueOutput.limit(12).toPandas()

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
gdeltTargetOutput2 = preprocessedGDELT.groupBy('ActionGeo_FullName','EventTimeDate', 'if_conflict').agg(F.avg('Confidence').alias('avgConfidence'),
                                                                                                      F.avg('GoldsteinScale').alias('GoldsteinReportValue'),
                                                                                                      F.avg('MentionDocTone').alias('ToneReportValue'),
                                                                                                      F.sum('nArticles').alias('nArticles')
                                                                                                     ).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((gdeltTargetOutput2.count(), len(gdeltTargetOutput2.columns)))
gdeltTargetOutput2.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Rolling Average Windows
# create a 3 day Window, 1 day previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling1d_window = Window.partitionBy('ActionGeo_FullName', 'if_conflict').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(1), 0)

# create a 60 day Window, 60 days previous to the current day (row), using previous casting of timestamp to long (number of seconds)
rolling60d_window2 = Window.partitionBy('ActionGeo_FullName', 'if_conflict').orderBy(F.col('EventTimeDate').cast('timestamp').cast('long')).rangeBetween(-days(60), 0)

# COMMAND ----------

# DBTITLE 1,Goldstein Report Value (GRV) within Country Window
# get WEIGHTED 1d average
weightedGRA1 = gdeltTargetOutput2.withColumn('wGRA_1d_num', F.sum(F.col('GoldsteinReportValue') * F.col('nArticles')).over(rolling1d_window))
weightedGRA1 = weightedGRA1.withColumn('wGRA_1d_dem', F.sum('nArticles').over(rolling1d_window))
weightedGRA1 = weightedGRA1.withColumn('wGRA_1d', F.col('wGRA_1d_num')/F.col('wGRA_1d_dem'))

# get WEIGHTED 60d average
weightedGRA2 = weightedGRA1.withColumn('wGRA_60d_num', F.sum(F.col('GoldsteinReportValue') * F.col('nArticles')).over(rolling60d_window2))
weightedGRA2 = weightedGRA2.withColumn('wGRA_60d_dem', F.sum('nArticles').over(rolling60d_window2))
weightedGRA2 = weightedGRA2.withColumn('wGRA_60d', F.col('wGRA_60d_num')/F.col('wGRA_60d_dem'))

# drop extra columns
weightedGRA2 = weightedGRA2.drop('wGRA_1d_num', 'wGRA_1d_dem', 'wGRA_60d_num', 'wGRA_60d_dem')
weightedGRA2.limit(2).toPandas()

# COMMAND ----------

weightedGRA2.limit(12).toPandas()

# COMMAND ----------

# DBTITLE 1,Tone Report Value (TRV) within Country Window
# get WEIGHTED 1d average
weightedTRA1 = weightedGRA2.withColumn('wTRA_1d_num', F.sum(F.col('ToneReportValue') * F.col('nArticles')).over(rolling1d_window))
weightedTRA1 = weightedTRA1.withColumn('wTRA_1d_dem', F.sum('nArticles').over(rolling1d_window))
weightedTRA1 = weightedTRA1.withColumn('wTRA_1d', F.col('wTRA_1d_num')/F.col('wTRA_1d_dem'))

# get WEIGHTED 60d average
weightedTRA2 = weightedTRA1.withColumn('wTRA_60d_num', F.sum(F.col('ToneReportValue') * F.col('nArticles')).over(rolling60d_window2))
weightedTRA2 = weightedTRA2.withColumn('wTRA_60d_dem', F.sum('nArticles').over(rolling60d_window2))
weightedTRA2 = weightedTRA2.withColumn('wTRA_60d', F.col('wTRA_60d_num')/F.col('wTRA_60d_dem'))

# drop extra columns
targetValueOutput2 = weightedTRA2.drop('wTRA_1d_num', 'wTRA_1d_dem', 'wTRA_60d_num', 'wTRA_60d_dem')

# verify output data
print((targetValueOutput2.count(), len(targetValueOutput2.columns)))
targetValueOutput2.limit(2).toPandas()

# COMMAND ----------

targetValueOutput2.filter((F.col('ActionGeo_FullName') == 'Afghanistan')).orderBy('EventTimeDate').limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Save Target Data as CSV
targetValueOutput.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/targetvalues_confidence40plus.csv')
targetValueOutput2.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/gold_tone_targetvalues_confidence40plus.csv')