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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the sake of data storage, the *Weighted Average* of the Target Variables will be assessed, since the average numerical value per global event id per country per date was created in the previous preprocessing process.

# COMMAND ----------

# DBTITLE 1,Import PySpark Modules
from functools import reduce
from itertools import chain
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
preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Only Conflict Events
conflictEvents = preprocessedGDELT.filter(F.col('QuadClassString').isin('Verbal Conflict', 'Material Conflict'))
print((conflictEvents.count(), len(conflictEvents.columns)))
conflictEvents.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1) Event report value (ERV)
# MAGIC #### Calculated as the percentage of total articles categorized as belonging to a country that are categorized as matches for an event type

# COMMAND ----------

# DBTITLE 1,Create Column to Count Number of Daily Articles by Country by EventRootCode
nEventsDaily = conflictEvents.select('ActionGeo_FullName','EventTimeDate','EventRootCodeString','nArticles').groupBy('ActionGeo_FullName','EventTimeDate','EventRootCodeString').agg(F.sum('nArticles').alias('nArticles')).sort(['EventTimeDate', 'ActionGeo_FullName'], ascending=True)
print((nEventsDaily.count(), len(nEventsDaily.columns)))
nEventsDaily.select('nArticles').describe().show()
nEventsDaily.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Event Report Value (ERV)
# create a 'Window', country by date
nArticles_window = Window.partitionBy('ActionGeo_FullName').orderBy('EventTimeDate')

# get daily percent of articles for each Event Code string within 'Window'
nEventsDaily = nEventsDaily.withColumn('EventReportValue', F.col('nArticles')/F.sum('nArticles').over(nArticles_window))
nEventsDaily.limit(10).toPandas()

# COMMAND ----------

# verify output
AFG_01feb2021 = nEventsDaily.filter((F.col('ActionGeo_FullName') == 'Afghanistan') & (F.col('EventTimeDate') == '2021-02-01'))
print(AFG_01feb2021.select(F.sum('EventReportValue')).collect()[0][0])
AFG_01feb2021.limit(20).toPandas()

# COMMAND ----------

display(AFG_01feb2021)

# COMMAND ----------

# DBTITLE 1,Get Daily Percent of Articles by Event Type
nEventsDailyPercentages = nEventsDaily.select('ActionGeo_FullName','EventTimeDate',
                                              'EventRootCodeString','sum(nArticles)').groupBy('ActionGeo_FullName','EventTimeDate',
                                                                                              'EventRootCodeString').agg(F.sum(F.col('sum(nArticles)')),
                                                                                                                         
                                                                                                                        )
nEventsDailyPercentages.limit(10).toPandas()

# COMMAND ----------

nEventsDailyPercentagesT = nEventsDailyPercentages.withColumn('%', grouped_df['sum(sum(nArticles))'] / grouped_df['sum(nArticles)'].sum()) * 100

# COMMAND ----------

# DBTITLE 1,Display
display(streaming_df.groupBy().count(), processingTime = "5 seconds", checkpointLocation = "dbfs:/<checkpoint-path>")