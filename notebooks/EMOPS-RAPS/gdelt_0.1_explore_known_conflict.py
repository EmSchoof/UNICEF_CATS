# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## As of February 2021, the following countries have, in addition to the COVID-19 Pandemic, experienced conflict events and have been categorized as Level 3 Seriousness:
# MAGIC - (1) **Afghanistan**
# MAGIC   - Factionalized Civil War
# MAGIC   - Urban Conflict
# MAGIC   - Environmental Drought
# MAGIC - (2) **Myanmar**
# MAGIC   - Armed Conflict
# MAGIC   - Protests and Civil Unrest
# MAGIC - (3) **Somalia**
# MAGIC   - Non-State Armed Group Attacks
# MAGIC   - Civil Unrest
# MAGIC   - Intercommunal Violance
# MAGIC - (4) **Guinea**
# MAGIC   - Ebola Outbreak
# MAGIC 
# MAGIC The script will be used to visual explore the GDELT February data associated with the above countries to confirm accurate detection of conflict events.

# COMMAND ----------

# DBTITLE 1,Import PySpark Functions
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

# DBTITLE 1,Optimize CSV
OPTIMIZE preprocessedGDELT
  ZORDER BY (ActionGeo_FullName, EventTimeDate)

# COMMAND ----------

# DBTITLE 1,Select Only Conflict Events in Specific Countries
# filter on conflict events
conflictEvents = preprocessedGDELT.filter(F.col('QuadClassString').isin('Verbal Conflict', 'Material Conflict'))
print((conflictEvents.count(), len(conflictEvents.columns)))

# filter on flagged countries from Horizan Scan
conflictEventsHorizonCountries = conflictEvents.filter(F.col('ActionGeo_FullName').isin('Afghanistan','Guinea','Myanmar','Somalia'))
print((conflictEventsHorizonCountries.count(), len(conflictEventsHorizonCountries.columns)))

# COMMAND ----------

# Cast Event Date columns as Date and Select Data for the Month of February
conflictEventsHorizonCountries = conflictEventsHorizonCountries.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
conflictEventsHorizonCountries = conflictEventsHorizonCountries.filter( (F.col('EventTimeDate') >= F.lit('2021-02-01')) & (F.col('EventTimeDate') < F.lit('2021-03-01' )))
conflictEventsHorizonCountries.select('EventTimeDate').describe().show()
conflictEventsHorizonCountries.limit(10).toPandas()

# COMMAND ----------

display(conflictEventsHorizonCountries)

# COMMAND ----------

display(conflictEventsHorizonCountries)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Afghanistan
# MAGIC   - Factionalized Civil War
# MAGIC   - Urban Conflict
# MAGIC   - Environmental Drought

# COMMAND ----------

AFG = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Afghanistan'))
print('February Global CONFLICT Events in Afghanistan: ', AFG.select('GLOBALEVENTID').distinct().count())
AFG.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 0,Afghanistan
display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Myanmar
# MAGIC - Armed Conflict
# MAGIC - Protests and Civil Unrest

# COMMAND ----------

MMR = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Myanmar'))
print('February Global CONFLICT Events in Myanmar: ', MMR.select('GLOBALEVENTID').distinct().count())
MMR.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 0,Myanmar
display(MMR)

# COMMAND ----------

display(MMR)

# COMMAND ----------

display(MMR)

# COMMAND ----------

display(MMR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Somalia**
# MAGIC   - Non-State Armed Group Attacks
# MAGIC   - Civil Unrest
# MAGIC   - Intercommunal Violance

# COMMAND ----------

SOM = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Somalia'))
print('February Global CONFLICT Events in Somalia: ', SOM.select('GLOBALEVENTID').distinct().count())
SOM.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 0,Somalia
display(SOM)

# COMMAND ----------

display(SOM)

# COMMAND ----------

display(SOM)

# COMMAND ----------

display(SOM)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### **Guinea**
# MAGIC   - Ebola Outbreak

# COMMAND ----------

GIN = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Guinea'))
print('February Global CONFLICT Events in Guinea: ', GIN.select('GLOBALEVENTID').distinct().count())
GIN.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Guinea
display(GIN)

# COMMAND ----------

display(GIN)

# COMMAND ----------

display(GIN)

# COMMAND ----------

display(GIN)

# COMMAND ----------

GIN.limit(25).toPandas()