# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## As of February 2021, the following countries have, in addition to the COVID-19 Pandemic, experienced conflict events and have been categorized as Level 3 Seriousness:
# MAGIC - (1) **Afghanistan**
# MAGIC   - Factionalized Civil War
# MAGIC   - Urban Conflict
# MAGIC   - Environmental Drought
# MAGIC - (2) **Muanmar**
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

# DBTITLE 1,Select Only Conflict Events in Specific Countries
# filter on conflict events
conflictEvents = preprocessedGDELT.filter(F.col('QuadClassString').isin('Verbal Conflict', 'Material Conflict'))
print((conflictEvents.count(), len(conflictEvents.columns)))

# filter on flagged countries from Horizan Scan
conflictEventsHorizonCountries = conflictEvents.filter(F.col('ActionGeo_FullName').isin('Afghanistan','Guinea','Myanmar','Somalia')).drop_duplicates()
print((conflictEventsHorizonCountries.count(), len(conflictEventsHorizonCountries.columns)))
conflictEventsHorizonCountries.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Afghanistan
# MAGIC   - Factionalized Civil War
# MAGIC   - Urban Conflict
# MAGIC   - Environmental Drought

# COMMAND ----------

# DBTITLE 0,Afghanistan
AFG = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Afghanistan'))
print('February Global CONFLICT Events in Afghanistan: ', AFG.count())
AFG.limit(5).toPandas()
display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

display(AFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Muanmar
# MAGIC - Armed Conflict
# MAGIC - Protests and Civil Unrest

# COMMAND ----------

# DBTITLE 0,Myanmar
MMR = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Myanmar'))
print('February Global CONFLICT Events in Myanmar: ', MMR.count())
MMR.limit(5).toPandas()
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

# DBTITLE 0,Somalia
SOM = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Somalia'))
print('February Global CONFLICT Events in Somalia: ', SOM.count())
SOM.limit(5).toPandas()
display(SOM)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### **Guinea**
# MAGIC   - Ebola Outbreak

# COMMAND ----------

# DBTITLE 1,Guinea
GIN = conflictEventsHorizonCountries.filter((F.col('ActionGeo_FullName') == 'Somalia'))
print('February Global CONFLICT Events in Guinea: ', GIN.count())
GIN.limit(5).toPandas()
display(GIN)

# COMMAND ----------



# COMMAND ----------

