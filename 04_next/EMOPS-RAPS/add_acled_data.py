# Databricks notebook source
# DBTITLE 1,Import PySpark Modules
from functools import reduce
from itertools import chain
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession

# COMMAND ----------

# DBTITLE 1,Import Data from ACLED
# The import the ACLED excel file 
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# COMMAND ----------

# import Africa conflict data
acledAfrica = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/FileStore/tables/tmp/Africa_1997_2021_Apr23.csv")
print((acledAfrica.count(), len(acledAfrica.columns)))
acledAfrica.limit(3).toPandas()

# COMMAND ----------

# import Middle Easte conflict data
acledME = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/FileStore/tables/tmp/MiddleEast_2015_2021_Apr23.csv")
print((acledME.count(), len(acledME.columns)))
acledME.limit(3).toPandas()

# COMMAND ----------

# merge dataframes
acledSelect = acledAfrica.union(acledME)
print((acledSelect.count(), len(acledSelect.columns)))
acledSelect.limit(3).toPandas()

# COMMAND ----------

# Select specific columns
select_cols = ['COUNTRY','YEAR','EVENT_DATE','EVENT_TYPE','SUB_EVENT_TYPE', 'FATALITIES']
acledSelectCols = acledSelect.select(select_cols)
print((acledSelectCols.count(), len(acledSelectCols.columns)))
acledSelectCols.limit(3).toPandas()

# COMMAND ----------

data_schema = [
               StructField('COUNTRY', StringType(), True), # numerical
               StructField('YEAR', IntegerType(), True), # numerical
               StructField('EVENT_DATE', StringType(), True),
               StructField('EVENT_TYPE', StringType(), True), 
               StructField('SUB_EVENT_TYPE', StringType(), True),
               StructField('FATALITIES', IntegerType(), True),
            ]

final_struc = StructType(fields = data_schema)

# COMMAND ----------

# DBTITLE 1,Create DataFrame for Manipulation with Defined Schema
acledRDD = sqlContext.createDataFrame(acledSelectCols.rdd, final_struc)
acledRDD.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Event Types and Sub Event Types
eventTypes = acledRDD.select('EVENT_TYPE', 'FATALITIES').groupBy('EVENT_TYPE').sum('FATALITIES') #'SUB_EVENT_TYPE'
eventTypes.toPandas()

# COMMAND ----------

# DBTITLE 0,Assess Event Types and Sub Event Types
eventTypes = acledRDD.select('SUB_EVENT_TYPE', 'FATALITIES').groupBy('SUB_EVENT_TYPE').sum('FATALITIES') #'EVENT_TYPE'
eventTypes.toPandas()

# COMMAND ----------

# DBTITLE 1,Drop rows with None
acledRDD.describe().show()

# COMMAND ----------

acledRDDNoNulls = acledRDD.na.drop(subset=select_cols)
print((acledRDDNoNulls.count(), len(acledRDDNoNulls.columns)))
acledRDDNoNulls.limit(5).toPandas()

# COMMAND ----------

acledRDDNoNulls.describe().show()

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
from dateutil.parser import parse

# Convert date-strings to date columns
date_udf = F.udf(lambda d: parse(d), DateType())

acledRDDNoNulls = acledRDDNoNulls.withColumn('EVENT_DATE', date_udf('EVENT_DATE'))
acledRDDNoNulls.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Data from 2021
acled2021 = acledRDDNoNulls.filter(F.col('YEAR') >= 2021).filter(acledRDDNoNulls['EVENT_DATE'] >= F.lit('2021-03-01'))
acled2021.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess DataFrame Distributions in Target Variables
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
acledTargetOutput = acled2021.groupBy('COUNTRY','EVENT_DATE','EVENT_TYPE') \
                              .agg(F.sum('FATALITIES').alias('FATALITIES'))
acledTargetOutput.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Remaining Event Dates
datesDF = acledTargetOutput.select('EVENT_DATE')
min_date, max_date = datesDF.select(F.min('EVENT_DATE'),F.max('EVENT_DATE')).first()
min_date, max_date

# COMMAND ----------

acledPandas = acledTargetOutput.toPandas().groupby(['COUNTRY', 'EVENT_DATE'], as_index=False).agg({'FATALITIES' : 'sum'})
acledPandas.head()

# COMMAND ----------

import numpy as np
countries = np.unique(acledPandas['COUNTRY'])
countries

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(16, 24))
sns.boxplot(data=acledPandas,
            x="FATALITIES",
            y="COUNTRY",
            ax=ax)

# COMMAND ----------

for country in countries:
  fig, ax = plt.subplots(figsize=(10, 6))
  acledPandas.loc[ acledPandas['COUNTRY'] == country ].set_index('EVENT_DATE').plot(ax=ax)
  ax.set(title="Conflict Articles Over Time in " + country,
         xlabel="EVENT_DATE",
         ylabel="FATALITIES")
  ax.axhline(0, linestyle="dashed", color="black", alpha=0.5)
  plt.show()

# COMMAND ----------

