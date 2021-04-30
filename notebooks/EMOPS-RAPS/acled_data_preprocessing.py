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

acledAfrica = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/FileStore/tables/tmp/Africa_1997_2021_Apr23.csv")
print((acledAfrica.count(), len(acledAfrica.columns)))
acledAfrica.limit(5).toPandas()

# COMMAND ----------

# Select specific columns
select_cols = ['COUNTRY','YEAR','EVENT_DATE','EVENT_TYPE','SUB_EVENT_TYPE']
acledAfricaSelect = acledAfrica.select(select_cols)
acledAfricaSelect = acledAfricaSelect.withColumn('nEvent', F.lit(1))
print((acledAfricaSelect.count(), len(acledAfricaSelect.columns)))
acledAfricaSelect.limit(5).toPandas()

# COMMAND ----------

data_schema = [
               StructField('COUNTRY', StringType(), True), # numerical
               StructField('YEAR', IntegerType(), True), # numerical
               StructField('EVENT_DATE', StringType(), True),
               StructField('EVENT_TYPE', StringType(), True), 
               StructField('SUB_EVENT_TYPE', StringType(), True),
               StructField('nEvent', IntegerType(), True),
            ]

final_struc = StructType(fields = data_schema)

# COMMAND ----------

# DBTITLE 1,Create DataFrame for Manipulation with Defined Schema
acledAfricaRDD = sqlContext.createDataFrame(acledAfricaSelect.rdd, final_struc)
acledAfricaRDD.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Event Types and Sub Event Types
eventTypes = acledAfricaRDD.select('EVENT_TYPE').groupBy('EVENT_TYPE').count() #'SUB_EVENT_TYPE'
eventTypes.toPandas()

# COMMAND ----------

# DBTITLE 0,Assess Event Types and Sub Event Types
eventTypes = acledAfricaRDD.select('SUB_EVENT_TYPE').groupBy('SUB_EVENT_TYPE').count() #'EVENT_TYPE'
eventTypes.toPandas()

# COMMAND ----------

# DBTITLE 1,Drop rows with None
acledAfricaRDD.describe().show()

# COMMAND ----------

acledAfricaRDDNoNulls = acledAfricaRDD.na.drop(subset=select_cols)
print((acledAfricaRDDNoNulls.count(), len(acledAfricaRDDNoNulls.columns)))
acledAfricaRDDNoNulls.limit(5).toPandas()

# COMMAND ----------

acledAfricaRDDNoNulls.describe().show()

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
from dateutil.parser import parse

# Convert date-strings to date columns
date_udf = F.udf(lambda d: parse(d), DateType())

acledAfricaRDDNoNulls = acledAfricaRDDNoNulls.withColumn('EVENT_DATE', date_udf('EVENT_DATE'))
acledAfricaRDDNoNulls.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Select Data from 2021
acled2021 = acledAfricaRDDNoNulls.filter(F.col('YEAR') >= 2021)
acled2021.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess DataFrame Distributions in Target Variables
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
acledTargetOutput = acled2021.groupBy('COUNTRY','EVENT_DATE','EVENT_TYPE') \
                              .agg(F.sum('nEvent').alias('nArticles'))
acledTargetOutput.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Remaining Event Dates
datesDF = acledTargetOutput.select('EVENT_DATE')
min_date, max_date = datesDF.select(F.min('EVENT_DATE'),F.max('EVENT_DATE')).first()
min_date, max_date

# COMMAND ----------

acledPandas = acledTargetOutput.toPandas()

# COMMAND ----------

import numpy as np
np.unique(acledPandas['COUNTRY'])

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(16, 24))

sns.boxplot(data=acledPandas,
            x="nArticles",
            y="COUNTRY",
            ax=ax)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Save DataFrame as CSV
outputPreprocessedGDELT.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/FileStore/tables/tmp/gdelt/preprocessed.csv')