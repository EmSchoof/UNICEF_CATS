# Databricks notebook source
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

preprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Count Unique Global Event Ids
print((preprocessedGDELT.count(), len(preprocessedGDELT.columns)))
preprocessedGDELT.agg(F.countDistinct(F.col("GLOBALEVENTID")).alias("nEvents")).show()