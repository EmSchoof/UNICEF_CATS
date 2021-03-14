# Databricks notebook source
# DBTITLE 1,Import PySpark Modules
from functools import reduce
from itertools import chain
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

preprocessedGDELT.columns

# COMMAND ----------

# DBTITLE 1,Create Columns for Grouped String Columns with Window Functions
# EventRootCodeString
groupedERCS = preprocessedGDELT.select('GLOBALEVENTID', 'EventRootCodeString').groupBy('GLOBALEVENTID', 'EventRootCodeString').count()
windowERCS = Window.partitionBy('EventRootCodeString').orderBy(F.desc('count'))
groupedERCS = groupedERCS \
    .withColumn('order', F.row_number().over(windowERCS))\
    .where(F.col('order') == 1)

groupedERCS.limit(10).toPandas()

# COMMAND ----------

# QuadClassString
groupedQCS = preprocessedGDELT.select('GLOBALEVENTID', 'QuadClassString').groupBy('GLOBALEVENTID', 'QuadClassString').count()
windowQCS = Window.partitionBy('QuadClassString').orderBy(F.desc('count'))
groupedQCS = groupedQCS \
    .withColumn('order', F.row_number().over(windowQCS))\
    .where(F.col('order') == 1)

groupedQCS.limit(10).toPandas()

# COMMAND ----------

# ActionGeo_FullName
groupedAGFN = preprocessedGDELT.select('GLOBALEVENTID', 'ActionGeo_FullName').groupBy('GLOBALEVENTID', 'ActionGeo_FullName').count()
windowAGFN = Window.partitionBy('ActionGeo_FullName').orderBy(F.desc('count'))
groupedAGFN = groupedAGFN \
    .withColumn('order', F.row_number().over(windowAGFN))\
    .where(F.col('order') == 1)

groupedAGFN.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Target Variable Column
df.groupBy('x').count().select('x', f.col('count').alias('n')).show()


nEventsGDELT = preprocessedGDELT /
               .groupBy('GLOBALEVENTID') /
               .agg(
                 F.mode()
                 F.avg()
               )
preprocessedGDELT.limit(5).toPandas()