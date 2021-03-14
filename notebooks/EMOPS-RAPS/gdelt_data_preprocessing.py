# Databricks notebook source
# DBTITLE 1,Import PySpark Modules
from functools import reduce
from itertools import chain
import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Import Data from Big Query
bq_gdelt = spark.read.format("bigquery").option("table",'unicef-gdelt.february2021.events-mentions').load()
print((bq_gdelt.count(), len(bq_gdelt.columns)))
bq_gdelt.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Define Data Schema
data_schema = [
               StructField('GLOBALEVENTID', IntegerType(), True), # numerical
               StructField('EventTimeDate', LongType(), True), # numerical
               StructField('MentionTimeDate', LongType(), True), # numerical
               StructField('Confidence', IntegerType(), True),
               StructField('MentionDocTone', FloatType(), True),
               StructField('EventCode', StringType(), True),
               StructField('EventRootCode', StringType(), True),
               StructField('QuadClass', IntegerType(), True), # numerical
               StructField('GoldsteinScale', FloatType(), True), # numerical
               StructField('ActionGeo_Type', StringType(), True),
               StructField('ActionGeo_FullName', StringType(), True),
               StructField('ActionGeo_CountryCode', StringType(), True),
               StructField('ActionGeo_Lat', FloatType(), True), # numerical
               StructField('ActionGeo_Long', FloatType(), True), # numerical
               StructField('SOURCEURL', StringType(), True),
            ]

final_struc = StructType(fields = data_schema)

# COMMAND ----------

# DBTITLE 1,Create DataFrame for Manipulation with Defined Schema
gdeltFeb = sqlContext.createDataFrame(bq_gdelt.rdd, final_struc)
gdeltFeb.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess DataFrame Distributions in Target Variables
gdeltFeb.select("GLOBALEVENTID","EventTimeDate","MentionTimeDate", "Confidence", "MentionDocTone", "EventCode", "EventRootCode", "QuadClass", "GoldsteinScale").describe().show()

# COMMAND ----------

# DBTITLE 1,Replace Obvious, Irregular Values with Nulls
# Event Code
gdeltFeb = gdeltFeb.withColumn(
    'EventCode',
    F.when(
        F.col('EventCode').isin('---', '--'),
        None
    ).otherwise(F.col('EventCode')).cast('int')
)

# Event Root Code
gdeltFeb = gdeltFeb.withColumn(
    'EventRootCode',
    F.when(
        F.col('EventRootCode').isin('---', '--'),
        None
    ).otherwise(F.col('EventRootCode')).cast('int')
)

# Verify Output
print(gdeltFeb.dtypes)
gdeltFeb.select("EventCode", "EventRootCode").describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Based on project requirements, the data source for visualization presence of non-null values in the following columns:
# MAGIC   
# MAGIC - GlobalEventId
# MAGIC - EventTimeDate
# MAGIC - ActionGeo_CountryCode
# MAGIC - EventCode
# MAGIC - GoldsteinScale
# MAGIC - MentionDocTone

# COMMAND ----------

# DBTITLE 1,Assess Null Values
def count_missings(spark_df, sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(df) == 0:
        print("There are no any missing values!")
        return None

    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count",ascending=False)

    return df

# COMMAND ----------

# Non String Values
count_missings(gdeltFeb)

# COMMAND ----------

# DBTITLE 1,DATA REMOVAL (1):  Drop Rows with Nulls in Key Columns (add all in list as a precaution)
print('Original Dataframe: ', (gdeltFeb.count(), len(gdeltFeb.columns)))
gdeltFebNoNulls = gdeltFeb.na.drop(subset=["GLOBALEVENTID","EventTimeDate","MentionTimeDate", "Confidence", "MentionDocTone", "EventRootCode", "QuadClass", "GoldsteinScale"])

# Verify output
print('Removal of Nulls Dataframe: ', (gdeltFebNoNulls.count(), len(gdeltFebNoNulls.columns)))
count_missings(gdeltFebNoNulls)

# COMMAND ----------

gdeltFebNoNulls.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
# Convert date-int columns to string columns
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('EventTimeDate', F.expr("CAST(EventTimeDate AS STRING)"))
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('MentionTimeDate', F.expr("CAST(MentionTimeDate AS STRING)"))

# Convert date-strings to date columns
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('EventTimeDate', F.to_date(F.unix_timestamp(F.col('EventTimeDate'), 'yyyyMMddHHmmss').cast("timestamp")))
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('MentionTimeDate', F.to_date(F.unix_timestamp(F.col('MentionTimeDate'), 'yyyyMMddHHmmss').cast("timestamp")))

# Confirm output
gdeltFebNoNulls.printSchema()
gdeltFebNoNulls.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Calculate Days Between Mentions and Events Data
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('DaysBetween', F.datediff(F.col('MentionTimeDate'),F.col('EventTimeDate')).cast('int'))
gdeltFebNoNulls.printSchema()
gdeltFebNoNulls.limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,DATA REMOVAL (2): Select Mentions within first 15 Days of an Event
print('Removal of Nulls Dataframe: ', (gdeltFebNoNulls.count(), len(gdeltFebNoNulls.columns)))

# Select Data Based on DaysBetween Column
gdeltFebNoNullsSelectD = gdeltFebNoNulls.where(F.col('DaysBetween') <= 15)

# Confirm output
print('Mentions within 15days of Event Dataframe: ', (gdeltFebNoNullsSelectD.count(), len(gdeltFebNoNullsSelectD.columns)))

# COMMAND ----------

# DBTITLE 1,Create Cameo Code Root Integer Values with Associated String
# Create CAMEO verbs list
cameo_verbs = ['MAKE PUBLIC STATEMENT','APPEAL','EXPRESS INTENT TO COOPERATE','CONSULT',
              'ENGAGE IN DIPLOMATIC COOPERATION','ENGAGE IN MATERIAL COOPERATION','PROVIDE AID',
               'YIELD','INVESTIGATE','DEMAND','DISAPPROVE','REJECT','THREATEN','PROTEST',
               'EXHIBIT MILITARY POSTURE','REDUCE RELATIONS','COERCE','ASSAULT','FIGHT',
               'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE']
print(cameo_verbs)

# Create distinct list of CAMEO EventRootCodes
cameo_codes = gdeltFebNoNullsSelectD.select('EventRootCode').distinct().rdd.map(lambda r: r[0]).collect()
cameo_codes_ordered = sorted(cameo_codes)
print(cameo_codes_ordered)

# Create CAMEO EventRootCodes: CAMEO verbs dictionary
cameo_verbs_dict = dict(zip(cameo_codes_ordered, cameo_verbs))
cameo_verbs_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_verbs_dict.items())])
gdeltFebNoNullsSelectD = gdeltFebNoNullsSelectD.withColumn('EventRootCodeString', mapping_expr[F.col('EventRootCode')])

# Confirm accurate output
gdeltFebNoNullsSelectD.select('EventRootCode', 'EventRootCodeString').dropDuplicates().sort(F.col('EventRootCode')).show()
gdeltFebNoNullsSelectD.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Cameo QuadClass Integer Values with Associated String
# Create CAMEO QuadClass String List
cameo_quadclass = ['Verbal Cooperation','Material Cooperation','Verbal Conflict','Material Conflict']
print(cameo_quadclass)

# Create distinct list of CAMEO QuadClass codes
cameo_quadclass_codes = gdeltFebNoNullsSelectD.select('QuadClass').distinct().rdd.map(lambda r: r[0]).collect()
cameo_quadclass_codes_ordered = sorted(cameo_quadclass_codes)
print(cameo_quadclass_codes_ordered)

# Create CAMEO QuadClass: CAMEO QuadClass String dictionary
cameo_quadclass_dict = dict(zip(cameo_quadclass_codes_ordered, cameo_quadclass))
cameo_quadclass_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_quadclass_dict.items())])
gdeltFebNoNullsSelectD = gdeltFebNoNullsSelectD.withColumn('QuadClassString', mapping_expr[F.col('QuadClass')])

# Confirm accurate output
gdeltFebNoNullsSelectD.select('QuadClass', 'QuadClassString').dropDuplicates().sort(F.col('QuadClass')).show()
gdeltFebNoNullsSelectD.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Country Code Integer Values with Country Names
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files.  
country_codes_df = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/Filestore/tables/countries.csv")

# Replace \x81 with an empty string
#country_codes_df = country_codes_df.withColumn('name', F.regexp_replace('name', '\x81Aland Islands', 'Aland Islands'))  
country_codes_df.limit(5).toPandas()

# COMMAND ----------

# Create Country Name and Country FIPS 10-4 Code Lists
country_names = country_codes_df.select('Country name').rdd.flatMap(lambda x: x).collect()
country_fips104 = country_codes_df.select('FIPS 10-4').rdd.flatMap(lambda x: x).collect()

# Create 2-Digit Country Code: Country Name dictionary
country_fips104_dict = dict(zip(country_fips104, country_names))
country_fips104_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*country_fips104_dict.items())])
gdeltFebNoNullsSelectD = gdeltFebNoNullsSelectD.withColumn('ActionGeo_FullName', mapping_expr[F.col('ActionGeo_CountryCode')])

# Confirm accurate output
gdeltFebNoNullsSelectD.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).show()
gdeltFebNoNullsSelectD.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Import Python Plotting Modules
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# COMMAND ----------

# DBTITLE 1,Visualize Events by Country
# create a grouped dataframe
gdeltPandas_df = gdeltFebNoNullsSelectD.select('GLOBALEVENTID', 'ActionGeo_FullName').groupby('ActionGeo_FullName').agg(F.countDistinct('GLOBALEVENTID')).alias('numEvents').toPandas()
gdeltPandas_df.head()

# COMMAND ----------

gdeltPandas_df.dtypes

# COMMAND ----------

# display the plot
#plt.figure(figsize=(100,20))
#myplot = gdeltPandas_df.plot(kind='barh', y='count(GLOBALEVENTID)', x='ActionGeo_FullName')
#display(myplot.figure)
gdeltFebNoNullsSelectD.columns

# COMMAND ----------

# DBTITLE 1,Save Select Columns of DataFrame as CSV
# select specific columns
select_columns = ['GLOBALEVENTID',
                 'EventTimeDate',
                 'EventRootCodeString',
                 'QuadClassString',
                 'MentionTimeDate',
                 'Confidence',
                 'MentionDocTone',
                 'GoldsteinScale',
                 'ActionGeo_FullName',
                 'ActionGeo_Lat',
                 'ActionGeo_Long'
                 ]

# save as CSV
gdeltFebNoNullsSelectD.select(select_columns).write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/Filestore/tables/tmp/gdelt/preprocessed.csv')