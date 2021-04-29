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
select_cols = ['COUNTRY','EVENT_DATE','EVENT_TYPE','SUB_EVENT_TYPE']
acledAfricaSelect = acledAfrica.select(select_cols)
acledAfricaSelect = acledAfricaSelect.withColumn('nArticles', F.lit(1))
print((acledAfricaSelect.count(), len(acledAfricaSelect.columns)))
acledAfricaSelect.limit(5).toPandas()

# COMMAND ----------

data_schema = [
               StructField('COUNTRY', StringType(), True), # numerical
               StructField('EVENT_DATE', StringType(), True), # numerical
               StructField('EVENT_TYPE', StringType(), True), # numerical
               StructField('SUB_EVENT_TYPE', StringType(), True),
               StructField('nArticles', IntegerType(), True),
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

acledAfricaRDD.describe().show()

# COMMAND ----------

# DBTITLE 1,Assess DataFrame Distributions in Target Variables
# Create New Dataframe Column to Count Number of Daily Articles by Country by EventRootCode
acledTargetOutput = acledAfricaRDD.groupBy('COUNTRY','EVENT_DATE','EVENT_TYPE') \
                                  .agg(F.sum('nArticles').alias('nArticles'))
acledTargetOutput.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,DATA REMOVAL (1):  Drop Rows with Nulls in Key Columns (add all in list as a precaution)
print('Original Dataframe: ', (gdeltFeb.count(), len(gdeltFeb.columns)))

# drop Nulls in Integer Columns
gdeltFebNoNulls1 = gdeltFeb.na.drop(subset=["GLOBALEVENTID","EventTimeDate", "ActionGeo_CountryCode", "Confidence", "MentionDocTone", "EventRootCode","GoldsteinScale"])

# drop Nulls in string columns
gdeltFebNoNulls = gdeltFebNoNulls1.where(F.col('ActionGeo_CountryCode').isNotNull())

# verify output
print('Removal of Nulls Dataframe: ', (gdeltFebNoNulls.count(), len(gdeltFebNoNulls.columns)))
count_missings(gdeltFebNoNulls)

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
from dateutil.parser import parse

# Convert date-strings to date columns
date_udf = F.udf(lambda d: parse(d), DateType())

df = acledTargetOutput.withColumn('EVENT_DATE', date_udf('EVENT_DATE'))


# Confirm output
df.show()#limit(2).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Remaining Event Dates
datesDF = gdeltFebNoNullsSelectD.select('EventTimeDate')
min_date, max_date = datesDF.select(F.min('EventTimeDate'),F.max('EventTimeDate')).first()
min_date, max_date

# COMMAND ----------

# DBTITLE 1,Select Data with Confidence of 40% or higher
# create confidence column of more than 
print(gdeltFebNoNullsSelectD.count())
gdeltFebNoNullsSelectDcon40 = gdeltFebNoNullsSelectD.filter(F.col('Confidence') >= 40)
print(gdeltFebNoNullsSelectDcon40.count())

# convert datetime column to dates
gdeltFebNoNullsSelectDcon40 = gdeltFebNoNullsSelectDcon40.withColumn('EventTimeDate', F.col('EventTimeDate').cast('date'))
gdeltFebNoNullsSelectDcon40.limit(2).toPandas()

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
cameo_codes = gdeltFebNoNullsSelectDcon40.select('EventRootCode').distinct().rdd.map(lambda r: r[0]).collect()
cameo_codes_ordered = sorted(cameo_codes)
print(cameo_codes_ordered)

# Create CAMEO EventRootCodes: CAMEO verbs dictionary
cameo_verbs_dict = dict(zip(cameo_codes_ordered, cameo_verbs))
cameo_verbs_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_verbs_dict.items())])
gdeltFebNoNullsSelectDcon40 = gdeltFebNoNullsSelectDcon40.withColumn('EventRootCodeString', mapping_expr[F.col('EventRootCode')])

# Confirm accurate output
gdeltFebNoNullsSelectDcon40.select('EventRootCode', 'EventRootCodeString').dropDuplicates().sort(F.col('EventRootCode')).show()
gdeltFebNoNullsSelectDcon40.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Create Cameo QuadClass Integer Values with Associated String
# Create CAMEO QuadClass String List
cameo_quadclass = ['Verbal Cooperation','Material Cooperation','Verbal Conflict','Material Conflict']
print(cameo_quadclass)

# Create distinct list of CAMEO QuadClass codes
cameo_quadclass_codes = gdeltFebNoNullsSelectDcon40.select('QuadClass').distinct().rdd.map(lambda r: r[0]).collect()
cameo_quadclass_codes_ordered = sorted(cameo_quadclass_codes)
print(cameo_quadclass_codes_ordered)

# Create CAMEO QuadClass: CAMEO QuadClass String dictionary
cameo_quadclass_dict = dict(zip(cameo_quadclass_codes_ordered, cameo_quadclass))
cameo_quadclass_dict

# COMMAND ----------

# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*cameo_quadclass_dict.items())])
gdeltFebNoNullsSelectDcon40 = gdeltFebNoNullsSelectDcon40.withColumn('QuadClassString', mapping_expr[F.col('QuadClass')])

# Confirm accurate output
gdeltFebNoNullsSelectDcon40.select('QuadClass', 'QuadClassString').dropDuplicates().sort(F.col('QuadClass')).show()
gdeltFebNoNullsSelectDcon40.limit(1).toPandas()

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
  .load("/FileStore/tables/tmp/gdelt/countries.csv")

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
gdeltFebNoNullsSelectDcon40 = gdeltFebNoNullsSelectDcon40.withColumn('ActionGeo_FullName', mapping_expr[F.col('ActionGeo_CountryCode')])

# Confirm accurate output
print(gdeltFebNoNullsSelectDcon40.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).count())
gdeltFebNoNullsSelectDcon40.limit(1).toPandas()

# COMMAND ----------

# DBTITLE 1,Assess Remaining Null Values for Country Name
# Assess Nulls in Country Name Strings
nullCountries = gdeltFebNoNullsSelectDcon40.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).where(F.col('ActionGeo_FullName').isNull())
print(nullCountries.count())
nullCountries.show()

# COMMAND ----------

# MAGIC %md
# MAGIC With the incidence of the country code, 'YI', it appears that GDELT uses the *depreciated* version of FIPS 10-4
# MAGIC 
# MAGIC "YI - Serbia and Montenegro (deprecated FIPS 10-4 country code, now RB (Serbia) and MJ (Montenegro))"

# COMMAND ----------

# DBTITLE 1,Replace Missing FIPS 10-4 Country Code Names
#gdeltFebNoNullsSelectDFIPS = gdeltFebNoNullsSelectDcon40.withColumn(
   # 'ActionGeo_FullName',
   # F.when(F.col('ActionGeo_CountryCode') == 'YI', "Serbia and Montenegro")
   # .when(F.col('ActionGeo_CountryCode') == 'IP', "Clipperton Island")
   # .otherwise(F.col('ActionGeo_FullName'))
#)

# COMMAND ----------

# verify output
#nullCountries = gdeltFebNoNullsSelectDFIPS.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).where(F.col('ActionGeo_FullName').isNull())
#print(nullCountries.count())
#nullCountries.show()

# COMMAND ----------

# DBTITLE 1,Assess Countries Associated with OC and OS Country Codes
#unknownCountries = gdeltFebNoNullsSelectDFIPS.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_CountryCode')).where(F.col('ActionGeo_CountryCode').isin('OC','OS'))
#unknownCountries.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC After looking at a couple of investigative queries in BigQuery of the GDELT Events data, it appears that 'OC' relates to larger oceans, (Atlantic, Artic, Pacific, and Indian), while 'OS' refers to Oceans (general). Since, without the imported Action Geo Country name to further specific which ocean is referenced by either FIPS code, the rows with FIPS codes lacking the lat/long for the Event will be dropped.

# COMMAND ----------

# DBTITLE 1,Date Removal (3): Add Ocean FIPS Code Strings and Drop Rows w/o Coordinates
#gdeltFebNoNullsSelectDFIPSocean = gdeltFebNoNullsSelectDFIPS.withColumn(
  #  'ActionGeo_FullName',
  #  F.when(F.col('ActionGeo_CountryCode') == 'OC', "Oceans, (Atlantic, Artic, Pacific, or Indian)")
  #  .when(F.col('ActionGeo_CountryCode') == 'OS', "Oceans, (general)")  
  # .otherwise(F.col('ActionGeo_FullName'))
#)

# verify output
#nullCountriesOceans = gdeltFebNoNullsSelectDFIPSocean.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).where(F.col('ActionGeo_FullName').isNull())
#print(nullCountriesOceans.count())
#nullCountriesOceans.show()

# COMMAND ----------

# DBTITLE 1,Verify NoNulls in Target Variables
#print('Original Dataframe: ', (gdeltFebNoNullsSelectDFIPSocean.count(), len(gdeltFebNoNullsSelectDFIPSocean.columns)))

# drop rows where FIPS codes are present without Event coordinates
gdeltPreprocessedData = gdeltFebNoNullsSelectDcon40.na.drop(subset=['ActionGeo_Lat', 'ActionGeo_Long'])

# verify output
print('Removal of Nulls Dataframe: ', (gdeltPreprocessedData.count(), len(gdeltPreprocessedData.columns)))
count_missings(gdeltPreprocessedData)

# COMMAND ----------

# DBTITLE 1,Assess Remaining Event Dates
datesDF = gdeltPreprocessedData.select('EventTimeDate')
min_date, max_date = datesDF.select(F.min('EventTimeDate'),F.max('EventTimeDate')).first()
min_date, max_date

# COMMAND ----------

# DBTITLE 1,Count Unique Global Events
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


outputPreprocessedGDELT = gdeltPreprocessedData.select(select_columns)
print((outputPreprocessedGDELT.count(), len(outputPreprocessedGDELT.columns)))
outputPreprocessedGDELT = outputPreprocessedGDELT.withColumn("nArticles", F.lit(1))
outputPreprocessedGDELT.agg(F.countDistinct(F.col("GLOBALEVENTID")).alias("nEvents")).show()

# COMMAND ----------

print(outputPreprocessedGDELT.columns)
outputPreprocessedGDELT.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Save DataFrame as CSV
outputPreprocessedGDELT.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('/FileStore/tables/tmp/gdelt/preprocessed.csv')