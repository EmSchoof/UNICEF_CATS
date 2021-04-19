# Databricks notebook source
# DBTITLE 1,Import PySpark Modules
from functools import reduce
from itertools import chain
import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Import Data from Big Query
# import december
bq_gdelt_dec = spark.read.format("bigquery").option("table",'unicef-gdelt.december2020.events-mentions').load()
print('December 2020 Event-Mentions Data: ', (bq_gdelt_dec.count(), len(bq_gdelt_dec.columns)))

# import january
bq_gdelt_jan = spark.read.format("bigquery").option("table",'unicef-gdelt.january2021.events-mentions').load()
print('January 2021 Event-Mentions Data: ', (bq_gdelt_jan.count(), len(bq_gdelt_jan.columns)))

# import february
bq_gdelt_feb = spark.read.format("bigquery").option("table",'unicef-gdelt.february2021.events-mentions').load()
print('February 2021 Event-Mentions Data: ', (bq_gdelt_feb.count(), len(bq_gdelt_feb.columns)))

# import march
bq_gdelt_march = spark.read.format("bigquery").option("table",'unicef-gdelt.march2021.events-mentions').load()
print('March 2021 Event-Mentions Data: ', (bq_gdelt_march.count(), len(bq_gdelt_march.columns)))

# COMMAND ----------

# merge dataframes
bq_gdelt1 = bq_gdelt_dec.union(bq_gdelt_jan)
bq_gdelt2 = bq_gdelt1.union(bq_gdelt_feb)
bq_gdelt = bq_gdelt2.union(bq_gdelt_march)
print('December 2020 through March 2021 Event-Mentions Data: ', (bq_gdelt.count(), len(bq_gdelt.columns)))
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

# drop Nulls in Integer Columns
gdeltFebNoNulls1 = gdeltFeb.na.drop(subset=["GLOBALEVENTID","EventTimeDate","MentionTimeDate", "ActionGeo_CountryCode", "Confidence", "MentionDocTone", "EventRootCode", "QuadClass", "GoldsteinScale"])

# drop Nulls in string columns
gdeltFebNoNulls = gdeltFebNoNulls1.where(F.col('ActionGeo_CountryCode').isNotNull())

# verify output
print('Removal of Nulls Dataframe: ', (gdeltFebNoNulls.count(), len(gdeltFebNoNulls.columns)))
count_missings(gdeltFebNoNulls)

# COMMAND ----------

gdeltFebNoNulls.printSchema()
gdeltFebNoNulls.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
# Convert date-int columns to string columns
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('EventTimeDate', F.expr("CAST(EventTimeDate AS STRING)"))
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('MentionTimeDate', F.expr("CAST(MentionTimeDate AS STRING)"))

# Confirm output
gdeltFebNoNulls.limit(2).toPandas()

# COMMAND ----------

# Convert date-strings to date columns
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('EventTimeDate', F.unix_timestamp(F.col('EventTimeDate'), 'yyyyMMddHHmmss').cast("timestamp"))
gdeltFebNoNulls = gdeltFebNoNulls.withColumn('MentionTimeDate', F.unix_timestamp(F.col('MentionTimeDate'), 'yyyyMMddHHmmss').cast("timestamp"))

# Confirm output
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
gdeltFebNoNullsSelectDFIPS = gdeltFebNoNullsSelectDcon40.withColumn(
    'ActionGeo_FullName',
    F.when(F.col('ActionGeo_CountryCode') == 'YI', "Serbia and Montenegro")
    .when(F.col('ActionGeo_CountryCode') == 'PF', "Paracel Islands")
    .when(F.col('ActionGeo_CountryCode') == 'NT', "Netherlands Antilles")
    .when(F.col('ActionGeo_CountryCode') == 'PG', "Spratly Islands")
    .when(F.col('ActionGeo_CountryCode') == 'GZ', "Gaza Strip")
    .when(F.col('ActionGeo_CountryCode') == 'RB', "Serbia")
    .when(F.col('ActionGeo_CountryCode') == 'WQ', "Wake Island")
    .when(F.col('ActionGeo_CountryCode') == 'KV', "Kosovo")
    .when(F.col('ActionGeo_CountryCode') == 'DA', "Denmark")
    .when(F.col('ActionGeo_CountryCode') == 'UP', "Ukraine")
    .when(F.col('ActionGeo_CountryCode') == 'HQ', "Howland Island")
    .when(F.col('ActionGeo_CountryCode') == 'VM', "Vietnam")
    .when(F.col('ActionGeo_CountryCode') == 'JN', "Jan Mayen")
    .when(F.col('ActionGeo_CountryCode') == 'LQ', "Palmyra Atoll")
    .when(F.col('ActionGeo_CountryCode') == 'BQ', "Navassa Island")
    .when(F.col('ActionGeo_CountryCode') == 'JQ', "Johnston Atoll")
    .when(F.col('ActionGeo_CountryCode') == 'BS', "Bassas da India")
    .when(F.col('ActionGeo_CountryCode') == 'FQ', "Baker Island")
    .when(F.col('ActionGeo_CountryCode') == 'IP', "Clipperton Island")
    .otherwise(F.col('ActionGeo_FullName'))
)

# COMMAND ----------

# verify output
nullCountries = gdeltFebNoNullsSelectDFIPS.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).where(F.col('ActionGeo_FullName').isNull())
print(nullCountries.count())
nullCountries.show()

# COMMAND ----------

# DBTITLE 1,Assess Countries Associated with OC and OS Country Codes
unknownCountries = gdeltFebNoNullsSelectDFIPS.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_CountryCode')).where(F.col('ActionGeo_CountryCode').isin('OC','OS'))
unknownCountries.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC After looking at a couple of investigative queries in BigQuery of the GDELT Events data, it appears that 'OC' relates to larger oceans, (Atlantic, Artic, Pacific, and Indian), while 'OS' refers to Oceans (general). Since, without the imported Action Geo Country name to further specific which ocean is referenced by either FIPS code, the rows with FIPS codes lacking the lat/long for the Event will be dropped.

# COMMAND ----------

# DBTITLE 1,Date Removal (3): Add Ocean FIPS Code Strings and Drop Rows w/o Coordinates
gdeltFebNoNullsSelectDFIPSocean = gdeltFebNoNullsSelectDFIPS.withColumn(
    'ActionGeo_FullName',
    F.when(F.col('ActionGeo_CountryCode') == 'OC', "Oceans, (Atlantic, Artic, Pacific, or Indian)")
    .when(F.col('ActionGeo_CountryCode') == 'OS', "Oceans, (general)")  
    .otherwise(F.col('ActionGeo_FullName'))
)

# verify output
nullCountriesOceans = gdeltFebNoNullsSelectDFIPSocean.select('ActionGeo_CountryCode', 'ActionGeo_FullName').dropDuplicates().sort(F.col('ActionGeo_FullName')).where(F.col('ActionGeo_FullName').isNull())
print(nullCountriesOceans.count())
nullCountriesOceans.show()

# COMMAND ----------

# DBTITLE 1,Verify NoNulls in Target Variables
print('Original Dataframe: ', (gdeltFebNoNullsSelectDFIPSocean.count(), len(gdeltFebNoNullsSelectDFIPSocean.columns)))

# drop rows where FIPS codes are present without Event coordinates
gdeltPreprocessedData = gdeltFebNoNullsSelectDFIPSocean.na.drop(subset=['ActionGeo_Lat', 'ActionGeo_Long'])

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