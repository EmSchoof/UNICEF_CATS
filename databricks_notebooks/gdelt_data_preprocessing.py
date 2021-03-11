# Databricks notebook source
# DBTITLE 1,Import PySpark Modules
from functools import reduce
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

# DBTITLE 1,Drop Rows with Nulls in Key Columns (add all in list as a precaution)
gdeltFebNoNulls = gdeltFeb.na.drop(subset=["GLOBALEVENTID","EventTimeDate","MentionTimeDate", "Confidence", "MentionDocTone", "EventRootCode", "QuadClass", "GoldsteinScale"])

# Verify output
count_missings(gdeltFebNoNulls)

# COMMAND ----------

# DBTITLE 1,Convert Integer Date Columns to Strings then to DateTimes
# Create function to convert string cells to datetimes
date_func =  F.udf (lambda x: datetime.strptime(x, 'yyyyMMddHHmmss'), DateType())

# Apply to date columns
gdeltFebDates = gdeltFeb.withColumn('EventTimeDate', expr("CAST(EventTimeDate AS INTEGER)"))
gdeltFebDates = gdeltFebDates.withColumn('EventTimeDate', date_func(F.col('EventTimeDate')))
gdeltFebDates = gdeltFeb.withColumn('EventTimeDate', expr("CAST(EventTimeDate AS INTEGER)"))
gdeltFebDates = gdeltFebDates.withColumn('MentionTimeDate', date_func(F.col('MentionTimeDate')))
gdeltFebDates.printSchema()

# COMMAND ----------

# DBTITLE 1,Calculate Days Between Mentions and Events Data
def get_diff(x, y):
    result = F.datediff(x,y)
    return result

gdeltFebDates = gdeltFebDates.withColumn('DaysBetween',get_diff('MentionTimeDate','EventTimeDate')).show(2)
gdeltFebDates.printSchema()

# COMMAND ----------

# DBTITLE 1,Convert 2 Lists to Dictionary
from itertools import chain
dict_func = F.udf (lambda keys, vals: dict(zip(keys, vals)))

mapping_expr = F.udf (lambda mapping: F.create_map([F.lit(x) for x in F.chain(*mapping.items())]))



def get_dictionary(l1, l2):
    """
    Create Column for Strings Associated with Code Column
    
    :param df: dataframe of cleaned data
    :param codes_col: name of the code column in dataframe
    :return: dict
    """
    
    assert len(df[codes_col].unique()) == len(strings), "Length of codes and strings list are not equal"
    
    # Convert lists to dictionary 
    codes = df[codes_col].sort_values(ascending=True).unique()
    code_dict = {codes[i]: strings[i] for i in range(len(codes))}
    
    # Add column for code strings
    codes_string_col = codes_col+'String'
    df[codes_string_col] = df[codes_col].map(code_dict)

    # verify output
    verify_df = df[[codes_col, codes_string_col]].sort_values(by=codes_col, ascending=True).drop_duplicates()

    # return df and verified output
    return df, verify_df

# COMMAND ----------

# DBTITLE 1,Define Reusable Python Functions
def get_code_strings(df, codes_col: str, strings: list):
    """
    Create Column for Strings Associated with Code Column
    
    :param df: dataframe of cleaned data
    :param codes_col: name of the code column in dataframe
    :param strings: list of strings associated with code column
    :rtype: dataframes
    :return: 
        :df: param dataframe with new string column
        :verified_df: dataframe to confirm correct code/string column creation
    """
    
    assert len(df[codes_col].unique()) == len(strings), "Length of codes and strings list are not equal"
    
    # Convert lists to dictionary 
    codes = df[codes_col].sort_values(ascending=True).unique()
    code_dict = {codes[i]: strings[i] for i in range(len(codes))}
    
    # Add column for code strings
    codes_string_col = codes_col+'String'
    df[codes_string_col] = df[codes_col].map(code_dict)

    # verify output
    verify_df = df[[codes_col, codes_string_col]].sort_values(by=codes_col, ascending=True).drop_duplicates()

    # return df and verified output
    return df, verify_df

# COMMAND ----------

import pandas as pd
df = spark.createDataFrame(pd.DataFrame({'integers': [1,2,3,4,5]}))
strings = [1,2,3,4,5]
F.length(df.select(F.countDistinct('integers'))) == F.lenght(strings)

# COMMAND ----------

# DBTITLE 1,Select Mentions within first 60 Days of an Event
# Calculate days between
#bq_gdeltFebDaysBetween = bq_gdelt.withColumn('DaysBetween', bq_gdelt.select(F.col("MentionTimeDate")-F.col("EventTimeDate")))
bq_gdelt.select(F.col("MentionTimeDate")-F.col("EventTimeDate"))

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

# DBTITLE 0,Select Data Associated with Target Country Code List
# Drop all rows in merged_df with nulls in the specified columns
required_value_columns = ['GLOBALEVENTID', 'EventTimeDate', 'ActionGeo_CountryCode', 
                          'EventCode', 'GoldsteinScale', 'MentionDocTone']

cleaned_merged_df = merged_df[~pd.isnull(merged_df[required_value_columns]).any(axis=1)].reset_index(drop=True)
print(cleaned_merged_df.shape)
cleaned_merged_df.head(1)

# COMMAND ----------

# DBTITLE 1,Create Cameo Code Root Integer Values with Associated String
# Convert column to integer
new_column_udf = udf(lambda name: None if name == 0 else name, StringType())
gdeltFeb = gdeltFeb.withColumn("EventRootCode", new_column_udf(gdeltFeb.EventRootCode))
cameo_codes = gdeltFeb.select('EventRootCode').distinct().rdd.map(lambda r: r[0]).collect()
print(cameo_codes)

# COMMAND ----------

# DBTITLE 1,Create Cameo Code Root Integer Values with Associated String
cameo_verbs = ['MAKE PUBLIC STATEMENT','APPEAL','EXPRESS INTENT TO COOPERATE','CONSULT',
              'ENGAGE IN DIPLOMATIC COOPERATION','ENGAGE IN MATERIAL COOPERATION','PROVIDE AID',
               'YIELD','INVESTIGATE','DEMAND','DISAPPROVE','REJECT','THREATEN','PROTEST',
               'EXHIBIT MILITARY POSTURE','REDUCE RELATIONS','COERCE','ASSAULT','FIGHT',
               'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE']
print(cameo_verbs)
cameo_codes = gdeltFeb.select('EventRootCode').distinct().rdd.map(lambda r: r[0]).collect()
print(cameo_codes)

# COMMAND ----------

# verify output
cleaned_merged_df.head(2)

# COMMAND ----------

# DBTITLE 1,Create Cameo QuadClass Integer Values with Associated String
cameo_quadclass = ['Verbal Cooperation','Material Cooperation','Verbal Conflict','Material Conflict']
print(cameo_quadclass)

# get string column
cleaned_merged_df, cameo_quadclass_df = get_code_strings(cleaned_merged_df, 'QuadClass', cameo_quadclass)
cameo_quadclass_df

# COMMAND ----------

# DBTITLE 0,Start Visualizing Data
# verify output
cleaned_merged_df.head(2)

# COMMAND ----------

# DBTITLE 1,Create Country Code Integer Values with Country Names
# get country data
country_codes_df = pd.DataFrame(pd.read_csv('../select_data/data/countries.csv', encoding= 'unicode_escape'))

# convert lists to dictionary 
country_code_dict = {country_codes_df['alpha-2'][i]: country_codes_df['name'][i] for i in range(len(country_codes_df))}

# Add column for cameo code root strings (verbs)
cleaned_merged_df['ActionGeo_FullName'] = cleaned_merged_df['ActionGeo_CountryCode'].map(country_code_dict)
cleaned_merged_df.head()

# COMMAND ----------

# DBTITLE 0,Plot Incidences by Country
# verify output
cameo_country_df = cleaned_merged_df[['ActionGeo_CountryCode', 'ActionGeo_FullName']].sort_values(by='ActionGeo_CountryCode',
                                                                                ascending=True).drop_duplicates()
cameo_country_df.head(50)

# COMMAND ----------

# DBTITLE 1,Verify Cleaned-Data Output
# Select Desired Columns for Data Factory Output
cleaned_merged_df = cleaned_merged_df[desired_columns]
print('Cleaned Data with Desired Columns: ',cleaned_merged_df.shape)
cleaned_merged_df.head()