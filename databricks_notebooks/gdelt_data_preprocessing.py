# Databricks notebook source
# DBTITLE 0,Preliminary: Import Non-Standard Libraries in to Databricks
# For libraries that are not part of the standard distribution of Databricks, use the dbutils.library methods
#dbutils.library.installPyPI("pandas")
#dbutils.library.installPyPI("numpy")
#dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import PySpark Modules
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
               StructField('GLOBALEVENTID', IntegerType(), True),
               StructField('EventTimeDate', StringType(), True),
               StructField('MentionTimeDate', StringType(), True),
               StructField('Confidence', IntegerType(), True),
               StructField('MentionDocTone', FloatType(), True),
               StructField('EventCode', StringType(), True),
               StructField('EventRootCode', StringType(), True),
               StructField('QuadClass', IntegerType(), True),
               StructField('GoldsteinScale', FloatType(), True),
               StructField('ActionGeo_Type', StringType(), True),
               StructField('ActionGeo_FullName', StringType(), True),
               StructField('ActionGeo_CountryCode', StringType(), True),
               StructField('ActionGeo_Lat', FloatType(), True),
               StructField('ActionGeo_Long', FloatType(), True),
               StructField('SOURCEURL', StringType(), True),
            ]

final_struc = StructType(fields = data_schema)

# COMMAND ----------

gdeltFeb = sqlContext.createDataFrame(bq_gdelt.rdd, final_struc)
gdeltFeb.limit(10).toPandas()

# COMMAND ----------

# DBTITLE 1,Convert String Date Columns to Datetimes
# This function converts the string cell into a date:
func =  F.udf (lambda x: datetime.strptime(x, 'yyyyMMddHHmmss'), DateType())
gdeltFeb1 = gdeltFeb.withColumn('EventTimeDate', func(F.col('EventTimeDate')))
gdeltFebDates = gdeltFeb1.withColumn('MentionTimeDate', func(F.col('MentionTimeDate')))
gdeltFebDates.printSchema()

# COMMAND ----------

# DBTITLE 1,Assess GDELT Data Distributions
gdeltFeb.select("GLOBALEVENTID","EventTimeDate","MentionTimeDate", "Confidence", "MentionDocTone", "EventRootCode", "QuadClass", "GoldsteinScale").describe().show()

# COMMAND ----------

# DBTITLE 1,Assess Null Values
result2 = spark.sql(
  """
  SELECT 
    education,
    ROUND( AVG( if(LTRIM(marital_status) = 'Never-married',1,0) ), 2) as bachelor_rate
  FROM 
    adult 
  GROUP BY education
  ORDER BY bachelor_rate DESC
  """)

# register the df we just made as a table for spark sql named a table -> "result2" 
sqlContext.registerDataFrameAsTable(result2, "result2")
spark.sql("SELECT * FROM result2").show(1)





gdeltFebNulls = gdeltFeb.filter(gdeltFeb.GLOBALEVENTID.isNull() | gdeltFeb.EventTimeDate.isNull() | gdeltFeb.MentionTimeDate.isNull() | 
                                gdeltFeb.Confidence.isNull() | gdeltFeb.MentionDocTone.isNull() | gdeltFeb.EventRootCode.isNull() | 
                                gdeltFeb.QuadClass.isNull() | gdeltFeb.GoldsteinScale.isNull() )
gdeltFebNulls.describe()

# COMMAND ----------

# DBTITLE 1,Assess Null Values
gdeltFebNulls = gdeltFeb.na.drop(subset=["GLOBALEVENTID","EventTimeDate","MentionTimeDate", "Confidence", "MentionDocTone", "EventRootCode", "QuadClass", "GoldsteinScale"])

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
cameo_verbs = ['MAKE PUBLIC STATEMENT','APPEAL','EXPRESS INTENT TO COOPERATE','CONSULT',
              'ENGAGE IN DIPLOMATIC COOPERATION','ENGAGE IN MATERIAL COOPERATION','PROVIDE AID',
               'YIELD','INVESTIGATE','DEMAND','DISAPPROVE','REJECT','THREATEN','PROTEST',
               'EXHIBIT MILITARY POSTURE','REDUCE RELATIONS','COERCE','ASSAULT','FIGHT',
               'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE']
print(cameo_verbs)

# Add string column
cleaned_merged_df, cameo_code_df = get_code_strings(cleaned_merged_df, 'EventRootCode', cameo_verbs)
cameo_code_df

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