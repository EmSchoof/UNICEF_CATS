# Databricks notebook source
# MAGIC %md
# MAGIC ### Create CSV for UNICEF Regions by FIPS 10-4 Code

# COMMAND ----------

# DBTITLE 1,Import Modules
from itertools import chain
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Import FIPS 10-4 Country Codes
# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files.  
country_codes_df = spark.read.format("csv") \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load("/FileStore/tables/tmp/gdelt/updated_fips104_codes.csv")

country_codes_df.limit(5).toPandas()

# COMMAND ----------

# DBTITLE 1,Import UNICEF Regions Data
# source: country column
unicef_countries = ["Afghanistan","Angola","Anguilla","Albania","United Arab Emirates","Argentina","Armenia","Antigua and Barbuda","Azerbaijan","Burundi","Benin","Burkina Faso","Bangladesh","Bulgaria","Bahrain","Bosnia and Herzegovina","Belarus","Belize","Plurinational State of Bolivia","Brazil","Barbados","Bhutan","Botswana","Central African Republic","Chile","China","Cote d’Ivoire","Cameroon","DRC","ROC","Colombia","Comoros","Cabo Verde","Costa Rica","Cuba","Djibouti","Dominica","Dominican Republic","Algeria","Ecuador","Egypt","Eritrea","Western Sahara","Ethiopia","Fiji","Federated States of Micronesia","Gabon","Georgia","Ghana","Guinea","Gambia","Guinea-Bissau","Equatorial Guinea","Grenada","Guatemala","Guyana","Honduras","Croatia","Haiti","Indonesia","India","Islamic Republic of Iran","Iraq","Jamaica","Jordan","Kazakhstan","Kenya","Kyrgyzstan","Cambodia","Kiribati","Saint Kitts and Nevis","Kuwait","Lao People’s Democratic Republic","Lebanon","Liberia","Libya","Saint Lucia","Sri Lanka","Lesotho","Morocco","Moldova","Madagascar","Maldives","Mexico","Marshall Islands","Macedonia (former Yugoslav)","Mali","Myanmar","Montenegro","Mongolia","Mozambique","Mauritania","Montserrat","Malawi","Malaysia","Namibia","Niger","Nigeria","Nicaragua","Nepal ","Nauru","Oman","Pakistan","Panama","Peru","Philippines","Palau","Papua New Guinea","North Korea","Paraguay","Palestine (West Bank)","Qatar","Kosovo","Romania","Rwanda","Saudi Arabia","Sudan","Senegal","Solomon Islands","Sierra Leone","El Salvador","Somalia","Serbia","South Sudan","Sao Tome and Principe","Suriname","Eswatini","Syrian","Turks and Caicos Islands","Chad","Togo","Thailand","Tajikistan","Tokelau","Turkmenistan","Timor-Leste","Tonga","Trinidad and Tobago", "Tunisia","Turkey","Tuvalu",
"Tanzania","Uganda","Ukraine","Uruguay","Uzbekistan","Saint Vincent and the Grenadines","Venezuela","British Virgin Islands","Vietnam","Vanuatu","Samoa","Yemen","South Africa","Zambia","Zimbabwe"]

# source: unicef region column
unicef_region_ordered = ["ROSA", "ESARO", "LACRO", "ECARO", "MENARO", "LACRO", "ECARO", "LACRO", "ECARO", "ESARO", "WCARO", "WCARO", "ROSA", "ECARO", "MENARO", "WCARO", "ECARO", "LACRO", "LACRO", "LACRO", "LACRO", "ROSA", "ESARO", "WCARO", "LACRO", "EAPRO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "ESARO", "WCARO", "LACRO", "LACRO", "MENARO", "LACRO", "LACRO", "MENARO", "LACRO", "MENARO", "ESARO", "MENARO", "ESARO", "EAPRO", "EAPRO", "WCARO", "ECARO", "WCARO", "WCARO", "WCARO", "WCARO", "WCARO", "LACRO", "LACRO", "LACRO", "LACRO", "ECARO", "LACRO", "EAPRO", "ROSA", "MENARO", "MENARO", "LACRO", "MENARO", "ECARO", "ESARO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "EAPRO", "MENARO", "WCARO", "MENARO", "LACRO", "ROSA", "ESARO", "MENARO", "ECARO", "ESARO", "ROSA", "LACRO", "EAPRO", "ECARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ESARO", "WCARO", "LACRO", "ESARO", "EAPRO", "ESARO", "WCARO", "WCARO", "LACRO", "ROSA", "EAPRO", "MENARO", "ROSA", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "EAPRO", "LACRO", "MENARO", "MENARO", "ECARO", "ECARO", "ESARO", "MENARO", "MENARO", "WCARO", "EAPRO", "WCARO", "LACRO", "ESARO", "ECARO", "ESARO", "WCARO", "LACRO", "ESARO", "MENARO", "LACRO", "WCARO", "WCARO", "EAPRO", "ECARO", "EAPRO", "ECARO", "EAPRO", "EAPRO", "LACRO", "MENARO", "ECARO", "EAPRO", "ESARO", "ESARO", "ECARO", "LACRO", "ECARO", "LACRO", "LACRO", "LACRO", "EAPRO", "EAPRO", "EAPRO", "MENARO", "ESARO", "ESARO", "ESARO"]

# Create Country: Country Region dictionary
country_cluster_dict = dict(zip(unicef_countries, unicef_region_ordered))

# COMMAND ----------

# DBTITLE 1,Map and Merge
# Map dictionary over df to create string column
mapping_expr = F.create_map([F.lit(x) for x in chain(*country_cluster_dict.items())])
country_codes_df = country_codes_df.withColumn('UNICEF_regions', mapping_expr[F.col('Country name')])
country_codes_df.toPandas()

# COMMAND ----------

# DBTITLE 1,Prepare for Output
country_codes_nonulls_df = country_codes_df.na.drop(subset='UNICEF_regions')
country_codes_nonulls_df.toPandas()

# COMMAND ----------

# DBTITLE 1,Add Missing Countries
# Democratic Republic of the Congo
newRow1 = spark.createDataFrame([('Democratic Republic of the Congo','CG','DR','WCARO')])
country_codes_final_df = country_codes_nonulls_df.union(newRow1)

# Republic of the Congo
newRow2 = spark.createDataFrame([('Republic of the Congo','CF','CD','WCARO')])
country_codes_final_df = country_codes_final_df.union(newRow2)
country_codes_final_df.toPandas()

# COMMAND ----------

# DBTITLE 1,Store
import os

TEMPORARY_TARGET="dbfs:/Filestore/tables/tmp/gdelt/countries_fips_regions"
DESIRED_TARGET="dbfs:/Filestore/tables/tmp/gdelt/countries_fips_regions.csv"

country_codes_final_df.coalesce(1).write.option("header", "true").mode('overwrite').csv(TEMPORARY_TARGET)
temporary_csv = os.path.join(TEMPORARY_TARGET, dbutils.fs.ls(TEMPORARY_TARGET)[3][1])
dbutils.fs.cp(temporary_csv, DESIRED_TARGET)