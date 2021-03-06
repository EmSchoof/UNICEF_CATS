""" This Script is designed to verify the connection between PyCharm Py dash_dashboard and Google BigQuery """

# set the OS environment to BigQuery
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../Desktop/UNICEF/unicef-gdelt-bq_conn.json"

# connect to big query client
from google.cloud import bigquery
client = bigquery.Client()

# define a test query
QUERY = ('SELECT * FROM `unicef-gdelt.february2021.events-mentions` LIMIT 1000000')

# initiate query request
query_job = client.query(QUERY)

# get result
query_result = query_job.result()

# save output of result to csv
df = query_result.to_dataframe().to_csv('data/bq_data_feb2021.csv.csv', encoding='utf-8')
#print(df)

# Confirmed: verified output