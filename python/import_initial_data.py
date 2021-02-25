""" This Script is designed to verify the connection between PyCharm and Google BigQuery """

# set the OS environment to BigQuery
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../Desktop/UNICEF/owner-bq_conn.json"

# connect to big query client
from google.cloud import bigquery

client = bigquery.Client()

# define a test query
QUERY = ('SELECT * FROM `testing-queries-feb2021.initial_data_pull.last-two-years` LIMIT 1000')

# initiate query request
query_job = client.query(QUERY)

# get result
query_result = query_job.result()

# test output of result
df = query_result.to_dataframe()
print(df)

# Confirmed: verified output
