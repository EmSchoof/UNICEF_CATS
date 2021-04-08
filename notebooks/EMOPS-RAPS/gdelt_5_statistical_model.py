# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### After careful evaluation and verification, it has been established that none of the target variables are normally distributed. Thus, using the standard deviation of these variables would be insufficient for establishing a viable alert system. 
# MAGIC 
# MAGIC *New Methodology*
# MAGIC - (1.0) Compute X day average for all the data values across all years (2019 to present, result will be X = [x1, x2, x3, ... xn]).
# MAGIC - (2.0) Set a threshhold parameter z. // benchmark special parameter (needs to be established per country therefore there is need to come up with a "Ground Truth" value)
# MAGIC - (3.0) For each value in X, compare with z. If X is greater than z, alert.
# MAGIC - (4.0) Verify z threshold with past (known) data.
# MAGIC 
# MAGIC *Execution of Methodology*
# MAGIC - 1 - [create a list of 3 day moving averages from today - 365 days] // compare this list with defined 'z' anomalist behavior to the current 3 day average per EventRootCode
# MAGIC - 2 - [create a list of 60 day moving averages from today - 730 days] // compare this list with defined 'z' anomalist behavior to the current 60 day average per EventRootCode