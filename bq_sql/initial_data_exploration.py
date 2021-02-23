# Initial Date: 22 February 2021

# import global variables
from pilot_variables import PILOT_EVENTTYPES, PILOT_IOS2

# SQL Purpose: To explore GDELT Events and Mentions Data
# Specifications:
# (1) Event Took Place in One of 60 Pilot Countries
# - ActionGeo_CountryCode IN PILOT_IOS2
# (2) Event was a Material or Verbal Conflict
# - QuadClass IN (3,4)
# (3) Event Occurred within the in Last 2 years
# - Year >= EXTRACT(YEAR FROM CURRENT_TIMESTAMP()) - 2
# (4) Mention Confidence was 50% or Higher
# - Confidence >= 50
# (5)

# write BQ SQL
PILOT_INITIAL = """SELECT  e.country,
                           m.global_event_id,
                           e.cameo_class,
                           e.cameo_code,
                           e.cameo_base_code,
                           e.cameo_root_code,
                           e.event_avg_tone,
                           e.goldstein,
                           m.global_event_id,
                           m.event_time_15min,
                           m.mention_time_15min,
                           m.gdelt_confidence,
                           m.mention_tone
                        FROM 
                          (SELECT
                            GLOBALEVENTID AS global_event_id,
                            EventTimeDate AS event_time_15min,
                            MentionTimeDate AS mention_time_15min,
                            Confidence AS gdelt_confidence,
                            MentionDocTone AS mention_tone
                          FROM 
                            `gdelt-bq.gdeltv2.eventmentions_partitioned`
                            WHERE 
                               Confidence >= 50) m
                            INNER JOIN 
                            (SELECT
                            ActionGeo_FullName AS country,
                            GLOBALEVENTID AS global_event_id,
                            QuadClass AS cameo_class,
                            EventCode AS cameo_code,
                            EventBaseCode AS cameo_base_code,
                            EventRootCode AS cameo_root_code,
                            AvgTone AS event_avg_tone,
                            GoldsteinScale AS goldstein
                          FROM
                            `gdelt-bq.gdeltv2.events`
                          WHERE
                            Year >= EXTRACT(YEAR FROM CURRENT_TIMESTAMP()) - 2
                            AND ActionGeo_CountryCode IN {0}
                            AND EventRootCode IN {1}
                            AND QuadClass IN (3,4)) e
                            ON m.global_event_id = e.global_event_id""".format(PILOT_IOS2, PILOT_EVENTTYPES)
