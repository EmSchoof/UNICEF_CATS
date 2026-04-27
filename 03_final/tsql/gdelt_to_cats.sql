
/** Microsoft TSQL Syntax for Manipulating Raw GDELT Data into CATS Base Variables 


/****** Object:  View [cats].[vw_export]    Script Date: 6/13/2021 6:13:05 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [cats].[vw_export]
AS
SELECT * FROM cats.export_2019
UNION ALL
SELECT * FROM cats.export_2020
UNION ALL
SELECT * FROM cats.export_2021
GO


/****** Object:  View [cats].[vw_mentions]    Script Date: 6/13/2021 6:13:47 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [cats].[vw_mentions]
AS
SELECT * FROM cats.mentions_2019
UNION ALL
SELECT * FROM cats.mentions_2020
UNION ALL
SELECT * FROM cats.mentions_2021
GO


/****** Object:  View [cats].[vw_ContextTrendAlert]    Script Date: 6/13/2021 6:14:05 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [cats].[vw_ContextTrendAlert]
AS
SELECT T1.GLOBALEVENTID,
T3.ActionGeo_CountryName,
T1.ActionGeo_CountryCode,
T3.UNICEF_regions,
CONVERT(date,LEFT(EventTimeDate,8)) AS EventDate,
CONVERT(Date,LEFT(MentionTimeDate, 8)) AS MentionDate,
CONVERT(INT,Confidence) AS Confidence,
EventCode,
(CASE
 WHEN EventRootCode = '08' THEN 'YIELD'
 WHEN EventRootCode = '09' THEN 'INVESTIGATE'
 WHEN EventRootCode = '10' THEN 'DEMAND'
 WHEN EventRootCode = '11' THEN 'DISAPPROVE'
 WHEN EventRootCode = '12' THEN 'REJECT'
 WHEN EventRootCode = '13' THEN 'THREATEN'
 WHEN EventRootCode = '14' THEN 'PROTEST'
 WHEN EventRootCode = '15' THEN 'EXHIBIT MILITARY POSTURE'
 WHEN EventRootCode = '16' THEN 'REDUCE RELATIONS'
 WHEN EventRootCode = '17' THEN 'COERCE'
 WHEN EventRootCode = '18' THEN 'ASSAULT'
 WHEN EventRootCode = '19' THEN 'FIGHT'
 WHEN EventRootCode = '20' THEN 'ENGAGE IN UNCONVENTIONAL MASS VIOLENCE'
 ELSE EventRootCode
 END) AS EventRootCode,
(CASE 
 WHEN QuadClass = '3' THEN 'Verbal Conflict' 
 WHEN QuadClass = '4' THEN 'Material Conflict' 
END) AS QuadClass,
MentionDocTone,
GoldsteinScale
FROM cats.vw_export AS T1
JOIN
 cats.vw_mentions AS T2
ON T1.GLOBALEVENTID=T2.GLOBALEVENTID
JOIN
  cats.tbl_CountryCode AS T3
ON T3.ActionGeo_CountryCode = T1.ActionGeo_CountryCode
WHERE CONVERT(INT,Confidence) >= 40
AND DATEDIFF(day,CONVERT(Date,LEFT(MentionTimeDate, 8)),CONVERT(date,LEFT(EventTimeDate,8))) <= 15
AND QuadClass IN ('3', '4')
AND T1.GLOBALEVENTID IS NOT NULL
AND EventTimeDate IS NOT NULL
AND MentionTimeDate IS NOT NULL
AND T1.ActionGeo_CountryCode IS NOT NULL
AND Confidence IS NOT NULL
AND MentionDocTone IS NOT NULL
AND EventRootCode IS NOT NULL
AND QuadClass IS NOT NULL
AND GoldsteinScale IS NOT NULL
GO
