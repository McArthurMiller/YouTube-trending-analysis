import duckdb

ARCHIVO = "data/usvideos_limpio.csv"

# 1. ¿Cuántas filas y cuántos videos distintos hay?
consulta = f"""
SELECT
    COUNT(*)                 AS filas,
    COUNT(DISTINCT video_id) AS videos_distintos
FROM '{ARCHIVO}'
"""

duckdb.sql(consulta).show()

consulta = f"""
SELECT
    video_id,
    ANY_VALUE(channel_title)        AS channel_title,
    MIN(trending_date)              AS first_trending,
    MAX(trending_date)              AS last_trending,
    COUNT(DISTINCT trending_date)   AS days_in_trending,
    MAX(views)                      AS max_views
FROM '{ARCHIVO}'
GROUP BY video_id
ORDER BY max_views DESC
LIMIT 5
"""

duckdb.sql(consulta).show()

consulta = f"""
SELECT
    category_name,
    COUNT(DISTINCT video_id) AS videos
FROM '{ARCHIVO}'
GROUP BY category_name
ORDER BY  videos DESC
LIMIT 5
"""

duckdb.sql(consulta).show()


consulta = f"""
SELECT
    channel_title,
    COUNT(DISTINCT video_id) AS videos_distintos 
FROM '{ARCHIVO}'
GROUP BY channel_title
ORDER BY videos_distintos DESC
LIMIT 5
"""

duckdb.sql(consulta).show()