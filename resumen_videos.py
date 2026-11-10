import pandas as pd
from pathlib import Path

# 1. Rutas
ARCHIVO_LIMPIO = Path("data") / "usvideos_limpio.csv"
ARCHIVO_VIDEOS = Path("data") / "videos_resumen.csv"

# 2. Cargar el CSV limpio, para indicar cuáles columnas son fechas
df = pd.read_csv(ARCHIVO_LIMPIO, parse_dates = ["trending_date", "publish_time"])

# 3. Cuantas filas y cuantos videos distintos hay?
print(f"Filas: {df.shape[0]}")
print(f"Videos distintos: {df['video_id'].nunique()}")

# 4. Agrupar por video: una fila por cada video_id
videos = (
    df.groupby("video_id")
    .agg(
        title=("title", "last"),
        channel_title=("channel_title", "first"),
        category_name=("category_name", "first"),
        publish_time=("publish_time", "first"),
        first_trending=("trending_date", "min"),
        last_trending=("trending_date", "max"),
        days_in_trending=("trending_date", "nunique"),
        max_views=("views", "max"),
    )
    .reset_index()
)

print(videos.shape)
print(videos.head())

# 5. Mostrar todas las columnas al imprimir
pd.set_option("display.max_columns", None)
print(videos.head())

# 6. Comparar la suma inflada con la real
print(f"Suma inflada (una vez por fila): {df['views'].sum():,}")
print(f"Suma real (una vez por video): {videos['max_views'].sum():,}")

# 7. Cuantos dias dura un video en tendencia?
print(videos["days_in_trending"].describe())

videos["days_to_first_trending"] = (videos["first_trending"] - videos["publish_time"].dt.normalize()).dt.days
print(videos["days_to_first_trending"].describe())

# 8. Guardar la tabla de una fila por video
videos.to_csv(ARCHIVO_VIDEOS, index=False)
print(f"Guardado: {ARCHIVO_VIDEOS}")
