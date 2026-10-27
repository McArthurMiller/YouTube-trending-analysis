# ––– Limpieza del dataset de YouTube para usarlo en Tableau –––

import pandas as pd
from pathlib import Path

# 1. Rutas: el CSV vive dentro del proyecto, en la carpeta data/
CARPETA_DATOS = Path("data")
ARCHIVO_ORIGINAL = CARPETA_DATOS / "USvideos.csv"
ARCHIVO_LIMPIO = CARPETA_DATOS / "usvideos_limpio.csv"

df = pd.read_csv(ARCHIVO_ORIGINAL)
print(f"Filas originales: {df.shape[0]}")

# 2. Fechas: de texto a datetime
df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")
df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True).dt.tz_localize(None)

# 3. Columnas nuevas a partir de publish_time (hora en UTC)
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_weekday"] = df["publish_time"].dt.day_name()

# 4. Nombre de la categoría
category_map = {
    1: "Film & Animation", 2: "Autos", 10: "Music",
    15: "Pets", 17: "Sports", 19: "Travel", 20: "Gaming",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News", 26: "How-to", 27: "Education",
    28: "Science", 29: "Nonprofits", 43: "Shows"
}
df["category_name"] = df["category_id"].map(category_map).fillna("Otra")
print("IDs sin nombre:", df.loc[df["category_name"] == "Otra", "category_id"].unique())

#print(df[["trending_date", "publish_time"]].head())

df["days_to_trending"] = (df["trending_date"] - df["publish_time"].dt.normalize()).dt.days
print(df["days_to_trending"].describe())

# 5. Quitar columnas que Tableau no necesita
df = df.drop(columns=["thumbnail_link", "description"])

# 6. Guardar el CSV limpio
df.to_csv(ARCHIVO_LIMPIO, index=False)
print(f"Guardado: {ARCHIVO_LIMPIO} ({df.shape[0]} filas, {df.shape[1]} columnas)")
print(df.dtypes)
