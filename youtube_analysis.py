# ––– 1. Importa Librerias –––

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ––– 2. Carga de Datos –––

df = pd.read_csv("data/USvideos.csv")

# ––– 3. Exploración Inicial –––

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.dtypes)

# ––– 4. Estadistica Basica –––

print(df[["views","likes","dislikes","comment_count"]].describe())

# ––– 5. Top 10 canales en Trending –––

top_canales = df["channel_title"].value_counts().head(10)
print(top_canales)

# ––– 6. Graficando el Top 10 –––

plt.figure(figsize=(12,5))
top_canales.plot(kind="bar", color="steelblue")
plt.title("Top 10 Canales en Trending")
plt.xlabel("Canal")
plt.ylabel("Número de videos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ––– 7. Vistas promedio por categoría –––

category_map = {
    1: "Film & Animation", 2: "Autos", 10: "Music",
    15: "Pets", 17: "Sports", 19: "Travel", 20: "Gaming",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News", 26: "How-to", 27: "Education",
    28: "Science", 29: "Nonprofits"
}

df["category_name"] = df["category_id"].map(category_map)

promedio_categoria = df.groupby("category_name")["views"].mean().sort_values(ascending=False)
print(promedio_categoria)

# ––– 8. Graficando las vistas por categoria –––

plt.figure(figsize=(12,5))
promedio_categoria.plot(kind="bar", color="coral")
plt.title("Promedio de Vistas por Categoria")
plt.xlabel("Categoria")
plt.ylabel("Vistas promedio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("categorias.png")
plt.show()

# ––– 9. Relación entre Likes y Vistas –––

plt.figure(figsize=(8,5))
plt.scatter(df["views"], df["likes"], alpha=0.3, color="purple")
plt.title("Relación entre Vistas y Likes")
plt.xlabel("Vistas")
plt.ylabel("Likes")
plt.tight_layout()
plt.show()

# ––– 10. Analisis de Trending por dia de la semana –––

df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")
df["dia_semana"] = df["trending_date"].dt.day_name()

orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
conteo_dias = df["dia_semana"].value_counts().reindex(orden_dias)

print(conteo_dias)

plt.figure(figsize=(10,5))
conteo_dias.plot(kind="bar", color="green")
plt.title("Videos en Trending por Dia de la Semana")
plt.xlabel("Dia")
plt.ylabel("Número de videos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ––– 11. Top 10 Videos con más vistas –––

top_videos = df[["title","channel_title", "views"]].sort_values("views", ascending=False).drop_duplicates(subset="title").head(10)
print(top_videos.to_string(index=False))

# ––– 12. Calculando Likes / Vistas –––

df["engagement_ratio"] = df["likes"] / df["views"]
top_engagement = df[["title","channel_title","views","likes","engagement_ratio"]].sort_values("engagement_ratio",ascending=False).drop_duplicates(subset="title").head(10)

print(top_engagement.to_string(index=False))
 
# ––– 13. Gráfica de Distribución de Vstas –––

plt.figure(figsize=(10,5))
df[df["views"] < 10000000]["views"].plot(kind="hist", bins=50, color="teal")
plt.title("Distribución de Vistas (videos bajo 10M)")
plt.xlabel("Vistas")
plt.ylabel("Cantidad de videos")
plt.tight_layout()
plt.show()