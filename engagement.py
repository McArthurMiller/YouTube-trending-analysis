import pandas as pd
from pathlib import Path

ARCHIVO = "data/usvideos_limpio.csv"
df = pd.read_csv(ARCHIVO)

df["engagement"] = (df["likes"] + df["comment_count"]) / df["views"]

print(df[["title", "views", "likes", "comment_count", "engagement"]].head())
print(df["engagement"].describe())

videos = df.drop_duplicates(subset="video_id", keep="last")
videos = videos[(videos["ratings_disabled"] == False) & (videos["comments_disabled"] == False)]

print(len(videos))
print(videos["engagement"].describe())

print(videos.nsmallest(5, "engagement")[["title", "views", "likes", "comment_count", "engagement"]])

tabla = videos.pivot_table(index="category_name", values="engagement", aggfunc="median")

print(tabla.sort_values("engagement", ascending=False))

resumen = videos.groupby("category_name").agg(
    mediana=("engagement", "median"),
    videos=("views", "count")
)

print(resumen.sort_values("mediana", ascending=False))

videos.to_csv("data/videos_engagement.csv", index=False)