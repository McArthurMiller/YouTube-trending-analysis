import pandas as pd

ARCHIVO = "data/usvideos_limpio.csv"
df = pd.read_csv(ARCHIVO)

df["controversia"] = df["dislikes"] / (df["likes"] + df["dislikes"])

videos = df.drop_duplicates(subset="video_id", keep="last")
videos = videos[(videos["ratings_disabled"] == False)]

print(len(videos))
print(videos["controversia"].describe())

print(videos.nlargest(5, "controversia")[["title", "channel_title", "likes", "dislikes", "controversia"]])

resumen = videos.groupby("publish_weekday").agg(
    mediana=("controversia", "median"),
    videos=("views", "count")
)

print(resumen.sort_values("mediana", ascending=False))

videos.to_csv("data/videos_controversia.csv", index=False)