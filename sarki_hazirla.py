import pandas as pd

d = pd.read_csv("data/tracks.csv")
t = d[d["track_genre"] == "turkish"].copy()
t = t.drop_duplicates(subset="track_id")
t = t.drop_duplicates(subset=["track_name", "artists"])

sutunlar = ["track_id", "track_name", "artists", "valence", "energy", "tempo", "danceability", "mode", "popularity"]
t = t[sutunlar].reset_index(drop=True)
t.to_csv("data/turkce_sarkilar.csv", index=False)

print(len(t))
print(t[["valence", "energy", "tempo", "danceability"]].describe().round(2))
print(t.sample(15, random_state=1)[["track_name", "artists", "valence", "energy", "tempo"]].to_string(index=False))