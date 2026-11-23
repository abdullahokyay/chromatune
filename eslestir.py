import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ozellikler import oku, parlaklik_doygunluk, kenar_yogunlugu, renk_sicakligi, kontrast, canli_oran

def sinirla(x):
    return float(np.clip(x, 0, 1))

def gorselden_muzige(img):
    doygunluk, parlaklik = parlaklik_doygunluk(img)
    sicaklik = renk_sicakligi(img)
    p = sinirla((parlaklik - 0.2) / 0.6)
    d = sinirla(doygunluk)
    s = sinirla((sicaklik + 1) / 2)
    kon = sinirla((kontrast(img) - 0.3) / 0.3)
    canli = sinirla(canli_oran(img) / 0.5)
    valence = 0.4 * p + 0.3 * d + 0.3 * s
    energy = 0.4 * kon + 0.3 * d + 0.3 * canli
    return valence, energy

def oneri(img, sarkilar, model, n=5):
    v, e = gorselden_muzige(img)
    idx = model.kneighbors([[v, e]], n_neighbors=n, return_distance=False)[0]
    return v, e, sarkilar.iloc[idx]

if __name__ == "__main__":
    sarkilar = pd.read_csv("data/turkce_sarkilar.csv")
    model = NearestNeighbors().fit(sarkilar[["valence", "energy"]].values)
    img = oku(sys.argv[1])
    v, e, sonuc = oneri(img, sarkilar, model)
    print("hedef valence:", round(v, 2), "energy:", round(e, 2))
    print(sonuc[["track_name", "artists", "valence", "energy"]].to_string(index=False))