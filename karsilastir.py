import os
import pandas as pd
from ozellikler import oku, parlaklik_doygunluk, kontrast, canli_oran
from eslestir import gorselden_muzige, sinirla

def yeni_enerji(img):
    doygunluk, parlaklik = parlaklik_doygunluk(img)
    d = sinirla(doygunluk)
    kon = sinirla((kontrast(img) - 0.3) / 0.3)
    canli = sinirla(canli_oran(img) / 0.5)
    return 0.4 * kon + 0.3 * d + 0.3 * canli

satirlar = []
for ad in sorted(os.listdir("images")):
    if not ad.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = oku(os.path.join("images", ad))
    v, e_eski = gorselden_muzige(img)
    satirlar.append({
        "gorsel": ad,
        "valence": round(v, 2),
        "energy_eski": round(e_eski, 2),
        "energy_yeni": round(yeni_enerji(img), 2),
    })
tablo = pd.DataFrame(satirlar)
tablo.to_csv("data/karsilastirma.csv", index=False)
print(tablo.to_string(index=False))