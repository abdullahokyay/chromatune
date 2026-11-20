import os
import pandas as pd
from ozellikler import oku, parlaklik_doygunluk, kenar_yogunlugu, renk_sicakligi

klasor = "images"
satirlar = []

for ad in sorted(os.listdir(klasor)):
    if not ad.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = oku(os.path.join(klasor, ad))
    doygunluk, parlaklik = parlaklik_doygunluk(img)
    satirlar.append({
        "gorsel": ad,
        "doygunluk": round(doygunluk, 3),
        "parlaklik": round(parlaklik, 3),
        "kenar": round(kenar_yogunlugu(img), 3),
        "sicaklik": round(renk_sicakligi(img), 3),
    })

tablo = pd.DataFrame(satirlar)
tablo.to_csv("data/gorsel_ozellikleri.csv", index=False)
print(tablo.to_string(index=False))