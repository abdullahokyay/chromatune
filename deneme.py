import os
import pandas as pd
from ozellikler import oku, kontrast, canli_oran

satirlar = []
for ad in sorted(os.listdir("images")):
    if not ad.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = oku(os.path.join("images", ad))
    satirlar.append({
        "gorsel": ad,
        "kontrast": round(kontrast(img), 3),
        "canli": round(canli_oran(img), 3),
    })

print(pd.DataFrame(satirlar).to_string(index=False))