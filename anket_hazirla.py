import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image

giris = "ham"
cikis = "anket"
os.makedirs(cikis, exist_ok=True)

def guvenli_oku(yol):
    veri = np.fromfile(yol, dtype=np.uint8)
    img = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(yol).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

adlar = [a for a in sorted(os.listdir(giris)) if a.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
random.seed(42)
random.shuffle(adlar)

kayit = []
for ad in adlar:
    img = guvenli_oku(os.path.join(giris, ad))
    if img is None:
        print("okunamadi:", ad)
        continue
    oran = 1000 / img.shape[1]
    img = cv2.resize(img, (1000, int(img.shape[0] * oran)))
    yeni = "s" + str(len(kayit) + 1).zfill(2) + ".jpg"
    cv2.imwrite(os.path.join(cikis, yeni), img)
    kayit.append({"anket_adi": yeni, "ham_adi": ad})

pd.DataFrame(kayit).to_csv("data/anket_anahtar.csv", index=False, encoding="utf-8-sig")
print(len(kayit), "gorsel hazirlandi")