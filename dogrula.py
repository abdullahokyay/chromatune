import pandas as pd
from scipy.stats import spearmanr

t = pd.read_csv("data/karsilastirma.csv")
e = pd.read_csv("data/etiketler.csv")
m = t.merge(e, on="gorsel").dropna()

for sutun in ["energy_eski", "energy_yeni"]:
    r, p = spearmanr(m[sutun], m["beklenen"])
    print(sutun, "korelasyon:", round(r, 2), "p:", round(p, 3))