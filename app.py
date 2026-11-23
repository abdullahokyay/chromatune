import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from eslestir import gorselden_muzige, sinirla
from ozellikler import parlaklik_doygunluk, renk_sicakligi, kontrast, canli_oran

@st.cache_data
def sarkilari_yukle():
    return pd.read_csv("data/turkce_sarkilar.csv")

def goruntu_coz(dosya, genislik=800):
    veri = np.frombuffer(dosya.getvalue(), np.uint8)
    img = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    oran = genislik / img.shape[1]
    return cv2.resize(img, (genislik, int(img.shape[0] * oran)))

def aciklama(doygunluk, parlaklik, sicaklik, kon, canli):
    parcalar = []
    parcalar.append("parlak" if parlaklik > 0.5 else "karanlık")
    parcalar.append("canlı renkli" if doygunluk > 0.45 else "soluk renkli")
    if sicaklik > 0.2:
        parcalar.append("sıcak tonlu")
    elif sicaklik < -0.2:
        parcalar.append("soğuk tonlu")
    else:
        parcalar.append("nötr tonlu")
    parcalar.append("yüksek kontrastlı" if kon > 0.5 else "düşük kontrastlı")
    return ", ".join(parcalar)

st.title("chromatune")
st.write("Görsel yükle, ona uygun Türkçe şarkıları bulalım.")

sarkilar = sarkilari_yukle()
model = NearestNeighbors().fit(sarkilar[["valence", "energy"]].values)

dosya = st.file_uploader("Görsel seç", type=["jpg", "jpeg", "png"])

if dosya is not None:
    img = goruntu_coz(dosya)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")

    doygunluk, parlaklik = parlaklik_doygunluk(img)
    sicaklik = renk_sicakligi(img)
    kon = kontrast(img)
    canli = canli_oran(img)
    v, e = gorselden_muzige(img)

    st.subheader("Görselden ne çıkardık?")
    ozellikler = pd.DataFrame(
        {
            "değer": [
                sinirla((parlaklik - 0.2) / 0.6),
                sinirla(doygunluk),
                sinirla((sicaklik + 1) / 2),
                sinirla((kon - 0.3) / 0.3),
                sinirla(canli / 0.5),
            ]
        },
        index=["parlaklık", "doygunluk", "sıcaklık", "kontrast", "canlı renk oranı"],
    )
    st.bar_chart(ozellikler)
    st.write("Bu görsel " + aciklama(doygunluk, parlaklik, sicaklik, kon, canli) + ".")
    st.write(f"Bu yüzden hedef neşe (valence): **{v:.2f}**, enerji: **{e:.2f}**")

    st.subheader("Önerilen şarkılar")
    idx = model.kneighbors([[v, e]], n_neighbors=5, return_distance=False)[0]
    for _, s in sarkilar.iloc[idx].iterrows():
        link = "https://open.spotify.com/track/" + s["track_id"]
        st.markdown(
            f"**{s['track_name']}** — {s['artists']}  \n"
            f"neşe {s['valence']:.2f} | enerji {s['energy']:.2f} | [Spotify'da aç]({link})"
        )