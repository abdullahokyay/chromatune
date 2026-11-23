import io
import os
import uuid
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from eslestir import gorselden_muzige, sinirla
from ozellikler import parlaklik_doygunluk, renk_sicakligi, kontrast, canli_oran

GERI_BILDIRIM = "data/geri_bildirim.csv"

@st.cache_data
def sarkilari_yukle():
    return pd.read_csv("data/turkce_sarkilar.csv")

def goruntu_coz(dosya, genislik=800):
    ham = dosya.getvalue()
    img = cv2.imdecode(np.frombuffer(ham, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil = Image.open(io.BytesIO(ham)).convert("RGB")
        except Exception:
            return None
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    oran = genislik / img.shape[1]
    return cv2.resize(img, (genislik, int(img.shape[0] * oran)))

def aciklama(doygunluk, parlaklik, sicaklik, kon):
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

def kaydet(satir):
    var = os.path.exists(GERI_BILDIRIM)
    pd.DataFrame([satir]).to_csv(GERI_BILDIRIM, mode="a", header=not var, index=False, encoding="utf-8")

if "kullanici" not in st.session_state:
    st.session_state["kullanici"] = uuid.uuid4().hex[:8]

st.title("chromatune")
st.write("Görsel yükle, ona uygun Türkçe şarkıları bulalım.")

test_modu = st.sidebar.checkbox("Test modu (önce soruları sor, cevapları kaydet)", value=True)

sarkilar = sarkilari_yukle()
model = NearestNeighbors().fit(sarkilar[["valence", "energy"]].values)

dosya = st.file_uploader("Görsel seç", type=["jpg", "jpeg", "png", "webp", "avif"])

if dosya is not None:
    img = goruntu_coz(dosya)
    if img is None:
        st.error("Bu görsel okunamadı, başka bir dosya dene.")
        st.stop()
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
    fid = str(dosya.file_id)

    if test_modu:
        algi_anahtar = "algi_" + fid
        if algi_anahtar not in st.session_state:
            with st.form("algi_formu_" + fid):
                st.write("Önce sen söyle, bu görsel sana ne hissettiriyor?")
                e_algi = st.radio("Sakin (1) ↔ Enerjik (5)", [1, 2, 3, 4, 5], index=None, horizontal=True)
                n_algi = st.radio("Hüzünlü (1) ↔ Neşeli (5)", [1, 2, 3, 4, 5], index=None, horizontal=True)
                devam = st.form_submit_button("Önerileri göster")
            if devam:
                if e_algi is None or n_algi is None:
                    st.warning("İki soruyu da cevapla.")
                else:
                    st.session_state[algi_anahtar] = (e_algi, n_algi)
                    st.rerun()
            st.stop()
        e_algi, n_algi = st.session_state[algi_anahtar]

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
    st.write("Bu görsel " + aciklama(doygunluk, parlaklik, sicaklik, kon) + ".")
    st.write(f"Bu yüzden hedef neşe (valence): **{v:.2f}**, enerji: **{e:.2f}**")

    st.subheader("Önerilen şarkılar")
    idx = model.kneighbors([[v, e]], n_neighbors=5, return_distance=False)[0]
    oneriler = sarkilar.iloc[idx]
    for _, s in oneriler.iterrows():
        link = "https://open.spotify.com/track/" + s["track_id"]
        st.markdown(
            f"**{s['track_name']}** — {s['artists']}  \n"
            f"neşe {s['valence']:.2f} | enerji {s['energy']:.2f} | [Spotify'da aç]({link})"
        )

    if test_modu:
        kayit_anahtar = "kayit_" + fid
        if st.session_state.get(kayit_anahtar):
            st.success("Cevapların kaydedildi, teşekkürler! Yeni bir görsel yükleyebilirsin.")
        else:
            with st.form("uyum_formu_" + fid):
                uyum = st.radio("Bu şarkılar görsele uydu mu? (1 = hiç, 5 = çok)", [1, 2, 3, 4, 5], index=None, horizontal=True)
                gonder = st.form_submit_button("Gönder")
            if gonder:
                if uyum is None:
                    st.warning("Bir puan seç.")
                else:
                    kaydet(
                        {
                            "zaman": pd.Timestamp.now().isoformat(timespec="seconds"),
                            "kullanici": st.session_state["kullanici"],
                            "dosya": dosya.name,
                            "hedef_valence": round(v, 3),
                            "hedef_energy": round(e, 3),
                            "parlaklik": round(parlaklik, 3),
                            "doygunluk": round(doygunluk, 3),
                            "sicaklik": round(sicaklik, 3),
                            "kontrast": round(kon, 3),
                            "canli_oran": round(canli, 3),
                            "algi_enerji": e_algi,
                            "algi_nese": n_algi,
                            "uyum": uyum,
                            "oneriler": ";".join(oneriler["track_id"]),
                        }
                    )
                    st.session_state[kayit_anahtar] = True
                    st.rerun()