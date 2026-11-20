import sys
import cv2
import numpy as np

def oku(yol, genislik=800):
    img = cv2.imread(yol)
    if img is None:
        raise FileNotFoundError(yol)
    oran = genislik / img.shape[1]
    return cv2.resize(img, (genislik, int(img.shape[0] * oran)))

def parlaklik_doygunluk(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return s.mean() / 255, v.mean() / 255

def kenar_yogunlugu(img):
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gri = cv2.GaussianBlur(gri, (5, 5), 0)
    kenar = cv2.Canny(gri, 100, 200)
    return (kenar > 0).mean()

def renk_sicakligi(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gecerli = (s > 40) & (v > 40)
    if gecerli.sum() == 0:
        return 0.0
    ton = h[gecerli]
    sicak = ((ton <= 25) | (ton >= 160)).sum()
    soguk = ((ton >= 90) & (ton <= 130)).sum()
    return (sicak - soguk) / len(ton)
def kontrast(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2].std() / 128

def canli_oran(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return ((s > 128) & (v > 100)).mean()

if __name__ == "__main__":
    img = oku(sys.argv[1])
    doygunluk, parlaklik = parlaklik_doygunluk(img)
    print("doygunluk:", round(doygunluk, 3))
    print("parlaklik:", round(parlaklik, 3))
    print("kenar:", round(kenar_yogunlugu(img), 3))
    print("sicaklik:", round(renk_sicakligi(img), 3))