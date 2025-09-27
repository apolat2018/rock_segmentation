from ultralytics import YOLO
import cv2

# Modeli yükle (hazır model ya da kendi eğittiğiniz .pt dosyası)
model = YOLO("runs/segment/train4/weights/best.pt")

# Tahmin yap (confidence eşiği ekledik)
results = model.predict("datasets/images/val/187.png", conf=0.5, save=True)

# Sonuçları göster
for r in results:
    im = r.plot()  # kutular + maskeler + etiket + confidence çizilmiş görüntü
    cv2.imshow("Sonuc", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
