"""
çeşitli bilgileri içerem yaml uzantılı dosyanın hazırlanması gerekli. datasets klasörü içerisinde images ve labels olarak iki klasörde veriler toplanmalı.
maske verisi varsa bunları txt ye dönüştürecek kod max_to_txt.py dosyası ile yapılır. kayıt edilen model yeni örnekleri tahmin etmek için kullanılır.

Doç.Dr. Ali POLAT-2024
"""

from ultralytics import YOLO
def run():
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
            batch=12,
            #device="cpu",
            data="config.yaml",
            epochs=100,
            imgsz=256,
        )
  

if __name__ == '__main__':
    run()

