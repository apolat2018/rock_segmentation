
import numpy as np
from ultralytics import YOLO
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2

# YOLOv8 modelini yükle (segmentasyon modeli olmalı!)
model = YOLO("runs/segment/train4/weights/best.pt")

# Büyük görüntüyü yükle
image_path = "tif1.png"
image = cv2.imread(image_path)
h, w, _ = image.shape
confidence=0.75
# 512x512 bölme parametreleri
tile_size = 256
overlap = 0  # Çakışma miktarı (isteğe bağlı artırılabilir)

# Tüm segmentasyon sonuçlarını saklayacak liste
detections = []

for y in range(0, h, tile_size - overlap):
    for x in range(0, w, tile_size - overlap):
        # Parçanın sınırlarını belirle
        x_end = min(x + tile_size, w)
        y_end = min(y + tile_size, h)
        tile = image[y:y_end, x:x_end].copy()

        # Eğer parça 512x512 değilse, sıfırlarla doldur
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded_tile[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded_tile

        # YOLO tahmini yap
        results = model(tile)

        # Sonuçları büyük görüntüye göre hizalayarak sakla
        for result in results:
            if result.masks is not None:  # Eğer maske varsa işle
                for i, mask in enumerate(result.masks.xy):
                    conf = result.boxes.conf[i].item()  # Güven skorunu al
                    if conf >= confidence:  # Eğer güven skoru 0.7'den büyükse kaydet
                        mask[:, 0] += x  # x koordinatlarını kaydır
                        mask[:, 1] += y  # y koordinatlarını kaydır
                        detections.append(mask)

# Büyük görüntü üzerine segmentasyon maskelerini çiz
new_img=np.zeros((h,w),np.uint8)
for mask in detections:
    mask = np.array(mask, dtype=np.int32)
    cv2.polylines(image, [mask], isClosed=True, color=(255, 0, 0), thickness=4)
    cv2.fillPoly(new_img, [mask],  color=(255, 0, 0))

# Sonucu kaydet 
cv2.imwrite(f"output_segmented_{str(confidence)}.png", image)
cv2.imwrite(f"output_binary_{str(confidence)}.png", new_img)
