import os
import json
import numpy as np
from PIL import Image
from skimage import measure

# Giriş klasörleri train ve val için ayrı ayrı yapılmalı
image_dir = "C:\\AP\\maskrcnn_cpu\\dataset\\val\\images"   # Orijinal görüntüler
mask_dir = "C:\\AP\\maskrcnn_cpu\\dataset\\val\\masks"     # Maske görüntüleri (PNG)
output_json = "val.json"

# COCO formatı için temel şablon
coco_format = {
    "info": {
        "description": "Custom Dataset",
        "version": "1.0"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# Tek sınıf örneği (ör: "object")
coco_format["categories"].append({
    "id": 1,
    "name": "rock"
})

image_id = 1
annotation_id = 1

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Orijinal görüntü bilgisi
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)  # maskeler aynı isimde olmalı

        with Image.open(image_path) as img:
            width, height = img.size

        coco_format["images"].append({
            "id": image_id,
            "file_name": os.path.join(image_dir, filename),
            "width": width,
            "height": height
        })

        # Maskeyi yükle
        mask = np.array(Image.open(mask_path).convert("L"))
        # Nesne olan pikseller: 1
        mask = mask > 0

        # Maske kontur -> poligon
        contours = measure.find_contours(mask, 0.5)

        for contour in contours:
            contour = np.flip(contour, axis=1)  # x, y sırası

            segmentation = contour.ravel().tolist()
            if len(segmentation) < 6:  # en az 3 nokta
                continue

            # bounding box hesapla
            x_min, y_min = np.min(contour, axis=0)
            x_max, y_max = np.max(contour, axis=0)
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": float(mask.sum()),
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

# JSON kaydet
with open(output_json, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"✅ COCO JSON başarıyla oluşturuldu: {output_json}")