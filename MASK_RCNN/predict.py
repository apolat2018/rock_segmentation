import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# --- 1. Detectron2 modelini yükle ---
cfg = get_cfg()
cfg.merge_from_file("models/config.yaml")  # kendi config
cfg.MODEL.WEIGHTS = "models/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# --- 2. Büyük görüntüyü oku ---
img = cv2.imread("tif1.png")
H, W, C = img.shape
patch_size = 256

# --- 3. Boş canvas oluştur (tahminleri üzerine çizmek için) ---
canvas = img.copy()

# --- 4. Patch’leri dolaş ---
for y in range(0, H, patch_size):
    for x in range(0, W, patch_size):
        patch = img[y:y+patch_size, x:x+patch_size]

        # Eğer patch sınırları görüntü boyutunu aşarsa boyutu düzelt
        ph, pw, _ = patch.shape
        if ph != patch_size or pw != patch_size:
            pad_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            pad_patch[:ph, :pw] = patch
            patch = pad_patch

        # --- 5. Tahmin ---
        outputs = predictor(patch)

        # --- 6. Görselleştir ---
        v = Visualizer(patch[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        patch_result = out.get_image()[:, :, ::-1]

        # --- 7. Orijinal canvas’a geri yerleştir ---
        canvas[y:y+patch_size, x:x+patch_size] = patch_result[:ph, :pw]

# --- 8. Göster ve kaydet ---
cv2.imshow("Tahminler", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("large_image_prediction.jpg", canvas)
