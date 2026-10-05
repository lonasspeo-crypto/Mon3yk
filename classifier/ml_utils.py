"""
Utilitas ML - pipeline dua tahap:
  1) detector.crop_ke_hewan()  : OWLv2 zero-shot mencari primata, gambar di-crop ke kotaknya
  2) EfficientNet-B0           : klasifikasi 3 kelas pada gambar hasil crop

Model dan detektor di-load sekali per proses server (cached di memori).
Kalau detektor tidak menemukan primata, dipakai gambar utuh (fallback), sama seperti
saat evaluasi di Colab.
"""

import logging
from pathlib import Path

import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

IMG_SIZE = (288, 288)
MODEL_FILENAME = "primate_classifier_efficientnetb0_crop.keras"

# Urutan = alfabet nama folder = class_names di config_pipeline.json
CLASS_NAMES = ["bekantan", "orangutan_kalimantan", "orangutan_sumatra"]
CLASS_INDICES = {nama: i for i, nama in enumerate(CLASS_NAMES)}

USE_DETECTOR = True  # False = klasifikasi gambar utuh (tanpa torch/transformers)

_model = None  # cache model klasifikasi


def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model

        model_path = Path(settings.ML_MODEL_DIR) / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {model_path}. "
                f"Pastikan file '{MODEL_FILENAME}' sudah ditaruh di classifier/ml_model/."
            )
        _model = load_model(model_path, compile=False)  # inferensi saja, tidak perlu optimizer
    return _model


_get_model = get_model  # alias nama lama


def predict_image_detail(image_file):
    """
    image_file: file-like (dari request.FILES), pointer di awal.
    Return: (results, info)
      results = list of (label, confidence 0..1), diurutkan dari tertinggi
      info    = {"terdeteksi": bool, "skor": float|None, "kotak": tuple|None}
    """
    from PIL import Image

    img = Image.open(image_file).convert("RGB")
    info = {"terdeteksi": False, "skor": None, "kotak": None}

    if USE_DETECTOR:
        from .detector import crop_ke_hewan

        try:
            img, info = crop_ke_hewan(img)
        except Exception:
            # Detektor bermasalah -> tetap klasifikasi gambar utuh, tapi tercatat di log server.
            logger.exception("Detektor gagal, memakai gambar utuh (fallback)")

    # bilinear, sama dengan saat training/evaluasi di Colab
    arr = np.array(img.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # model sudah punya preprocessing sendiri (input 0-255)

    preds = get_model().predict(arr, verbose=0)[0]
    results = [(CLASS_NAMES[i], float(preds[i])) for i in range(len(preds))]
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results, info


def predict_image(image_file):
    """Antarmuka lama (views.py tidak perlu diubah): hanya mengembalikan daftar hasil."""
    return predict_image_detail(image_file)[0]
