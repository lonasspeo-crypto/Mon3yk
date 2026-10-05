"""
Tahap 1 pipeline dua tahap: deteksi primata (OWLv2 zero-shot) lalu crop.
Dipakai di ml_utils.predict_image() SEBELUM klasifikasi EfficientNet-B0.

Taruh file ini di folder classifier/ (sejajar ml_utils.py).
Parameter (query, threshold, margin) dibaca dari classifier/ml_model/config_pipeline.json
kalau file itu ada, supaya SAMA dengan yang dipakai saat evaluasi di Colab.
"""

import json
from pathlib import Path

DET_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a monkey", "an orangutan", "an ape", "a primate"]
DET_THRESHOLD = 0.10
CROP_MARGIN = 0.20

_bundle = None  # (processor, model, params), di-load sekali per proses server


def hitung_kotak_crop(box, lebar, tinggi, margin=CROP_MARGIN):
    """
    box = (x1, y1, x2, y2) dalam piksel gambar asli.
    Kotak diperbesar `margin` di tiap sisi, dijadikan persegi (sisi terpanjang)
    di sekitar pusat kotak, lalu digeser/dibatasi supaya tetap di dalam gambar.
    Return (x1, y1, x2, y2) integer.
    """
    x1, y1, x2, y2 = box
    # clamp tiap koordinat ke dalam gambar (aman juga untuk kotak yang keluar batas)
    x1, x2 = sorted((min(max(x1, 0.0), lebar), min(max(x2, 0.0), lebar)))
    y1, y2 = sorted((min(max(y1, 0.0), tinggi), min(max(y2, 0.0), tinggi)))
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sisi = max(w, h) * (1 + 2 * margin)
    sisi = max(1.0, min(sisi, lebar, tinggi))  # persegi tidak boleh lebih besar dari gambar
    sx = min(max(cx - sisi / 2, 0), lebar - sisi)
    sy = min(max(cy - sisi / 2, 0), tinggi - sisi)
    return int(round(sx)), int(round(sy)), int(round(sx + sisi)), int(round(sy + sisi))


def get_detector():
    """Load OWLv2 sekali (cached). Panggil juga di apps.py supaya preload saat server start."""
    global _bundle
    if _bundle is None:
        import torch
        from django.conf import settings
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        params = {"queries": QUERIES, "threshold": DET_THRESHOLD, "margin": CROP_MARGIN,
                  "nama": DET_MODEL_NAME}
        cfg_path = Path(settings.ML_MODEL_DIR) / "config_pipeline.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            params["queries"] = cfg.get("det_queries", params["queries"])
            params["threshold"] = cfg.get("det_threshold", params["threshold"])
            params["margin"] = cfg.get("crop_margin", params["margin"])
            params["nama"] = cfg.get("detektor", params["nama"])

        processor = Owlv2Processor.from_pretrained(params["nama"])
        model = Owlv2ForObjectDetection.from_pretrained(params["nama"]).eval()
        _bundle = (processor, model, params)
    return _bundle


def crop_ke_hewan(img):
    """
    img: PIL.Image RGB. Return (gambar_hasil, info).
    Kalau tidak ada deteksi dengan skor >= threshold -> kembalikan gambar utuh (fallback).
    info = {"terdeteksi": bool, "skor": float|None, "kotak": tuple|None}
    """
    import torch

    processor, model, params = get_detector()
    lebar, tinggi = img.size
    inputs = processor(text=[params["queries"]], images=img, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    # OWLv2 mem-pad gambar jadi persegi -> target_sizes pakai sisi terpanjang
    sisi = max(lebar, tinggi)
    hasil = processor.post_process_object_detection(
        outputs, threshold=params["threshold"],
        target_sizes=torch.tensor([[sisi, sisi]]),
    )[0]

    if len(hasil["scores"]) == 0:
        return img, {"terdeteksi": False, "skor": None, "kotak": None}

    i = int(hasil["scores"].argmax())
    skor = float(hasil["scores"][i])
    kotak = hitung_kotak_crop(hasil["boxes"][i].tolist(), lebar, tinggi, params["margin"])
    # kotak degenerate (lebar/tinggi ~0) -> perlakukan sebagai tidak terdeteksi
    if kotak[2] - kotak[0] < 8 or kotak[3] - kotak[1] < 8:
        return img, {"terdeteksi": False, "skor": skor, "kotak": None}
    return img.crop(kotak), {"terdeteksi": True, "skor": skor, "kotak": kotak}
