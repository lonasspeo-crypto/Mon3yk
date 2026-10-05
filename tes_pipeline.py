"""
Uji pipeline dari terminal TANPA browser. Jalankan dari folder yang ada manage.py:
    python tes_pipeline.py foto.jpg
"""
import os
import sys
import time

import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "primata_project.settings")
django.setup()

from classifier import ml_utils  # noqa: E402

if len(sys.argv) < 2:
    sys.exit("Pemakaian: python tes_pipeline.py foto.jpg")

for ulang in (1, 2):  # run ke-2 = waktu normal (model & detektor sudah di memori)
    t = time.time()
    with open(sys.argv[1], "rb") as f:
        hasil, info = ml_utils.predict_image_detail(f)
    print(f"\n[Run {ulang}] {time.time() - t:.1f} detik")
    print("Deteksi :", info)
    for label, conf in hasil:
        print(f"  {label:<22} {conf * 100:5.1f}%")
