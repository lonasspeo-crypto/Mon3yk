import os
from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    name = 'classifier'

    def ready(self):
        # runserver menjalankan 2 proses (auto-reloader). RUN_MAIN == 'true' hanya di
        # proses anak yang melayani request, jadi model cuma di-load sekali.
        if os.environ.get('RUN_MAIN') != 'true':
            return

        from . import ml_utils
        try:
            print("Memuat model EfficientNet-B0, mohon tunggu...")
            ml_utils.get_model()
            if ml_utils.USE_DETECTOR:
                from .detector import get_detector
                print("Memuat detektor OWLv2 (unduhan pertama sekitar 600 MB)...")
                get_detector()
            print("Model siap. Server siap menerima klasifikasi.")
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
        except ImportError as e:
            print(f"[WARNING] Pustaka detektor belum terpasang: {e}. "
                  f"Jalankan: pip install torch transformers")
