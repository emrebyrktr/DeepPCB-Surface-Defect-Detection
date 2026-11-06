# 🔬 Industrial PCB Defect Detection — End-to-End MLOps Pipeline

[![CI Pipeline](https://github.com/emrebyrktr/Proje1/actions/workflows/ci.yml/badge.svg)](https://github.com/emrebyrktr/Proje1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.31-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioned-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

Bu proje; endüstriyel baskılı devre kartları (PCB) üzerindeki mikro üretim hatalarını tespit eden, ham veriden canlı Docker konteynerine kadar **9 aşamalı tam döngü (end-to-end) bir MLOps hattıdır**.

<!-- ✏️ DÜZENLEME ALANI: Varsa Swagger arayüzü veya tespit ekran görüntüsünü ekleyin -->
<!-- ![PCB Defect Detection Demo](docs/images/demo.png) -->

---

## 🎯 Projenin Amacı ve Kusur Sınıfları

DeepPCB veri seti (1500 çift görsel, 10.000+ anotasyon) kullanılarak, insan gözünün kaçırabileceği 6 mikro kusur sınıfı yüksek hassasiyetle tespit edilir:

1. **`open`** (Açık Devre)
2. **`short`** (Kısa Devre — En kritik üretim hatası)
3. **`mousebite`** (Kenar / Fare Isırığı Çapağı)
4. **`spur`** (Yol Çıkıntısı / İstenmeyen Bakır Sivrilmesi)
5. **`copper`** (Artık / İzolasyonsuz Bakır Lekesi)
6. **`pin-hole`** (İğne Deliği / Delik Boşluğu)

---

## 🏗️ MLOps Mimari Özellikleri

* **Veri Kalite Kontrolü & Sızıntı Önleme:** Ham VOC etiketleri normalize YOLO formatına çevrilirken `clamp(0.0, 1.0)` ile sınır dışı koordinatlar filtrelendi. Train / Valid / Test kümesi baştan bağımsız izole edildi.
* **Veri & Model Versiyonlama (DVC + Google Drive):** Büyük veri setleri ve ikili modeller Git'i şişirmemek için DVC uzak deposunda (Google Drive) hash işaretçileri (`.dvc`) ile kilitlendi.
* **Kapsamlı Soykütüğü (Lineage & Tracking):** Her eğitim koşusuna MLflow üzerinde `dvc_data_hash`, `git_commit`, donanım bilgisi (`device: cuda/cpu`) ve ağırlık dosyası `model_sha256` parmak izi mühürlendi.
* **Hafif Çıkarım (ONNX Runtime):** Canlı serviste hantal PyTorch / Torchvision bağımlılığı kaldırıldı; C++ NMS ve ONNX Runtime motoru ile sadece CPU üzerinde **~250 ms** çıkarım süresi elde edildi.
* **Üretim Tipi Docker Konteyneri:** `python:3.11-slim` tabanlı, katman önbellekli, non-root `appuser` güvenlikli ve otomatik canlılık denetimli (`HEALTHCHECK`) Docker paketi hazırlandı (**Sıkıştırılmış boyut: sadece 186 MB**).

---

## 📊 Model Başarımı ve Benchmark Sonuçları

Model Google Colab Tesla T4 GPU üzerinde 50 epoch eğitilmiş ve bağımsız test setinde değerlendirilmiştir:

| Metrik | Değer | Açıklama |
| :--- | :---: | :--- |
| **mAP@50** | **%98.0** | Genel nesne tespit başarımı |
| **mAP@50-95** | **%76.8** | Farklı IoU eşiklerindeki katı doğruluk |
| **Precision** | **%97.3** | Yanlış alarm (False Positive) düşüklüğü |
| **Recall** | **%95.2** | Kusuru kaçırmama (False Negative) başarısı |
| **Latency (CPU Tekil)** | **~330 ms** | Uçtan uca toplam SLA süresi |

### Sınıf Bazlı mAP@50 Dağılımı
* `copper`: **%99.5**
* `pin-hole`: **%99.0**
* `mousebite`: **%98.6**
* `spur`: **%98.1**
* `open`: **%98.0**
* `short`: **%95.2** *(En kritik sınıf)*

### 🚀 Eşzamanlı Yük Testi & Kapasite (Locust Benchmark)
Canlı Docker konteyneri üzerinde `tests/locustfile.py` ile eşzamanlı istek kapasitesi ve yanıt süreleri test edilmiştir:

| Parametre / Metrik | Değer | Açıklama |
| :--- | :---: | :--- |
| **Eşzamanlı Kullanıcı** | **20 Sanal Kullanıcı** | Ramp-up: 5 kullanıcı / sn |
| **Hata Oranı (Fail Rate)** | **%0.00** | Yük altında kesinti/hata yaşanmadı |
| **Throughput (RPS)** | **~2.1 req/s** | CPU tabanlı Docker ortamında eşzamanlı görsel işleme |
| **Health Check Yanıtı** | **~480 ms (Med)** | Canlılık kontrolü yanıt süresi |
| **Predict Yanıtı** | **3.0 - 9.8 s** | Ağır nesne tespiti eşzamanlı kuyruk süresi |

Yük testini yerel ortamınızda interaktif web arayüzü ile çalıştırmak için:
```powershell
locust -f tests/locustfile.py --host http://localhost:8000
# Web arayüzü: http://localhost:8089
```

---

## 📁 Proje Dizin Yapısı

```text
DataProcess/
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── requirements-dev.txt
├── requirements.txt
├── configs/
│   ├── config.yaml
│   └── data.yaml
├── data/
│   ├── interim.dvc
│   ├── processed.dvc
│   └── raw.dvc
├── ml/
│   ├── evaluate.py
│   ├── export_package.py
│   ├── prepare_data.py
│   ├── tracker.py
│   └── train.py
├── models/
│   ├── metadata.json
│   ├── models.dvc
│   └── pcb_defect_best.onnx
├── app/
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── main.py
│   │   ├── schemas/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── health.py
│   │       │   ├── monitoring.py
│   │       │   └── predict.py
│   │       └── router.py
│   ├── core/
│   │   ├── base_detector.py
│   │   ├── config.py
│   │   ├── dummy_detector.py
│   │   ├── onnx_detector.py
│   │   └── security.py
│   ├── services/
│   │   ├── feature_pipeline.py
│   │   ├── inference_service.py
│   │   └── model_loader.py
│   ├── ui/
│   │   └── app.py
│   └── utils/
│       ├── config_loader.py
│       ├── decorators.py
│       ├── helpers.py
│       ├── logger.py
│       └── validators.py
└── tests/
    ├── load/
    │   ├── benchmark_stats.csv
    │   └── locustfile.py
    ├── test_api.py
    └── test_config.py
```

---

## 🚀 Hızlı Başlangıç

### Seçenek 1: Docker ile Çalıştırma (Tavsiye Edilen)

```powershell
# 1. Docker imajını derleyin
docker build -t pcb-defect-api:latest .

# 2. Konteyneri arka planda başlatın
docker run -d -p 8000:8000 --name pcb-api pcb-defect-api:latest

# 3. Canlı Swagger arayüzünü tarayıcınızdan açın
# http://localhost:8000/docs
```

### Seçenek 2: Yerel Geliştirme Ortamı

```powershell
# Sanal ortamı oluşturun ve aktif edin
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Geliştirme paketlerini kurun
pip install -r requirements-dev.txt

# API sunucusunu başlatın
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit görsel stüdyosunu başlatın (ayrı bir terminalde)
streamlit run app/ui/app.py

# Yük testini başlatın (Locust)
locust -f tests/load/locustfile.py --host http://localhost:8000
```

---

## 🔌 API Kullanımı

### 1. Sağlık Kontrolü (`GET /health`)
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_name": "DeepPCB-YOLOv8",
  "model_version": "1.0.0",
  "device": "cpu",
  "classes": ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
}
```

### 2. Kusur Tespiti (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@data/raw/test/images/01_PCB__31.jpg"
```
```json
{
  "success": true,
  "total_defects": 9,
  "detections": [
    {
      "box": { "x1": 508.0, "y1": 378.0, "x2": 543.0, "y2": 427.0 },
      "confidence": 0.9212,
      "class_id": 4,
      "class_name": "copper"
    }
  ],
  "latency_ms": {
    "preprocess": 14.5,
    "inference": 257.8,
    "postprocess": 59.7,
    "total": 332.0
  },
  "filename": "01_PCB__31.jpg"
}
```

---

## 🧪 Test Takımı

Projedeki yapılandırma bütünlüğü ve API uç noktaları `pytest` ile otomatik test edilir:

```powershell
pytest tests/ -v
```
*(11 birim ve entegrasyon testinin tamamı yeşildir).*

---

## 📦 Veri ve Model Ağırlıklarını İndirme (DVC)

Eğitim verilerine ve model ağırlıklarına erişmek için DVC ile uzaktan çekebilirsiniz:

```powershell
dvc pull
```

---

## 👤 İletişim & Katkı

<!-- ✏️ DÜZENLEME ALANI: Kendi isim ve LinkedIn profilinizi yazın -->
* **Geliştirici:** [Adınız Soyadınız]
* **GitHub:** [@emrebyrktr](https://github.com/emrebyrktr)
* **LinkedIn:** [Profil Linkiniz](https://linkedin.com/in/profil-linkiniz)
* **Lisans:** MIT

