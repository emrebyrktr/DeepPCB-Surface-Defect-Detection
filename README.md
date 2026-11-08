# 🔬 PCB Kusur Tespiti ve MLOps

[![CI Pipeline](https://github.com/emrebyrktr/Proje1/actions/workflows/ci.yml/badge.svg)](https://github.com/emrebyrktr/Proje1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.31-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioned-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)

Bu proje, endüstriyel baskılı devre kartları (PCB) üzerindeki altı kusur sınıfını YOLO ile tespit eder. Veri hazırlama, eğitim, ONNX paketleme, değerlendirme ve FastAPI servisini bir araya getirir. Veri ve modeller DVC + Google Drive ile sürümlenir; değerlendirme sonuçları MLflow ile yerelde kaydedilir.

Odak: kolay çalıştırılabilen, test edilmiş ve modelin sınırlarını gösteren bir portfolyo projesi. Aşağıdaki benchmark değerleri kayıtlı önceki koşulara aittir; her kod değişikliğinde yeniden ölçülmüş sonuçlar değildir.

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

* **Veri hazırlama:** Ham etiketler YOLO formatına çevrilir; eksik görsel/etiket eşleşmesi ve geçersiz kutular hata verir. Hazır train/valid/test bölümleri korunur. Kümeler arasında birebir aynı görsel bulunmadığı kontrol edilmiştir; benzer kart grupları düzeyindeki sızıntı ayrıca incelenmelidir.
* **Veri ve model sürümleme:** DVC + Google Drive, büyük veri ve model dosyalarını saklar; Git'te `data/raw.dvc`, `dvc.yaml` ve `dvc.lock` tutulur.
* **Yerel deney takibi:** Değerlendirme metrikleri, model hash'i ve kaydedilmiş veri sürümü MLflow'a yazılır. Eğitim koşularının otomatik MLflow bağlantısı henüz tamamlanmamıştır.
* **Hafif Çıkarım (ONNX Runtime):** Canlı serviste hantal PyTorch / Torchvision bağımlılığı kaldırıldı; C++ NMS ve ONNX Runtime motoru ile sadece CPU üzerinde **~250 ms** çıkarım süresi elde edildi.
* **Docker ile çalıştırma:** `python:3.11-slim` tabanlı imaj, non-root `appuser` ve `HEALTHCHECK` içerir. Model ve metadata eksik veya geçersizse API başlangıcı durur.

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

Canlı Docker konteyneri üzerinde `tests/locustfile.py` ile 20 eşzamanlı sanal kullanıcı simüle edilerek sistemin stres ve yük altındaki davranışı ölçülmüştür:

| Endpoint | İstek Sayısı | Hata (%) | Medyan (ms) | %95 Dilimi (p95) | Ortalama (ms) | Min / Max (ms) | Anlık RPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GET /health`** | 19 | %0.00 | 2,300 | 5,500 | 2,860 | 1,114 / 5,501 | 0.2 req/s |
| **`POST /predict`** | 81 | %0.00 | 6,400 | 11,000 | 6,805 | 4,256 / 11,714 | 3.0 req/s |
| **Genel Toplam (Aggregated)** | **100** | **%0.00** | **6,300** | **11,000** | **6,055** | **1,114 / 11,714** | **3.2 req/s** |

#### 📌 Yük Testi Çıkarımları:
* Bu kısa koşuda 100 istekte hata görülmedi.
* Toplam trafik **3.2 istek/sn**, tahmin trafiği **3.0 istek/sn** olarak kaydedildi.
* Yük altında tahmin p95 süresi **11 saniye** oldu; tekil görseldeki yaklaşık 330 ms bu yük için garanti değildir.

Yük testini yerel ortamınızda interaktif web paneliyle çalıştırmak için:
```powershell
locust -f tests/locustfile.py --host http://localhost:8000
# Web Paneli: http://localhost:8089
```

---

## 📁 Proje Dizin Yapısı

```text
DataProcess/
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── configs/
│   ├── config.yaml
│   └── data.yaml
├── data/
│   └── raw.dvc
├── ml/
│   ├── evaluate.py
│   ├── export_package.py
│   ├── prepare_data.py
│   ├── tracker.py
│   └── train.py
├── models/
│   ├── metadata.json
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
│   │       │   └── predict.py
│   │       └── router.py
│   ├── core/
│   │   ├── base_detector.py
│   │   ├── config.py
│   │   ├── dummy_detector.py
│   │   └── onnx_detector.py
│   ├── services/
│   │   ├── feature_pipeline.py
│   │   ├── inference_service.py
│   │   └── model_loader.py
│   ├── ui/
│   │   └── app.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
├── scripts/
│   ├── check_distribution.py
│   ├── create_ci_model.py
│   └── smoke_api.py
└── tests/
    ├── locustfile.py
    ├── test_api.py
    ├── test_config.py
    ├── test_detector.py
    ├── test_export_package.py
    ├── test_tracking.py
    └── test_training_pipeline.py
```

---

## 🚀 Hızlı Başlangıç

### Seçenek 1: Docker ile Çalıştırma (Tavsiye Edilen)

Derlemeden önce `models/pcb_defect_best.onnx` ve `models/metadata.json` bulunmalıdır. Yeni bir checkout'ta DVC erişimini yapılandırıp `dvc pull` çalıştırın. CI fixture modeli geliştirme testleri içindir; PCB tahmini için eğitilmiş model paketini kullanın.

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
locust -f tests/locustfile.py --host http://localhost:8000
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
python -m pytest tests/ -v
```
Normal testler açıkça seçilmiş test doubles kullanır; GPU veya eğitim çalıştırılması gerektirmez. Geliştirme ve test araçları `pip install -r requirements-dev.txt` ile kurulur. CI yalnız servis ve test paketlerini kurar.

Mevcut ONNX modeli ve metadata ile gerçek servis başlangıcını ayrıca doğrulamak için:

```powershell
$env:RUN_MODEL_INTEGRATION = "1"
python -m pytest tests/test_detector.py -m integration -v
Remove-Item Env:RUN_MODEL_INTEGRATION
```

CI, boş ONNX yerine geçerli bir test modeliyle konteyner açılışını ve tahmini kontrol eder; bu modelin PCB başarımını ölçmez. Model dosyası veya metadata eksik/geçersizse API başlangıcı hata verir.

`dvc repro` eğitim başlatabilir. Kaynak kod ve komutlar değiştiğinde eski `dvc.lock` geçmiş çıktının kaydı olarak korunur; pipeline yeniden çalıştırılıp doğrulanmadan hash'ler elle değiştirilmez.

---

## 📦 Veri ve Model Ağırlıklarını İndirme (DVC)

Eğitim verilerine ve model ağırlıklarına erişmek için DVC ile uzaktan çekebilirsiniz:

Google Drive erişim ayarları Git'in yok saydığı `.dvc/config.local` içinde tutulur. Ortak `.dvc/config` yalnız remote tanımını içerir; kendi erişim bilgilerinizi yerel yapılandırmaya ekleyin.

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

