# Deep PCB Defect Detection — Geliştirme Notları

## Proje Hakkında
- **Dataset:** Deep PCB (Kaggle) — 6 sınıflı PCB hata tespiti
- **Hedef:** YOLO tabanlı nesne tespiti modeli
- **Eğitim:** Google Colab
- **Versiyon takibi:** Git + DVC

---

## Sınıflar
`open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`

---

## Yapılanlar

### 1. Klasör Yapısı Oluşturuldu
```
DataProcess/
├── configs/        → YAML konfigürasyon dosyaları
├── data/
│   ├── raw/        → Orijinal Kaggle verisi (Git'e gitmez, DVC ile takip)
│   ├── interim/    → Ara işlem çıktıları (label dönüşümü vs.)
│   └── processed/  → Modele hazır final veri
├── ml/             → Eğitim pipeline kodu (prepare, train, evaluate, export)
├── models/         → Eğitilmiş model dosyaları
├── notebooks/      → EDA ve deneyler
├── scripts/        → Tek seferlik veri hazırlama scriptleri
├── src/            → API ve uygulama kodu
│   ├── api/
│   ├── core/
│   └── utils/
└── tests/          → Birim testler
```

### 2. Sanal Ortam Oluşturuldu
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 3. Bağımlılıklar Tanımlandı
- `requirements.txt` → Prod / Colab için (ultralytics, torch, torchvision, opencv, numpy, pyyaml)
- `requirements-dev.txt` → Local geliştirme için (+ jupyter, matplotlib)
- Colab'da kurulum yapıldıktan sonra sürümler `pip freeze` ile sabitlenecek

### 4. .gitignore Güncellendi
Şunlar Git'e gitmez:
- `data/raw/`, `data/interim/`, `data/processed/`
- `models/`
- `Data.zip`
- `.venv/`
- `__pycache__/`, `.env`, Jupyter checkpoints

### 5. Dataset İncelendi
- **Format:** `train/images` + `train/labels`, `valid/`, `test/`
- **Görüntü boyutu:** 640×640 JPG (YOLO için ideal, resize gerekmez)
- **Label formatı:** `x_min y_min x_max y_max class_id` (Pascal VOC, 1-indexed)
- **Split:** train 1200 / valid 150 / test 150 görüntü

### 6. Label Format Dönüşümü — `scripts/convert_labels.py`
Pascal VOC pixel formatından YOLO normalized formatına çevrildi:
```
# Önce (Pascal VOC):   x_min y_min x_max y_max class_id  (1-indexed)
# Sonra (YOLO):        class_id x_center y_center width height  (normalized 0-1, 0-indexed)
```
**Formül:**
```python
x_center = (x_min + x_max) / 2 / 640
y_center = (y_min + y_max) / 2 / 640
width    = (x_max - x_min) / 640
height   = (y_max - y_min) / 640
class_id = original_class_id - 1
```
Çıktı: `data/interim/{split}/labels/`

### 7. Sınıf Dağılımı Analizi — `scripts/check_distribution.py`
`data/interim/` altındaki label dosyaları taranarak her sınıfın kaç örneği olduğu sayılır.
sonuç olarak: 
--- TRAIN ---
  open        : 1562
  short       : 1169
  mousebite   : 1582
  spur        : 1307
  copper      : 1186
  pin-hole    : 1218

--- VALID ---
  open        : 189
  short       : 167
  mousebite   : 189
  spur        : 165
  copper      : 148
  pin-hole    : 147

--- TEST ---
  open        : 191
  short       : 170
  mousebite   : 194
  spur        : 153
  copper      : 140
  pin-hole    : 136

--- TOPLAM ---
  open        : 1942
  short       : 1506
  mousebite   : 1965
  spur        : 1625
  copper      : 1474
  pin-hole    : 1501
Bulunmaktadır.



---


---

## Notlar
- `scripts/` → tek seferlik veri hazırlama (elle tetiklenir, diske yazar)
- `ml/` → her eğitimde çalışan pipeline kodu (bellekte işler, modele verir)
- `data/interim/` → ara adımlar (bir şey patlarsa başa dönmeden buradan devam)
- `data/processed/` → modele girecek final, temiz veri

