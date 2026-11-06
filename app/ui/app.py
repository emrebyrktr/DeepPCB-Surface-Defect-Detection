import io
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PCB Defect Detection Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Proje kök dizini referansı
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Sınıflara özel endüstriyel renk paleti (BGR formatı cv2 için, HEX formatı UI için)
CLASS_COLORS = {
    "open": {"bgr": (0, 140, 255), "hex": "#FF8C00"},       # Turuncu
    "short": {"bgr": (0, 0, 255), "hex": "#FF0000"},         # Kırmızı (Kritik)
    "mousebite": {"bgr": (255, 191, 0), "hex": "#00BFFF"},   # Camgöbeği
    "spur": {"bgr": (255, 105, 30), "hex": "#1E90FF"},       # Mavi
    "copper": {"bgr": (0, 215, 255), "hex": "#FFD700"},      # Altın Sarısı
    "pin-hole": {"bgr": (214, 112, 218), "hex": "#DA70D6"},  # Mor
}


def check_api_health(base_url: str):
    """API servisinin sağlık durumunu kontrol eder."""
    for path in ["/api/v1/health", "/health"]:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}{path}", timeout=2.0)
            if resp.status_code == 200:
                return True, resp.json()
        except Exception:
            continue
    return False, None


def annotate_image(image_bytes: bytes, detections: list, conf_thresh: float, line_thick: int):
    """Görsel üzerine tespit kutularını ve sınıf etiketlerini çizer."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filtered = []

    for det in detections:
        conf = det.get("confidence", 0.0)
        if conf < conf_thresh:
            continue
        filtered.append(det)

        box = det["box"]
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cls_name = det.get("class_name", "unknown")

        color_info = CLASS_COLORS.get(cls_name, {"bgr": (0, 255, 0)})
        bgr = color_info["bgr"]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, line_thick)

        # Label kutusu ve metin
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - h - 6)), (x1 + w + 4, y1), bgr, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, max(10, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), filtered


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    api_url = st.text_input("FastAPI Servis Adresi", value="http://localhost:8000")
    is_healthy, health_data = check_api_health(api_url)

    if is_healthy:
        st.success(f"🟢 API Aktif ({health_data.get('model_name')})")
        st.caption(f"Versiyon: **{health_data.get('model_version')}** | Cihaz: **{health_data.get('device', '').upper()}**")
    else:
        st.error("🔴 API Bağlantısı Başarısız!")
        st.info("Konteynerin çalıştığından emin olun:\n`docker run -d -p 8000:8000 pcb-defect-api`")

    st.markdown("---")
    st.subheader("🎯 Filtreler")
    conf_threshold = st.slider("Güven Eşiği (Confidence)", 0.10, 0.95, 0.30, 0.05)
    line_thickness = st.slider("Kutu Çizgi Kalınlığı", 1, 4, 2)

    st.markdown("---")
    st.subheader("🏷️ Kusur Lejantı")
    for name, cinfo in CLASS_COLORS.items():
        st.markdown(
            f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
            f"<div style='width:14px; height:14px; background-color:{cinfo['hex']}; margin-right:8px; border-radius:3px;'></div>"
            f"<span style='font-size:14px;'>{name}</span></div>",
            unsafe_allow_html=True,
        )


# --- ANA EKRAN ---
st.title("🔬 Industrial PCB Defect Detection Studio")
st.markdown(
    "Endüstriyel baskılı devre kartları (PCB) üzerindeki mikro üretim hatalarını "
    "**ONNX Runtime** ve **FastAPI** mimarisiyle milisaniye seviyesinde tespit eden görsel denetim paneli."
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "İncelemek İstediğiniz PCB Görselini Yükleyin",
        type=["jpg", "jpeg", "png"],
    )

with col_sample:
    st.write("&nbsp;")
    use_sample = st.button("🧪 Örnek Test Görseli Yükle", use_container_width=True)

image_bytes = None
image_name = None

sample_path = PROJECT_ROOT / "data/raw/test/images/01_PCB__31.jpg"

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_name = uploaded_file.name
elif use_sample and sample_path.exists():
    with open(sample_path, "rb") as f:
        image_bytes = f.read()
    image_name = sample_path.name


if image_bytes is not None:
    st.markdown("---")

    if not is_healthy:
        st.error("Görsel işlenemedi: API sunucusu kapalı. Lütfen Docker veya yerel API servisini başlatın.")
    else:
        with st.spinner("Model çıkarım yapıyor..."):
            try:
                files = {"file": (image_name, image_bytes, "image/jpeg")}
                endpoint = f"{api_url.rstrip('/')}/api/v1/predict"
                resp = requests.post(endpoint, files=files, timeout=10.0)
                if resp.status_code == 404:
                    # Geriye dönük uyumluluk
                    resp = requests.post(f"{api_url.rstrip('/')}/predict", files=files, timeout=10.0)

                if resp.status_code == 200:
                    data = resp.json()
                    detections = data.get("detections", [])
                    latency = data.get("latency_ms", {})

                    annotated_img, filtered_dets = annotate_image(
                        image_bytes, detections, conf_threshold, line_thickness
                    )

                    col_left, col_right = st.columns([1.2, 1])

                    with col_left:
                        tab_result, tab_orig = st.tabs(["🎯 Kusur Tespiti", "🖼️ Orijinal Görsel"])
                        with tab_result:
                            st.image(annotated_img, caption=f"Tespit Edilen: {len(filtered_dets)} Kusur", use_container_width=True)
                        with tab_orig:
                            st.image(Image.open(io.BytesIO(image_bytes)), caption="Ham Görsel", use_container_width=True)

                    with col_right:
                        st.subheader("⏱️ SLA & Gecikme Metrikleri")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Preprocess", f"{latency.get('preprocess', 0)} ms")
                        m2.metric("Inference", f"{latency.get('inference', 0)} ms")
                        m3.metric("Postprocess", f"{latency.get('postprocess', 0)} ms")
                        m4.metric("Toplam", f"{latency.get('total', 0)} ms")

                        st.subheader(f"📋 Tespit Listesi ({len(filtered_dets)} adet)")
                        if filtered_dets:
                            table_data = []
                            class_counts = {}
                            for d in filtered_dets:
                                cname = d["class_name"]
                                class_counts[cname] = class_counts.get(cname, 0) + 1
                                b = d["box"]
                                table_data.append({
                                    "Kusur Sınıfı": cname,
                                    "Güven (%)": f"{d['confidence'] * 100:.1f}%",
                                    "Konum [x1, y1, x2, y2]": f"[{b['x1']}, {b['y1']}, {b['x2']}, {b['y2']}]",
                                })

                            st.dataframe(table_data, use_container_width=True, hide_index=True)

                            st.subheader("📊 Sınıf Dağılımı")
                            st.bar_chart(class_counts)
                        else:
                            st.success("Seçilen güven eşiğinde bu kart üzerinde herhangi bir kusur bulunamadı.")

                        with st.expander("🔍 Ham JSON Yanıtını Görüntüle"):
                            st.json(data)
                else:
                    st.error(f"API Hatası ({resp.status_code}): {resp.text}")

            except Exception as e:
                st.error(f"İstek sırasında hata oluştu: {e}")

