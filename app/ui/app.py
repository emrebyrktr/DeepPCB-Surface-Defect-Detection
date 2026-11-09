import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PCB Kusur Tespiti")
st.title("PCB Kusur Tespiti")

uploaded_file = st.file_uploader("Görsel yükle", type=["jpg", "jpeg", "png"])

if st.button("Tahmin et", disabled=uploaded_file is None):
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        st.write("Tahmin alınamadı. API bağlantısını ve yüklediğin dosyayı kontrol et.")
    else:
        st.json(result)
