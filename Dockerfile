FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Bagimliliklarin kurulmasi (katman onbellegi icin koddan once)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model paketi, konfigurasyon ve servis kodlari
COPY configs/ configs/
COPY models/pcb_defect_best.onnx models/metadata.json models/
COPY src/ src/

# Guvenlik icin standart calisma kullanicisi
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

