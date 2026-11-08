# Proje takip listesi

- [x] Temel pipeline ve servis hatalarını düzelt.
  Neden: Yanlış veya eksik modelle servis çalışmasın.
- [x] Kullanılmayan dosyaları ve importları temizle.
  Neden: Klasör yapısı sade ve anlaşılır olsun.
- [x] İlk imajı Docker Hub'a gönder: `emrebyrktr/pcb-defect-api:v1.0.0`.
  Neden: İmaj sürümlü olarak indirilebilsin. Bu adım kullanıcı tarafından elle yapıldı.
- [ ] Git yayınını hazırla: yeni dosyalar, lisans ve eski gizli ayar kontrolü.
  Neden: Repo eksiksiz ve paylaşılabilir olsun.
- [ ] README'deki çalıştırma adımlarını tek akışta doğrula.
  Neden: Repoyu indiren kişi projeyi kolayca çalıştırabilsin.
- [ ] Gerçek PCB görseliyle demo ve yanlış tahmin örnekleri ekle.
  Neden: Modelin ne yaptığını ve sınırlarını göstermek.

Kapsam: veri → eğitim → değerlendirme → API → Docker.
Deney takibi yerel MLflow, veri/model deposu DVC + Google Drive.

Ayrıntılı çalışma kaydı: `NOTES.md`.
