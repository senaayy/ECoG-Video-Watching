# ECoG Video Watching

Bu proje, video izletilirken kaydedilmiş elektrokortikografi (ECoG) sinyallerinden,
izlenen videodaki içerik kategorilerinin (ör. renkli/siyah-beyaz, şekil, yüz, vb.)
dekode edilmesini amaçlar. Epilepsi hastalarından intrakraniyal olarak toplanan ECoG
verisi, video uyaranlarıyla zamansal olarak hizalanır ve makine öğrenmesi yöntemleriyle
video içeriği sinyalden tahmin edilir.

## Amaç

- Ham ECoG (.mat) kayıtlarını işlenebilir hale getirmek
- Video içerik kategorilerini ECoG kayıtlarıyla zamansal olarak hizalamak/etiketlemek
- Sinyalden ayırt edici özellikler çıkarmak ve en bilgilendirici olanları seçmek
- Video içerik kategorisini sınıflandıran modeller eğitmek
- Model performansını değerlendirmek ve literatürdeki (SOTA) sonuçlarla karşılaştırmak
- Sonuçları görselleştirmek ve raporlamak

## İş Akışı

```
Ham ECoG (.mat) ──► Ön işleme ──► Etiketleme/hizalama ──► Özellik çıkarımı
   ──► Özellik seçimi ──► Sınıflandırma ──► Değerlendirme ──► SOTA karşılaştırma
   ──► Görselleştirme/rapor
```

## Proje Yapısı

```
data/
  raw/            Ham ECoG (.mat) dosyaları
  processed/      Ön işlenmiş / hizalanmış veri
notebooks/        Keşifsel analiz ve prototipleme not defterleri
scripts/          Veri indirme vb. yardımcı script'ler
src/
  data/           Veri yükleme / özetleme yardımcıları
  preprocessing/  Ham sinyal ön işleme (filtreleme, referanslama, artefakt temizleme)
  features/       Özellik çıkarımı ve özellik seçimi
  models/         Sınıflandırma modelleri
  evaluation/     Model değerlendirme ve SOTA karşılaştırma
reports/
  figures/        Üretilen grafikler ve görselleştirmeler
requirements.txt  Python bağımlılıkları
```

## Kurulum

Python 3.11 önerilir.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Veri Seti

"Faces vs Houses" ECoG veri setini (Miller, 2019, Stanford Digital Repository)
indirme adımları için bkz. `data/raw/README.md`.
