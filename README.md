# ECoG Faces vs Houses — Within-Subject Decoding & FFA Doğrulaması

Bu proje, epilepsi hastalarından intrakraniyal olarak toplanan
elektrokortikografi (ECoG) verisinden, kişinin o anda **yüz mü yoksa ev mi**
gördüğünü sinyalden dekode etmeyi ve bu ayrımın beynin hangi anatomik
bölgesinden geldiğini (**fusiform face area** hipotezi) doğrulamayı
amaçlar. Ham `.mat` kayıtlarından başlayan uçtan uca bir pipeline ile;
ön işleme, etiketleme, özellik çıkarımı, klasik makine öğrenmesi
(LDA/SVM/Random Forest) ve derin öğrenme (EEGNet) modelleri eğitilip
değerlendirilir, sonuçlar literatürle (EEGNet, HTNet) karşılaştırılır.

**Veri seti:** Miller (2019), *A library of human electrocorticographic
data and analyses*, Stanford Digital Repository — `faces_basic` görevi
(14 hasta, 400ms süreyle sunulan yüz/ev görselleri, 1000 Hz).

## Öne Çıkan Bulgular

- **FFA doğrulaması**: tek-kanal t-testi, çok-kanallı Random Forest
  feature importance ve 3D kortikal görselleştirme — üç bağımsız yöntem,
  fusiform gyrus'ta elektrotu olan hastalarda tutarlı şekilde face-selektif
  bir sinyale işaret ediyor.
- **Sınıflandırma**: 14 hasta × 3 model (within-subject, zamansal-blok
  `GroupKFold`), Random Forest ortalama %87.6 accuracy; sinyal gücü ile
  doğruluk arasında r=0.895 korelasyon.
- **Derin öğrenme (EEGNet)**: 5 temsili hastada, zayıf-sinyal
  hastalarında klasik modelleri geçtiği gözlemlendi (n=5 uyarısıyla,
  doğrulanmamış bir bulgu).

Tüm detaylar için: [`notebooks/08_final_report.ipynb`](notebooks/08_final_report.ipynb)
(sentez rapor) ve [`reports/presentation_summary.md`](reports/presentation_summary.md)
(sunum özeti).

## İş Akışı

```
Ham ECoG (.mat) ──► Ön işleme (notch+CAR+bandpass) ──► Etiketleme (face/house/baseline)
   ──► Özellik çıkarımı (high-gamma + bant güçleri, ROI kanalları) ──► Sınıflandırma
   (LDA/SVM/RF + EEGNet) ──► Değerlendirme (confusion matrix, feature importance, 3D)
   ──► SOTA karşılaştırma ──► Final rapor
```

## Notebook'lar

| Notebook | İçerik |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Keşifsel veri analizi: değişkenler, örnekleme hızı, kanal istatistikleri, stim şeması, elektrot koordinatları |
| [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) | Şebeke gürültüsü tespiti, notch/CAR/bandpass, bozuk kanal doğrulaması, anatomik etiketleme, toplu işleme (14 hasta) |
| [`03_label_alignment.ipynb`](notebooks/03_label_alignment.ipynb) | Stim şeması doğrulama, onset/event çıkarımı, sınıf dağılımı |
| [`04_feature_extraction.ipynb`](notebooks/04_feature_extraction.ipynb) | High-gamma + klasik bant güçleri, FFA t-testi (14 hasta) |
| [`05_classification.ipynb`](notebooks/05_classification.ipynb) | LDA/SVM/RF, GroupKFold, permütasyon testi, sinyal gücü karşılaştırması |
| [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Confusion matrix'ler, feature importance vs t-test, 3D kortikal görselleştirme |
| [`07_deep_learning.ipynb`](notebooks/07_deep_learning.ipynb) | EEGNet (PyTorch), 5 temsili hasta, baseline karşılaştırması |
| [`08_final_report.ipynb`](notebooks/08_final_report.ipynb) | Tüm bulguların tek akışta sentezi |

## Proje Yapısı

```
data/
  raw/            Ham ECoG (.mat) dosyaları, MRI (.nii), elektrot koordinatları (git'e dahil değil)
  processed/      Ön işlenmiş sinyal (npz), etiketler (parquet), özellikler (npz) (git'e dahil değil)
notebooks/        01-08: EDA → ön işleme → etiketleme → özellik → sınıflandırma → değerlendirme → derin öğrenme → final rapor
scripts/          Veri indirme yardımcı script'leri
src/
  data/           Veri yükleme / özetleme yardımcıları
  preprocessing/  Ön işleme (notch/CAR/bandpass), anatomik etiketleme, event/etiket hizalama
  features/       Spektral özellik çıkarımı (high-gamma + klasik bantlar)
  models/         Klasik baseline'lar (LDA/SVM/RF) ve derin öğrenme (EEGNet)
  evaluation/     Model değerlendirme (confusion matrix, feature importance, 3D görselleştirme)
reports/
  figures/                    Üretilen grafikler (git'e dahil değil)
  baseline_results.csv        Klasik model sonuçları (14 hasta × 3 model)
  deep_results.csv            EEGNet sonuçları (5 temsili hasta)
  final_evaluation.md         Değerlendirme özeti
  sota_comparison.md          Literatürle (EEGNet, HTNet) karşılaştırma
  presentation_summary.md     10 slaytlık sunum özeti
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

"Faces vs Houses" ECoG veri setini (Miller, 2019, Stanford Digital
Repository) indirme adımları için bkz. `data/raw/README.md`.

## Etiketler / Topics

`ecog` · `intracranial-eeg` · `brain-computer-interface` · `neural-decoding`
· `eegnet` · `deep-learning` · `pytorch` · `signal-processing`
· `neuroscience` · `fusiform-face-area` · `face-perception`
· `within-subject-classification` · `scikit-learn` · `python`
