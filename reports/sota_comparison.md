# SOTA Karşılaştırması — Kendi Sonuçlarımız vs Literatür

**Önemli metodolojik uyarı:** Bu belgedeki EEGNet ve HTNet görev tanımları
web araması ile doğrulandı (kaynaklar bölümüne bakınız). **Her iki makale
de bizim kullandığımız face/house algılama görevini veya bu Miller (2019)
`faces_basic` veri setini kullanmıyor** — biri EEG tabanlı BCI
paradigmaları, diğeri ECoG tabanlı kol hareketi decoding'i üzerine kurulu.
Bu nedenle aşağıda **doğrudan bir accuracy/AUC sayısı karşılaştırması
yapılmıyor** — yalnızca mimari/yöntemsel yaklaşım referans alınıyor ve
görev/veri seti farkları netleştiriliyor. Bu bir "SOTA yenildi/yenilmedi"
iddiası değildir.

## 1. Kendi sonuçlarımız (özet)

### Klasik baseline'lar (14 hasta, `reports/baseline_results.csv`)

| Model | Ortalama Accuracy | Ortalama Macro-F1 | Ortalama ROC-AUC |
|---|---|---|---|
| Random Forest | 0.876 | 0.876 | 0.924 |
| LDA | 0.856 | 0.854 | 0.915 |
| Linear SVM | 0.830 | 0.828 | 0.888 |

Tüm hasta × model kombinasyonlarında permütasyon testi p<0.01 (n_permutations=200).

### EEGNet (5 temsili hasta, `reports/deep_results.csv`)

| Hasta | Sinyal kategorisi | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| `zt` | güçlü | 0.987 | 0.987 | 0.998 |
| `rr` | zayıf | 0.860 | 0.860 | 0.905 |
| `ap` | güçlü | 0.833 | 0.833 | 0.919 |
| `aa` | orta | 0.757 | 0.756 | 0.840 |
| `fp` | zayıf | 0.733 | 0.732 | 0.790 |

EEGNet zayıf-sinyal hastalarında (`fp`, `rr`) en iyi klasik baseline'ı
sırasıyla **+0.070** ve **+0.101** accuracy puanıyla geçiyor; güçlü-sinyal
hastalarında (`ap`, `zt`) ise **-0.035** ve **-0.007** ile hafifçe geride
kalıyor (bkz. `notebooks/07_deep_learning.ipynb`).

## 2. Literatürle karşılaştırma

### EEGNet (Lawhern ve ark. 2018, *Journal of Neural Engineering*)

- **Doğrulanmış görev kapsamı**: orijinal makale EEGNet'i **dört EEG
  paradigmasında** değerlendiriyor — **P300** (görsel uyaran/dikkat),
  **ERN** (error-related negativity, hata-ilişkili negatiflik), **MRCP**
  (movement-related cortical potential, hareket-ilişkili kortikal
  potansiyel) ve **SMR** (sensorimotor rhythm, motor imgeleme). **SSVEP
  kullanılmamış.** Bunların hepsi **yüzeysel EEG** üzerinde; **ECoG değil**,
  ve **face/house görsel kategori ayrımı gibi bir paradigma yok**.
- Bu nedenle **EEGNet makalesinden bize doğrudan aktarılabilecek bir
  accuracy/AUC sayısı yok** — kayıt yöntemi (EEG vs ECoG), sinyal-gürültü
  oranı ve görev tipi (dikkat/hata/motor vs pasif görsel kategori
  algılama) o kadar farklı ki sayısal bir kıyas yanıltıcı olur.
- Bizim burada aldığımız, **EEGNet makalesinden yalnızca mimari
  fikirdir**: derinlemesine (depthwise) ve ayrılabilir (separable)
  konvolüsyonlarla EEG/ECoG'e özgü zamansal-uzamsal filtreleme yapan
  kompakt bir CNN — F1/D/F2 hiperparametreleri ve blok yapısı birebir
  orijinal makaleden alındı (bkz. `src/models/train_deep.py`), ancak
  girdi verisi, görev ve değerlendirme protokolü tamamen farklı.

### HTNet (Peterson ve ark. 2021, *Journal of Neural Engineering*, "Generalized neural decoders for transfer learning across participants and recording modalities")

- **Doğrulanmış görev kapsamı**: HTNet'in asıl benchmark görevi **kol
  hareketi decoding'i (move vs rest)** — face/house gibi bir görsel
  kategori ayrımı **değil**. Model, 12 katılımcıdan 11'inin ECoG
  verisiyle eğitilip görülmemiş katılımcılar (ECoG veya EEG) üzerinde
  test ediliyor; ardından hedef katılımcıya az miktarda veriyle
  (50 ECoG / 20 EEG event) ince ayar (fine-tuning) yapılıyor.
- HTNet'in temel yöntemsel katkısı iki parça: (a) veri-güdümlü
  frekanslarda spektral gücü hesaplayan bir Hilbert-dönüşüm katmanı,
  (b) **elektrot-seviyesi veriyi önceden tanımlanmış beyin bölgelerine
  projekte eden bir katman** — bu, ECoG'de elektrot yerleşiminin
  hastadan hastaya standart olmamasından kaynaklanan **hastalar-arası
  genelleme** sorununu çözmek için tasarlanmış.
- **Bizim çalışmamızda hastalar-arası genelleme (cross-subject transfer)
  yok** — her hasta ayrı ayrı (within-subject) modellendi, elektrot
  projeksiyonu gibi bir mekanizma kullanılmadı. Bu nedenle **HTNet
  makalesinden bize doğrudan aktarılabilecek bir accuracy sayısı da
  yok**; alınan yalnızca **metodolojik referans**: elektrot yerleşiminin
  hastadan hastaya değiştiği bu veri setinde (bkz. 02_preprocessing.ipynb
  Bölüm 9 — kanal sayısı 31-102 arası değişiyor), ileride hastalar-arası
  bir model denenecek olsaydı, HTNet'in elektrot-projeksiyon yaklaşımı
  doğal bir sonraki adım olurdu. Bu proje kapsamında uygulanmadı.

## 3. Metodolojik farklar — neden doğrudan kıyaslanamaz

| Boyut | Bu çalışma | EEGNet (Lawhern 2018) | HTNet (Peterson 2021) |
|---|---|---|---|
| Kayıt yöntemi | ECoG (subdural) | Yüzeysel EEG | ECoG (+ EEG'e transfer) |
| Görev | Face vs house algılama (pasif görsel kategori) | P300 / ERN / MRCP / SMR (dikkat, hata, motor) | Kol hareketi vs istirahat (motor) |
| Genelleme rejimi | **Within-subject** (hasta-bazlı, ayrı ayrı) | Paradigmaya göre değişir (çoğunlukla within-subject) | **Cross-subject** transfer + az veriyle ince ayar |
| Elektrot yerleşimi sorunu | Var (7-102 kanal, hastadan hastaya farklı montaj) — bu projede çözülmedi | Yok (standart EEG başlığı) | Var — elektrot-projeksiyon katmanıyla açıkça çözülüyor |
| Bizden alınan | — | Yalnızca mimari (depthwise/separable conv bloklar) | Yalnızca metodolojik fikir (projeksiyon katmanı), uygulanmadı |

**Sonuç:** Görev, veri modalitesi ve genelleme rejimi bu kadar farklı
olduğundan, **hiçbir sayısal accuracy/AUC karşılaştırması bu belgede
yapılmıyor** — yapılması bilimsel olarak yanıltıcı olurdu. Bu iki makale
buraya yalnızca **yöntemsel emsal** olarak alındı: EEGNet'in kompakt
CNN mimarisi doğrudan uygulandı (Bölüm 1), HTNet'in elektrot-projeksiyon
fikri ise bu projenin **doğal bir sonraki adımı** olarak not edildi
(hastalar-arası genelleme denenmek istenirse).

## 4. Kendi içimizdeki en dikkat çekici bulgu

Klasik baseline'lar (spektral bant-güç özellikleri) ile EEGNet (ham
zaman-serisi) arasındaki karşılaştırma, kendi veri setimiz içinde
literatürden bağımsız olarak dikkat çekici bir sonuç veriyor: EEGNet,
zayıf-sinyal hastalarında (`fp`, `rr`) klasik modelleri **geçiyor** (bkz.
`notebooks/07_deep_learning.ipynb`, Bölüm 4) — bu, "az veri + zayıf sinyal
→ derin öğrenme dezavantajlı" naif beklentisinin tersidir ve muhtemelen
ham sinyalin, elle tasarlanmış 5 sabit spektral banda sığmayan ek
zamansal/fazsal bilgi taşımasından kaynaklanıyor.

---

## Kaynaklar (web araması ile doğrulandı)

- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural
  network for EEG-based brain-computer interfaces. *Journal of Neural
  Engineering*. [IOPscience](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) ·
  [arXiv](https://arxiv.org/abs/1611.08024)
- Peterson, S. M., Steine-Hanson, Z., Davis, N., Rao, R. P. N., & Brunton,
  B. W. (2021). Generalized neural decoders for transfer learning across
  participants and recording modalities. *Journal of Neural Engineering*.
  [IOPscience](https://iopscience.iop.org/article/10.1088/1741-2552/abda0b) ·
  [GitHub (HTNet_generalized_decoding)](https://github.com/BruntonUWBio/HTNet_generalized_decoding)

*Diğer kaynak dosyalar: `src/models/train_deep.py`,
`notebooks/07_deep_learning.ipynb`, `reports/deep_results.csv`,
`reports/baseline_results.csv`.*
