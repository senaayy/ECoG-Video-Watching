# Sunum Özeti — ECoG Faces vs Houses

*8-10 slaytlık sunum taslağı. Her bölüm bir slayt. Kaynak: `notebooks/08_final_report.ipynb`.*

---

## Slayt 1 — Problem

**Soru:** İnsan beyni, subdural elektrotlarla (ECoG) doğrudan kayıt
edilen kortikal aktiviteden, bir kişinin o anda **yüz mü yoksa ev mi**
gördüğünü ne kadar güvenilir şekilde ayırt edebiliriz — ve bu ayrım
beynin hangi bölgesinden geliyor?

- Literatürde "fusiform face area (FFA)" adı verilen bölgenin yüz
  algısında özel bir rolü olduğu biliniyor (fMRI ve önceki ECoG
  çalışmalarından) — bu projede bunu **kendi ham veride, uçtan uca bir
  pipeline ile** doğrulamayı hedefledik.
- İkincil soru: klasik makine öğrenmesi mi, yoksa derin öğrenme mi bu
  görevde daha iyi çalışıyor — özellikle **sinyalin zayıf olduğu**
  hastalarda?

---

## Slayt 2 — Veri Seti

- **Miller (2019)**, Stanford Digital Repository, `faces_basic` görevi
  (Miller ve ark. 2016, *PLoS Comp. Biol.*), CC BY-SA 4.0.
- **14 hasta**, her biri epilepsi cerrahisi amaçlı kendi elektrot
  montajıyla (grid/strip) — kanal sayısı **31 ile 102 arası** değişiyor.
- Görev: 400ms süreyle art arda sunulan **yüz / ev** görselleri (150+150
  uyaran/hasta), 1000 Hz örnekleme.
- Elektrot koordinatları + el-ile-etiketlenmiş anatomik bölge kodları
  (`locs/*_xslocs.mat`) mevcut — bölge bazlı analiz için kritik.

---

## Slayt 3 — Yöntem: Pipeline

```
.mat dosyaları → Ön işleme (notch+CAR+bandpass) → Etiketleme
   (face/house/baseline) → Özellik çıkarımı (high-gamma + bant güçleri,
   yalnızca temporal/fusiform/occipital kanallar) → Sınıflandırma
   (LDA/SVM/RF, zamansal-blok GroupKFold) → Değerlendirme (confusion
   matrix, feature importance, 3D görselleştirme) → Derin öğrenme
   (EEGNet) → SOTA karşılaştırma
```

- Her aşama ayrı bir notebook'ta (`01`-`07`) belgelendi ve çalıştırıldı.
- Şebeke gürültüsü (50/60 Hz) her hastada **veriden doğrulandı**,
  varsayılmadı (13/14 hastada 60 Hz, 1 hastada 50 Hz).

---

## Slayt 4 — Bilimsel Bulgu: FFA Doğrulaması

Üç **bağımsız** yöntem aynı sonuca işaret ediyor:

1. **Tek-kanal t-testi** (high-gamma): en güçlü ayrımlar fusiform
   gyrus'ta (`ap` hastasında t=10.3, p=1.3×10⁻²¹)
2. **Random Forest feature importance**: 14 hastanın **13'ünde** modelin
   en önemli bulduğu kanalın bölgesi t-testininkiyle birebir aynı
   (Spearman ρ=0.629)
3. **3D kortikal görselleştirme**: en yüksek önem skorlu elektrotlar
   görsel olarak fusiform gyrus'ta kümeleniyor

→ **FFA bulgusu bu ECoG veri setinde sağlam şekilde yeniden üretildi.**

---

## Slayt 5 — Sınıflandırma Sonuçları

| Model | Ort. Accuracy | Ort. Macro-F1 | Ort. ROC-AUC |
|---|---|---|---|
| **Random Forest** | **0.876** | **0.876** | **0.924** |
| LDA | 0.856 | 0.854 | 0.915 |
| Linear SVM | 0.830 | 0.828 | 0.888 |

- 14 hasta × 3 model, **tamamında p<0.01** (200-permütasyonlu test)
- **Sinyal gücü ↔ accuracy korelasyonu: r=0.895** — güçlü sinyal
  kategorisindeki hastalar ort. %97 accuracy, zayıf kategori %76
- En çok karışan hasta `fp` (fusiform elektrotu yok, ~%40 karışma oranı)

---

## Slayt 6 — Derin Öğrenme Bulgusu (n=5 uyarısı ile)

**⚠️ Yalnızca 5 temsili hastada test edildi — 14 hastanın tamamında değil.**

| Hasta | Kategori | Baseline | EEGNet | Fark |
|---|---|---|---|---|
| `fp` | zayıf | 0.663 | **0.733** | **+0.070** |
| `rr` | zayıf | 0.759 | **0.860** | **+0.101** |
| `aa` | orta | 0.758 | 0.757 | ~0 |
| `ap` | güçlü | 0.868 | 0.833 | -0.035 |
| `zt` | güçlü | 0.994 | 0.987 | -0.007 |

- **Beklenmedik bulgu**: EEGNet zayıf-sinyal hastalarında baseline'ları
  geçiyor, güçlü-sinyal hastalarında hafifçe geride kalıyor
- Olası neden: ham zaman-serisi, sabit 5 spektral banda sığmayan
  faz/ERP bilgisi taşıyor
- **Bu bir hipotez — 14 hastanın tamamında doğrulanmadı**

---

## Slayt 7 — SOTA Karşılaştırma (Dürüst Çerçeve)

- **EEGNet (Lawhern 2018)**: EEG üzerinde P300/ERN/MRCP/SMR
  paradigmaları — ECoG değil, face/house değil → yalnızca **mimari**
  alındı, **sayısal kıyas yapılmadı**
- **HTNet (Peterson 2021)**: ECoG'de kol hareketi decoding'i,
  hastalar-arası genelleme için elektrot-projeksiyon katmanı → yalnızca
  **metodolojik fikir** referans alındı (uygulanmadı)
- **Sonuç:** doğrudan sayısal "SOTA'yı geçtik" iddiası **yok** — farklı
  görev/modalite/genelleme rejimi nedeniyle kıyaslanamaz

---

## Slayt 8 — Sınırlamalar

1. **Within-subject** — hastalar-arası genelleme hiç denenmedi
2. **`fp` hastası** — fusiform elektrotu yok, FFA kanıtı üretmiyor
3. **Yaklaşık kortikal mesh** — gerçek pial yüzey değil (Otsu+marching cubes)
4. **5-hasta derin öğrenme örneklemi** — istatistiksel güç sınırlı
5. **400ms kısa pencere** — düşük frekans bantlarında kaba spektral çözünürlük
6. **EEGNet için permütasyon testi yok** — anlamlılık doğrudan test edilmedi

---

## Slayt 9 — Sonuç

- Ham `.mat` dosyalarından başlayan **uçtan uca, tekrarlanabilir bir
  pipeline** kuruldu (14 hastanın tamamı için otomatik çalışıyor)
- **FFA bulgusu üç bağımsız yöntemle doğrulandı**
- Klasik ML (özellikle RF) **güçlü ve istatistiksel olarak sağlam**
  within-subject performans veriyor
- Derin öğrenme, **zayıf sinyalde umut verici ama doğrulanmamış** bir
  avantaj gösteriyor — gelecek çalışma için açık bir soru
- Tüm bulgular, sınırlamalarıyla birlikte şeffaf şekilde raporlandı

---

## Slayt 10 — Gelecek Çalışma (opsiyonel)

- EEGNet'i **14 hastanın tamamında** çalıştırıp n=5 bulgusunu doğrula/çürüt
- HTNet'in elektrot-projeksiyon katmanını uygulayıp **hastalar-arası
  genelleme** dene
- `jt` hastasındaki yazar-etiketli `bad_chans` listesini
  `compute_bad_channels()` ile birleştirerek kanal kalite tespitini iyileştir
- Daha uzun event pencereleri (400ms sonrası, geç bileşenler) veya farklı
  zaman-frekans yöntemleriyle (wavelet) düşük frekans bantlarını daha iyi çöz
