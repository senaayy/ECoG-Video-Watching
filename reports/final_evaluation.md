# Final Değerlendirme — ECoG Faces vs Houses (Miller 2019, `faces_basic`)

Bu rapor, `notebooks/06_evaluation.ipynb`'deki model değerlendirme
analizlerinin özetidir; önceki aşamalardaki bulgular (EDA, ön işleme,
özellik çıkarımı, sınıflandırma) bağlamında sunulmuştur.

## 1. En iyi model

`reports/baseline_results.csv`'e (14 hasta × 3 model × 200 permütasyon,
`GroupKFold(n_splits=5)`) göre **Random Forest** en iyi ortalama performansı
veriyor:

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| **RandomForest** | **0.876** | **0.876** | **0.924** |
| LDA | 0.856 | 0.854 | 0.915 |
| Linear SVM | 0.830 | 0.828 | 0.888 |

Random Forest, 14 hastanın 10'unda en yüksek accuracy'yi sağlıyor (LDA 3,
Linear SVM 1 hastada üstün). Tüm hasta × model kombinasyonlarında
permütasyon testi p<0.01 — face/house ayrımı şans düzeyinin çok üzerinde ve
istatistiksel olarak sağlam.

## 2. Out-of-fold confusion matrix'ler (14 hasta)

RandomForest ile, her hasta için zamansal-blok GroupKFold ile üretilen
out-of-fold tahminlerden normalize confusion matrix'ler hesaplandı
(bkz. `reports/figures/confusion_matrix_grid.png`).

- 14 hastanın çoğunda hem face hem house sınıfı için **>%80 doğru
  sınıflandırma** oranı var.
- En çok karışma **`fp`** hastasında (face→house %39, house→face %40 —
  neredeyse şans düzeyi) ve `rr` / `aa` / `rn` hastalarında (~%18-25)
  gözlendi. Bu hastalar, 04_feature_extraction.ipynb Bölüm 5'teki
  "orta/zayıf" sinyal kategorisiyle tutarlı.

## 3. Feature importance — anatomik eşleme ve t-test karşılaştırması

Her hastada Random Forest'ın kanal bazlı feature importance'ı, aynı
hastanın face-vs-house high-gamma t-test sonucuyla (04_feature_extraction.ipynb
Bölüm 5) karşılaştırıldı:

- Ortalama Spearman sıra korelasyonu: **0.629** (orta-güçlü pozitif ilişki)
- Top-3 kanalda **8/14 hastada tam örtüşme**, hiçbir hastada sıfır örtüşme yok
- **En yüksek önem skoruna sahip kanalın anatomik bölgesi, 14 hastanın
  13'ünde model (RF) ve t-test arasında birebir aynı** (yalnızca `rr`
  hastasında uyuşmazlık: fusiform vs temporal — bu hasta zaten en zayıf
  sinyale ve en az event sayısına sahip)

**Sonuç:** Random Forest, kanallar arası karmaşık/gizli etkileşimlerden
ziyade büyük ölçüde **aynı tek-kanal face-selektif sinyali** öğreniyor —
model "farklı bir şey öğrendiğine" dair güçlü bir kanıt yok, aksine
univariate ve multivariate yöntemler birbirini doğruluyor.

## 4. 3D kortikal yüzey görselleştirmesi (`ap`, `zt`)

`data/raw/faces_basic/brains/<hasta>_mri.nii` hacminden Otsu eşiğine dayalı
marching-cubes ile yaklaşık bir yüzey çıkarılıp elektrotlar feature
importance renk kodlamasıyla bindirildi
(`reports/figures/electrodes_on_cortex_ap_zt.png`).

**Önemli metodolojik not:** Bu veri setinde önceden hesaplanmış bir
pial/kortikal yüzey (FreeSurfer/SPM segmentasyonu) **bulunmuyor** —
yalnızca ham T1 `.nii` hacmi var (`ctmr_gauss_plot.m` gibi MATLAB
fonksiyonları normalde böyle bir `cortex.vert`/`cortex.tri` yapısı
bekliyor, ancak proje için hazır sağlanmamış; ayrıca `ctmr.zip` diye bir
arşiv bulunamadı — ilgili dizin `data/raw/ctmr/` zaten açılmış haldeydi ve
yalnızca MATLAB script'leri içeriyordu, önceden üretilmiş bir mesh
dosyası yoktu). Bu nedenle üretilen yüzey **gerçek bir kortikal (pial)
yüzey değil**, Otsu eşiğine dayalı yaklaşık bir doku zarfı (kafa
derisi/beyin dış sınırına yakın) — yalnızca elektrotların göreli 3D
konumunu görselleştirmek için kullanıldı. Elektrot koordinatları
(`locs/<hasta>_xslocs.mat`) ile `.nii` affine dönüşümü uygulanmış mesh
köşe noktaları aynı world (mm) koordinat uzayında doğrulandı.

Görsel olarak, `ap` ve `zt` hastalarında en yüksek importance'a sahip
elektrotlar fusiform gyrus bölgesinde kümeleniyor — sayısal analizlerle
tutarlı bir doğrulama.

## 5. Genel sonuç

Üç bağımsız yöntem — **(1) tek-kanal t-testi, (2) çok-kanallı Random
Forest feature importance, (3) 3D anatomik görselleştirme** — birbirini
doğruluyor: fusiform gyrus'ta elektrotu olan hastalarda (`ap`, `ca`, `ja`,
`mv`, `zt`) face-selektivite tutarlı şekilde en güçlü ve en çok fusiform
kanallara atfediliyor. Bu, literatürdeki **fusiform face area (FFA)**
bulgusunun bu ECoG veri setinde sağlam bir şekilde yeniden üretildiğini
gösteriyor.

`fp` hastası (fusiform elektrotu yok) her aşamada ayrı not düşülerek dahil
edildi; bu hastanın düşük performansı (accuracy ~%61-66, en yüksek
confusion oranı) ve occipital-kaynaklı ayrımı, FFA'ya özgü bir kanıt
olarak yorumlanmadı — beklenen tutarsızlık (fusiform yoksa FFA sinyali de
yok) doğrulandı.

---

*Kaynak dosyalar: `src/evaluation/evaluate.py`,
`notebooks/06_evaluation.ipynb`, `reports/baseline_results.csv`,
`reports/figures/confusion_matrix_grid.png`,
`reports/figures/electrodes_on_cortex_ap_zt.png`.*
