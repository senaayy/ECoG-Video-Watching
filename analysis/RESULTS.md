# Ek analiz: sinyal hangi bölgede yoğunlaşıyor?

## Soru

`scripts/run_pipeline.py` çalıştırıldığında Random Forest doğruluğu hastalar
arasında çok değişkendi (0.503 – 0.840, ortalama 0.693). Bu değişkenlik kod
hatasından mı, yoksa hastadan hastaya değişen elektrot yerleşiminden mi
kaynaklanıyor?

## Yöntem ve sonuç (4 adım, `electrode_signal_localization.py`)

1. **En iyi tek özelliğin etkisi vs doğruluk** — her hastanın en güçlü
   (kanal, bant) özelliğinin `|Cohen's d|` değeri RF doğruluğuyla güçlü
   korele (`r=0.897`). Ayrıca 7 hastanın da tam olarak 300 geçerli
   denemesi olduğu görüldü — yani **örneklem büyüklüğü farkı ihtimali
   elendi**. (Not: bu korelasyon kısmen beklenen bir şey — modelin
   başarısı zaten elindeki en iyi özelliğin gücüyle sınırlı.)
2. **Literatür-yaklaşık tek FFA noktasına uzaklık** — beklenenin (negatif
   korelasyon) TERSİNE `r=0.677` çıktı. **Bu yöntem hipotezi doğrulamadı.**
3. **Yaklaşık fusiform/ventral-temporal kutu içindeki elektrot sayısı** —
   `r=0.098`, neredeyse ilişkisiz. **Bu yöntem de doğrulamadı.**
4. **Veri setindeki GERÇEK anatomik etiketler** (`hemisphere`, `lobe`,
   `gyrus`, `Brodmann_Area` — 2 ve 3. adımda tahmin ettiğimiz koordinatlar
   yerine, veride zaten hazır duran alanlar) kullanılınca: tüm hastaların
   tüm kanalları havuzlanıp lobe/gyrus'a göre gruplandığında, **Lingual
   Gyrus (ort. |d|=0.303) ve Fusiform Gyrus (ort. |d|=0.268)** —
   görsel/yüz-işleme hattındaki iki bölge — Frontal/Limbic/Parietal/
   Caudate gibi ilgisiz bölgelerden (0.15-0.19) belirgin şekilde daha
   yüksek ayırt edicilik gösteriyor.

## Sonuç

Sinyal **gerçekten** beklenen anatomik bölgede (erken görsel korteksten
yüz-işleme bölgesine uzanan ventral hat) yoğunlaşıyor — bu bir tesadüf ya
da kod hatası değil. Ama hasta bazında "kaç elektrot o bölgede" gibi basit
bir sayım, n=7 ile RF doğruluğunu güvenilir şekilde tahmin etmeye yetmiyor;
önemli olan kanal SAYISI değil, o kanalların taşıdığı sinyalin GÜCÜ/KALİTESİ
gibi görünüyor. Lingual Gyrus'un Fusiform'dan bile önde çıkması, kısmen
düşük-seviyeli görsel farkların (kontrast, uzamsal frekans) da işin içinde
olabileceğine işaret ediyor — bu, "yüz vs ev" gibi decoding görevlerine
yönelik literatürde bilinen bir sınırlama (yüksek başarı, mutlaka yüksek
seviyeli/kavramsal bir temsilin kanıtı değildir).

Bu yüzden "neden bazı hastalarda düşük doğruluk" sorusunun kesin cevabı hâlâ
açık — ama artık *nerede* aradığımızı ve *neyin işe yarayıp yaramadığını*
biliyoruz. Kanıtlanmamış hipotez (elektrot-sayısı → doğruluk) ile
kanıtlanmış gözlem (sinyal doğru anatomik bölgede yoğunlaşıyor, ama coverage
sayımı yetersiz bir proxy) arasındaki fark, tam olarak roadmap'in "doğrulan-
mamış iddiayı doğrulanmıştan ayırt et" ilkesinin bir örneği.

## Not — daha önce yanlış bir sayı verilmişti

Önceki notlarda (roadmap, çalışma notları PDF'i) "kanal sayısı hastadan
hastaya 16-22 arası değişiyor" denmişti — bu YANLIŞTI. Gerçek veri 39-60
kanal arası (bkz. yukarıdaki analiz çıktısı). Düzeltiliyor.

Çalıştırma:
```bash
python analysis/electrode_signal_localization.py data/raw/faceshouses.npz
```
Tam çıktı: `analysis/electrode_signal_localization_output.txt`

---

## Ek doğrulama: Miller ve ark. (2017)'nin gürültü-eşiği bulgusunu tekrar üretme denemesi

Orijinal makale (Miller, K.J. ve ark., "Face percept formation in human
ventral temporal cortex", J Neurophysiol 2017 —
https://pmc.ncbi.nlm.nih.gov/articles/PMC5668462 ) bu VERİ SETİNİN
kaynağı: aynı 7 hasta, fusiform+lingual gyrus elektrotları, aynı iki görev
(`dat1`=localizer, `dat2`=gürültülü tespit). Ana bulguları: nöral broadband
yanıt, uyaran gürültüsü arttıkça kademeli azalıyor, ama algısal eşiğin
(~%50 gürültü) ÜZERİNDE aniden taban seviyesine düşüyor ("hepsi ya da
hiçbiri" örüntü).

Bunu `dat2` (pipeline'ın geri kalanı sadece `dat1` kullanıyordu) ile,
basitleştirilmiş bir yöntemle tekrar üretmeyi denedik
(`noise_threshold_replication.py`): her hastanın Fusiform+Lingual
kanallarının 100-400ms'lik high-gamma yanıtını, sadece gerçek yüz
uyaranlarında (`stim_cat==2`), gürültü seviyesine göre gruplayıp
hastalar arası havuzladık.

**Sonuç (`noise_threshold_replication.png`):** Genel eğilim doğru yönde —
düşük gürültüde (0-15%) ortalama yanıt ~0.18 (z-skor), orta gürültüde
(20-45%) ~0.00, yüksek gürültüde (50-100%) ~-0.07'ye düşüyor. Ama
nokta-nokta bakıldığında düzensiz (bazı gürültü seviyelerinde beklenmedik
sıçramalar var) ve makalenin öne çıkan KESKİN eşik örüntüsü net şekilde
görünmüyor — muhtemelen bizim yöntemimizin (kanalların kaba ortalaması,
basit z-skor normalizasyonu, n=7) orijinal makalenin çok daha titiz
istatistiksel yönteminden (elektrot-bazlı analiz) çok daha kaba olmasından.

**Dürüst sonuç:** kaba yönü (gürültü arttıkça sinyal zayıflıyor)
doğruladık; makalenin keskin eşik bulgusunu bu basit yöntemle net şekilde
yeniden üretemedik. Bu, "replikasyon başarısız" değil — kendi basit
yöntemimizin sınırlarını gösteren, dürüst bir kısmi doğrulama.

Çalıştırma:
```bash
python analysis/noise_threshold_replication.py data/raw/faceshouses.npz
```

---

## Ek analiz: LDA'nın düşük performansının kaynağı — kovaryans tahmini mi, algoritma mı?

### Soru

Ana pipeline'da LDA (0.531) en düşük ortalama doğruluğu veren modeldi. Bunun
sebebi LDA algoritmasının bu problem için doğası gereği yetersiz olması mı,
yoksa az örneklemle (hasta başına n=300 deneme) yüksek boyutlu (234-360
özellik, yani kanal×bant) bir kovaryans matrisini güvenilir tahmin
edememesi mi?

### Yöntem (`test_shrinkage_lda.py`)

Pipeline'ın geri kalanı (epoklama, bant gücü özellik çıkarımı, StandardScaler,
Stratified 5-Fold) BİREBİR AYNI bırakıldı; sadece LDA'nın kovaryans tahminine
"shrinkage" (regularizasyon) eklendi — tek satırlık bir ayar
(`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`).

### Sonuç — gerçek `faceshouses.npz` verisiyle, 7 hasta

| Hasta | özellik sayısı | LDA (düz) | LDA (shrinkage) | fark |
|---|---|---|---|---|
| 1 | 246 | 0.443 | 0.507 | +0.063 |
| 2 | 300 | 0.497 | 0.597 | +0.100 |
| 3 | 234 | 0.497 | 0.600 | +0.103 |
| 4 | 360 | 0.543 | 0.513 | −0.030 |
| 5 | 348 | 0.637 | 0.770 | +0.133 |
| 6 | 234 | 0.520 | 0.667 | +0.147 |
| 7 | 348 | 0.580 | 0.770 | +0.190 |

**Ortalama (hastalar arası): 0.531 ± 0.058 → 0.632 ± 0.101**

7 hastanın 6'sında iyileşme var (tek istisna: 4. hasta, hafif kötüleşme
−0.030). Düzeltilmiş LDA artık SVM'i (0.608) geçiyor ve Random Forest'a
(0.693) belirgin şekilde yaklaşıyor.

### Yorum

Bu, LDA'nın "kötü" olmasının algoritmanın kendisinden değil, örneklem
sayısının (n=300) özellik sayısına (234-360) çok yakın olduğu bu rejimde
standart kovaryans tahmininin güvenilirliğini yitirmesinden kaynaklandığını
doğruluyor: kovaryans matrisinin boyutu özellik sayısının karesiyle
büyüyor, ve n≈p durumunda bu tahmin neredeyse tekilleşip gürültülü hale
geliyor. Shrinkage bu tahmini yapay olarak sadeleştirerek düzeltiyor.
Sonuç olarak "az veri → basit model daha güvenli" sezgisi doğru, ama
"basitlik" ölçütü modelin çizdiği sınırın şekli değil, tahmin etmesi
gereken parametre sayısı olmalı — LDA'nın çizgisi basit görünse de, arka
plandaki kovaryans tahmini yüksek boyutta hiç basit değil.

Çalıştırma:
```bash
python analysis/test_shrinkage_lda.py data/raw/faceshouses.npz
```
(not: gerçek `faceshouses.npz` bu repoda değil, sadece kendi Downloads
klasöründe — komuta o yolu vermen gerekiyor, örn.
`python analysis/test_shrinkage_lda.py C:\Users\sena9\Downloads\faceshouses.npz`)

Grafik: `analysis/shrinkage_lda_comparison_1.png` (4 modelin
karşılaştırması — LDA düz, LDA shrinkage, SVM, RF)
