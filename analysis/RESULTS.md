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
