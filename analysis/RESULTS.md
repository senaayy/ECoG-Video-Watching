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
