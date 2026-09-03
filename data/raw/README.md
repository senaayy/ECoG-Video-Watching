# Faces vs Houses ECoG Veri Setini İndirme (Miller, 2019)

Kaynak: Miller, K.J. (2019). *A library of human electrocorticographic data
and analyses*, Nature Human Behaviour — Stanford Digital Repository (SDR).

- PURL: https://purl.stanford.edu/zk881ps0522
- SearchWorks kataloğu: https://searchworks.stanford.edu/view/zk881ps0522
- Lisans: CC BY-SA 4.0

Kütüphane 16 farklı davranışsal deney içeriyor (204 kayıt, 34 hasta), her
biri druid altında ayrı bir zip dosyası olarak sunuluyor
(`https://stacks.stanford.edu/file/druid:zk881ps0522/<dosya_adi>`, örn.
doğrulanmış bir örnek: `fingerflex.zip`). Bizim ihtiyacımız olan "Faces vs
Houses" görevi, makalede **`faces_basic`** olarak adlandırılıyor.

> **Not:** Bu sandbox ortamından `purl.stanford.edu` ve `stacks.stanford.edu`
> adreslerine ağ erişimi engellendiği için `faces_basic` deneyinin tam dosya
> adını/boyutunu buradan teyit edemedim. Aşağıdaki adımlarda önce manuel
> doğrulama, sonra indirme yer alıyor.

## Adımlar

1. Tarayıcıda https://purl.stanford.edu/zk881ps0522 sayfasını aç.
2. Sayfadaki dosya/klasör listesinde **"faces_basic"** (Faces vs Houses)
   satırını bul; gerçek dosya adını (muhtemelen `faces_basic.zip`) ve
   boyutunu not al.
3. Dosya boyutuna göre:
   - **Büyükse (onlarca MB ve üzeri):** `scripts/download_faces_houses.sh`
     scriptini kullan (bkz. aşağıda). Gerekirse dosya adını script'e argüman
     olarak ver:
     ```bash
     ./scripts/download_faces_houses.sh faces_basic.zip
     ```
   - **Küçükse / script indiremiyorsa:** Tarayıcıdan doğrudan indir, sonra
     manuel olarak `data/raw/` klasörüne taşı:
     ```bash
     mv ~/Downloads/faces_basic.zip data/raw/
     unzip data/raw/faces_basic.zip -d data/raw/
     ```
4. İndirilen klasörde her hasta için bir alt klasör / `.mat` dosyası
   bulunmalı. İçeriği doğrulamak için:
   ```bash
   python src/data/load_dataset.py data/raw
   ```
   Bu script hastalar/oturumlar, değişken isimleri, boyutlar ve örnekleme
   hızını özetler.

## Alternatif: Öğretim amaçlı hazır (küçük, ön-işlenmiş) sürüm

Neuromatch Academy, aynı Miller (2019) faces/houses verisinin 7 hasta x 2
oturumluk, z-score normalize edilmiş küçük bir alt kümesini `.npz` olarak
OSF'te barındırıyor (`https://osf.io/argh7/download`, dosya adı
`faceshouses.npz`). Ham `.mat` dosyalarına erişilemezse, hızlı prototipleme
için bu sürüm de kullanılabilir; ancak bu proje ham `.mat` verisiyle
çalışmayı hedeflediği için birincil kaynak yukarıdaki SDR indirmesidir.
