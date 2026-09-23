# Faces vs Houses ECoG Veri Setini İndirme (Miller, 2019)

Kaynak: Miller, K.J. ve ark. — "Face percept formation in human ventral
temporal cortex" (J Neurophysiol, 2017) ve ilişkili çalışmalar; veri seti
Miller (2019) *A library of human electrocorticographic data and analyses*
(Nature Human Behaviour) kütüphanesinin bir parçası.

## Birincil kaynak: hazır `.npz` (önerilen)

Neuromatch Academy, bu görevin (7 hasta x 2 oturum) z-score normalize
edilmiş, doğrudan kullanıma hazır bir sürümünü OSF'te barındırıyor:

- **URL:** https://osf.io/argh7/download
- **Dosya adı:** `faceshouses.npz`
- **Boyut:** birkaç MB (küçük, hızlı indirilir)

Bu sürümü öneriyoruz çünkü (1) dosya adı/boyutu doğrulanmış ve çalışan bir
linkten geliyor, (2) ham `.mat` dosyalarındaki gibi ekstra ayrıştırma
gerektirmiyor, (3) projenin asıl hedefi olan "yüz mü ev mi" sınıflandırması
için gereken her şeyi zaten içeriyor.

> **Not (ağ erişimi):** `osf.io` hem bu geliştirme ortamından hem de test
> ettiğimiz başka bir bağlı cihazdan engellenmiş durumda (proxy/allowlist
> kısıtlaması). Kendi normal tarayıcınızdan (Claude'un içinden değil) linki
> açıp dosyayı indirmeniz gerekiyor — bu kısıtlama sadece otomatik
> indirmeyi etkiliyor, sizin kendi internetinizi etkilemez.

### İndirme adımları

1. Kendi tarayıcından şu linki aç: https://osf.io/argh7/download
   (indirme otomatik başlamalı; başlamazsa sayfadaki indirme butonuna tıkla)
2. İnen `faceshouses.npz` dosyasını bu projenin `data/raw/` klasörüne taşı:
   ```bash
   mv ~/Downloads/faceshouses.npz data/raw/
   ```
3. İçeriği doğrulamak için:
   ```bash
   python src/data/load_faceshouses.py data/raw/faceshouses.npz
   ```

### Veri şeması

Dosya, her biri `dat1` (pasif izleme) ve `dat2` (gürültülü/keypress'li
tespit görevi) içeren 7 hastalık bir liste (`alldat`) barındırır:

**Deney 1 (`dat1`) — pasif izleme:**
| Alan | Açıklama |
|---|---|
| `V` | sürekli voltaj verisi (zaman × kanal) |
| `srate` | örnekleme hızı (1000 Hz) |
| `t_on` / `t_off` | uyaran başlangıç/bitiş zamanı (örnek indeksi); `t_off - t_on = 400` |
| `stim_id` | uyaran kimliği 1-100 (1-50 ev, 51-100 yüz) |
| `locs` | elektrotların 3B beyin yüzeyi konumları |

**Deney 2 (`dat2`) — gürültülü tespit görevi:**
| Alan | Açıklama |
|---|---|
| `V`, `srate`, `t_on`, `t_off`, `locs` | Deney 1'deki gibi (`t_off - t_on = 1000`) |
| `stim_cat` | uyaran kategorisi (1 = ev, 2 = yüz) — **sınıflandırma etiketi bu** |
| `stim_noise` | uyarana eklenen gürültü yüzdesi (0-100) |
| `key_press` | katılımcı "yüz" olarak algıladığında basılan tuş zamanı |

Bu proje öncelikle **Deney 1'i (`dat1`, temiz/gürültüsüz yüz-ev ayrımı)**
kullanıyor; Deney 2 (`dat2`) gürültü seviyesine göre performansın nasıl
değiştiğini incelemek için ek/ileri analiz olarak düşünülebilir.

## Alternatif: ham `.mat` dosyaları (Stanford Digital Repository)

Kütüphanenin tam/ham hali (204 kayıt, 34 hasta, 16 farklı görev) Stanford
Digital Repository'de duruyor:

- PURL: https://purl.stanford.edu/zk881ps0522
- SearchWorks kataloğu: https://searchworks.stanford.edu/view/zk881ps0522
- Lisans: CC BY-SA 4.0

> **Not:** Bu ortamdan `purl.stanford.edu` ve `stacks.stanford.edu`
> adreslerine de erişim engelli, ve "faces_basic" deneyinin tam dosya
> adı/boyutu buradan hâlâ doğrulanamadı. Yukarıdaki `.npz` sürümü aynı
> deneyin işlenmiş halini içerdiği için, ham `.mat` verisine ancak ek bir
> analiz (örn. farklı bir görev/hasta) gerekiyorsa ihtiyaç duyulur.
> Gerekirse `scripts/download_faces_houses.sh` script'i hâlâ dursun, ama
> önce https://purl.stanford.edu/zk881ps0522 sayfasından gerçek dosya
> adını doğrulaman gerekir.
