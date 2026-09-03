#!/usr/bin/env bash
# Miller (2019) "A library of human electrocorticographic data and analyses"
# Stanford Digital Repository (SDR) — Faces vs Houses ("faces_basic") deneyi
#
# PURL   : https://purl.stanford.edu/zk881ps0522
# Katalog: https://searchworks.stanford.edu/view/zk881ps0522
# Dosyalar, druid altında tek tek indirilebilir zip'ler halinde tutuluyor:
#   https://stacks.stanford.edu/file/druid:zk881ps0522/<dosya_adi>
# (Örn. finger-flexion deneyi için doğrulanmış dosya: fingerflex.zip)
#
# ÖNEMLİ: Bu sandbox ortamından stacks.stanford.edu / purl.stanford.edu
# adreslerine erişim ağ proxy'si tarafından engelleniyor, bu yüzden
# faces_basic deneyinin GERÇEK zip dosya adını buradan doğrulayamadım.
# Aşağıdaki varsayılan isim ("faces_basic.zip") repodaki adlandırma
# deseninden (fingerflex.zip) çıkarılan bir TAHMİNDİR.
#
# KULLANIM:
#   1. Tarayıcında https://purl.stanford.edu/zk881ps0522 sayfasını aç,
#      "faces_basic" (Faces vs Houses) satırındaki dosyanın gerçek adını
#      ve linkini kontrol et.
#   2. Gerekirse aşağıdaki FILENAME değişkenini o isimle güncelle.
#   3. ./scripts/download_faces_houses.sh çalıştır.
#
# Script, dosyayı data/raw/ altına indirir ve zip ise otomatik açar.

set -euo pipefail

DRUID="zk881ps0522"
FILENAME="${1:-faces_basic.zip}"
BASE_URL="https://stacks.stanford.edu/file/druid:${DRUID}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"

mkdir -p "$OUT_DIR"

URL="${BASE_URL}/${FILENAME}"
DEST="${OUT_DIR}/${FILENAME}"

echo "İndiriliyor: $URL"
echo "Hedef      : $DEST"

# -C - : kesilen indirmeyi kaldığı yerden devam ettir
# --fail : HTTP hata kodunda (404 gibi) sessizce boş dosya oluşturmak yerine hata ver
curl -L --fail --retry 4 --retry-delay 5 -C - -o "$DEST" "$URL"

echo "İndirme tamamlandı: $DEST"

if [[ "$FILENAME" == *.zip ]]; then
  echo "Zip açılıyor -> $OUT_DIR"
  unzip -o "$DEST" -d "$OUT_DIR"
fi

echo "Bitti. data/raw altındaki içeriği kontrol et:"
echo "  ls -la \"$OUT_DIR\""
