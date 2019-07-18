#!/bin/bash

set -euo pipefail

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

for INPUT_FILENAME in $DIR/scanned-images/raw/*.png; do
  OUTPUT_FILENAME=$DIR/scanned-images/normalised/`basename $INPUT_FILENAME`
  echo "Normalising $INPUT_FILENAME"
  magick convert \
    "$INPUT_FILENAME" \
    -resize 224 \
    -colorspace Gray \
    -alpha off \
    -strip \
    "$OUTPUT_FILENAME"
done
