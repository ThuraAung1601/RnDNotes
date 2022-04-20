#!/bin/bash
# OCR with freely available Google Tesseract OCR Engine (perparing the baseline)
# Author : Thura Aung
# Note: Install tesseract OCR engine on your computer first
# How to run: ocr.sh <language> <sample-images-path> <outputpath>

LANG=$1;
PATH=$2;
OUTFILE=$3;

for file in "$PATH"/*.png
do
        output="$(tesseract "$file" stdout --oem 1 --psm 7 -l $LANG)"          
        echo "$output" >> $OUTFILE                                           
done 
