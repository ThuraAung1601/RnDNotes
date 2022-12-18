#!/bin/bash
# Convert pdf files to png files in the given directory
# How to run : bash pdf2png4all.sh <folder-path>

for file in $1/*.pdf ;
do
  pdftoppm $file "$file" -png   
done
