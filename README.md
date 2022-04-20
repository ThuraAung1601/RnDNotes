## R&D Notes

### Change image extension

- Run the command - it will loop over all files
- Changing image formats is fine - jpg, png and tiff are popular
```shell
for f in *.jpg; do convert "$f" "${f%%.*}.png"; done
```

### Install Tesseract to the local machine 

Give execution permission to the script
```shell
chmod +x install-tesseract.sh
```
Run the script
```shell
./install-tesseract.sh
```

### Images to text using Tesseract

Used Tesseract Engine to recognize text in images

```shell
./ocr.sh <language> <sample-images-path> <outputpath>
```
- Default mode is --oem 1 --psm 7 
- oem : OCR Engine Mode
  - 0    Legacy engine only.
  - 1    Neural nets LSTM engine only.
  - 2    Legacy + LSTM engines.
  - 3    Default, based on what is available
- psm : Page Segmentation Method
  -  0    Orientation and script detection (OSD) only.
  -  1    Automatic page segmentation with OSD.
  -  2    Automatic page segmentation, but no OSD, or OCR.
  -  3    Fully automatic page segmentation, but no OSD. (Default)
  -  4    Assume a single column of text of variable sizes.
  -  5    Assume a single uniform block of vertically aligned text.
  -  6    Assume a single uniform block of text.
  -  7    Treat the image as a single text line.
  -  8    Treat the image as a single word.
  -  9    Treat the image as a single word in a circle.
  - 10    Treat the image as a single character.
  - 11    Sparse text. Find as much text as possible in no particular order.
  - 12    Sparse text with OSD.
  - 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
