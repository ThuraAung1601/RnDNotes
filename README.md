# R&D Notes

- [Table of Contents](#r-d-notes)
    + [Change image extension](#change-image-extension)
    + [Concatenate multiple text files into one file](#concatenate-multiple-text-files-into-one-file)
    + [Split into multiple files one line, one text file](#split-into-multiple-files-one-line-one-text-file)
    + [Install Tesseract to the local machine](#install-tesseract-to-the-local-machine)
    + [Images to text using Tesseract](#images-to-text-using-tesseract)
    + [Fine Tuning or reTraining Tesseract](#fine-tuning-or-retraining-tesseract)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### Change image extension

- Run the command - it will loop over all files
- density is a resolution unit DPI - Dot per Inch
- Changing image formats is fine - jpg, png and tiff are popular

```shell
for f in *.png; do convert "$f" -density 300 "${f%%.*}.tif"; done
```

### Concatenate multiple text files into one file

```python
python multi2onetxt.py <path-to-files> <concatenated-file>
```
### Split into multiple files one line one text file

```python
python one2multiple.py <input-file> <output-files-path> 
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

[Tesseract Experiments log](https://github.com/ThuraAung1601/RnDNotes/blob/main/tess-experiment.log.md)

### Images to text using Tesseract

Used Tesseract Engine to recognize text in images

```shell
./ocr.sh <language> <sample-images-path> <outputpath>
```
- Default mode is --oem 1 --psm 7 
```
oem : OCR Engine Mode
  - 0    Legacy engine only.
  - 1    Neural nets LSTM engine only.
  - 2    Legacy + LSTM engines.
  - 3    Default, based on what is available

psm : Page Segmentation Method
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
```

### Fine Tuning or reTraining Tesseract

- Clone the tesseract train repository for MakeFile
```shell
git clone https://github.com/tesseract-ocr/tesstrain
```
- Install Tesseract and other requirements to setup
```shell
make leptonica tesseract
```
--- or ---
```shell
./install-tesseract.sh
```
- After the set-up, prepare the ground-truth dataset in the way the documentation mentioned.
- Notice >> 
      - Images can only be .tif, .png, .bin.png or .nrm.png, not .tiff or .jpg
      - Label text files should end with .gt.txt extension
- make training
- For fine-tuning, START_MODEL is needed which is base-model and small amount of data is fine.
- For training, it is not but plenty of data shall be needed.
```shell
make training MODEL_NAME=<model_name> START_MODEL=eng PSM=7 TESSDATA=/usr/local/share/tessdata 
```

List of other important training parameters that can be explored:

```
Variables

    MODEL_NAME         Name of the model to be built. Default: foo
    START_MODEL        Name of the model to continue from. Default: ''
    PROTO_MODEL        Name of the proto model. Default: OUTPUT_DIR/MODEL_NAME.traineddata
    WORDLIST_FILE      Optional file for dictionary DAWG. Default: OUTPUT_DIR/MODEL_NAME.wordlist
    NUMBERS_FILE       Optional file for number patterns DAWG. Default: OUTPUT_DIR/MODEL_NAME.numbers
    PUNC_FILE          Optional file for punctuation DAWG. Default: OUTPUT_DIR/MODEL_NAME.punc
    DATA_DIR           Data directory for output files, proto model, start model, etc. Default: data
    OUTPUT_DIR         Output directory for generated files. Default: DATA_DIR/MODEL_NAME
    GROUND_TRUTH_DIR   Ground truth directory. Default: OUTPUT_DIR-ground-truth
    CORES              No of cores to use for compiling leptonica/tesseract. Default: 4
    LEPTONICA_VERSION  Leptonica version. Default: 1.78.0
    TESSERACT_VERSION  Tesseract commit. Default: 4.1.1
    TESSDATA_REPO      Tesseract model repo to use (_fast or _best). Default: _best
    TESSDATA           Path to the .traineddata directory to start finetuning from. Default: ./usr/share/tessdata
    MAX_ITERATIONS     Max iterations. Default: 10000
    EPOCHS             Set max iterations based on the number of lines for training. Default: none
    DEBUG_INTERVAL     Debug Interval. Default:  0
    LEARNING_RATE      Learning rate. Default: 0.0001 with START_MODEL, otherwise 0.002
    NET_SPEC           Network specification. Default: [1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c\#\#\#]
    FINETUNE_TYPE      Finetune Training Type - Impact, Plus, Layer or blank. Default: ''
    LANG_TYPE          Language Type - Indic, RTL or blank. Default: ''
    PSM                Page segmentation mode. Default: 13
    RANDOM_SEED        Random seed for shuffling of the training data. Default: 0
    RATIO_TRAIN        Ratio of train / eval training data. Default: 0.90
    TARGET_ERROR_RATE  Stop training if the character error rate (CER in percent) gets below this value. Default: 0.01
```


