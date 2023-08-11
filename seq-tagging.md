# ML-based Sequence Tagging for Myanmar language
The dataset used is mySentence corpus version 1.0: https://github.com/ye-kyaw-thu/mySentence 

- [CRF model](#CRF-model)
- [RDR model](#RDR-model)
- [HMM model](#HMM-model)

### CRF model
Conditional Random Fields <br /> 
Step 1: Download CRF++ Tool (CRF++-0.58.tar.gz)
 
 ```
 https://drive.google.com/drive/u/0/folders/0B4y35FiV1wh7fngteFhHQUN2Y1B5eUJBNHZUemJYQV9VWlBUb3JlX0xBdWVZTWtSbVBneU0
 ```
 
Step 2: Read dependencies for installation for CRF++ in detail and Install
 
 ```
$ ./configure 
$ make
$ sudo make install
 ```
 
Step 3: Check to use 
 
 ```
$ crf_learn --help
$ crf_test --help
 ```
 
Step 4: Training and Testing CRF++ with CRF format data in "./data/" directory
 ```
$ cat template
  # Unigram
  U00:%x[-2,0]
  U01:%x[-1,0]
  U02:%x[0,0]
  U03:%x[1,0]
  U04:%x[2,0]

  # Bigram
 ```
Change to column format using this program.
- ch2col.pl: https://github.com/ye-kyaw-thu/mySentence/blob/main/ver1.0/scripts/ch2col.pl
 ``` 
$ crf_learn ./template train.col ./crf_tagger.crf-model | tee crf.train.log
 ```

```
$ crf_test -m ./crf_tagger.crf-model test.col > ./result.col
$ head -5 result.col
  အခု		B	B
  သန့်စင်ခန်း	N	O
  ကို		N	N
  သုံး		N	N
  ပါရစေ		E	E
```
First column is text, second column is ground-truth label and third column is model prediction. Therefore cut "text" and "prediction" only.
```
$ cut -f1,3 ./result.col > ./result.col.f13
$ head -5 result.col.f13
  အခု		B
  သန့်စင်ခန်း	O
  ကို		N
  သုံး		N
  ပါရစေ		E
```
Convert column format to line (col2line.pl is in the program folder.)
- col2line.pl: https://github.com/ye-kyaw-thu/mySentence/blob/main/ver1.0/scripts/col2line.pl 
```
$ perl col2line.pl result.col.f13 > result.txt
$ head -5 result.txt
  အခု/B သန့်စင်ခန်း/O ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/O မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/N မ/O ရ/O အောင်/O ကျ/N သွား/N တယ်/E
  ဂီနီ/B နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```

### RDR model

Ripple Down Rule-based POSTagger <br />
Step 1: Download RDRPOSTagger

```
$ git clone https://github.com/datquocnguyen/RDRPOSTagger
```
Step 2: Change Directory to Python port and Train with tagged data
```
$ cd pSCRDRtagger
```
Train data example ...
```
$ head -5 train.tagged 
  ဘာ/B ရယ်/O လို့/O တိတိကျကျ/O ထောက်မပြ/O နိုင်/O ပေမဲ့/O ပြဿနာ/O တစ်/O ခု/O ခု/O ရှိ/O တယ်/N နဲ့/N တူ/N တယ်/E
  လူ့/B အဖွဲ့အစည်း/O က/O ရှပ်ထွေး/O လာ/O တာ/O နဲ့/O အမျှ/O အရင်/O က/O မ/O ရှိ/O ခဲ့/O တဲ့/O လူမှုရေး/O ပြဿနာ/O တွေ/O ဖြစ်ပေါ်/N လာ/N ခဲ့/N တယ်/E
  အခု/B အလုပ်/O လုပ်/N နေ/N ပါ/N တယ်/E
  ကြည့်/B ရေစာ/O တွေ/O က/O အဲဒီ/O တစ်/O ခု/O နဲ့/N မ/N တူ/N ဘူး/E
  ဘူမိ/B ရုပ်သွင်/O ပညာ/O သည်/O ကုန်းမြေသဏ္ဌာန်/O များ/O ကို/O လေ့လာ/O သော/N ပညာရပ်/N ဖြစ်/N သည်/E
```
```
$ python2.7 RDRPOSTagger.py train train.tagged
```
Step 3: Testing <br />
Test data format example
```
$ head test.my
  အခု သန့်စင်ခန်း ကို သုံး ပါရစေ
  လူငယ် တွေ က ပုံစံတကျ ရှိ မှု ကို မ ကြိုက် ဘူး
  ဒီ တစ် ခေါက် ကိစ္စ ကြောင့် ကျွန်တော့် ရဲ့ သိက္ခာ အဖတ်ဆယ် လို့ မ ရ အောင် ကျ သွား တယ်
  ဂီနီ နိုင်ငံ သည် ကမ္ဘာ ပေါ် တွင် ဘောက် ဆိုက် တင်ပို့ မှု အများဆုံး နိုင်ငံ ဖြစ် သည်
  ဘာ လုပ် ရ မလဲ ဟင်
```
Testing ...
```
$ python2.7 RDRPOSTagger.py tag train.tagged.RDR train.tagged.DICT test.my

	=> Read a POS tagging model from train.tagged.RDR

	=> Read a lexicon from train.tagged.DICT

	=> Perform POS tagging on test.my

	Output file: test.my.TAGGED
```
RDRPOSTagger tagged data ...
```
$ head -5 test.my.TAGGED 
  အခု/B သန့်စင်ခန်း/O ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/N မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/O မ/O ရ/O အောင်/O ကျ/N သွား/N တယ်/E
  ဂီနီ/B နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```

### HMM model
3-gram Hidden Markov Models POSTagger with jita-0.3.3. <br />
Step 1: Download jita-0.3.3.

```
https://github.com/danieldk/jitar/releases
```
Step 2: Train with jita-0.3.3.
```
$ jitar-0.3.3-bin/jitar-0.3.3$ bin/jitar-train brown ./train.tagged ./hmm.model | tee 3gHMM-training.log
```
Step 3: Testing ... <br/>
Test data format example
```
$ head test.my
  အခု သန့်စင်ခန်း ကို သုံး ပါရစေ
  လူငယ် တွေ က ပုံစံတကျ ရှိ မှု ကို မ ကြိုက် ဘူး
  ဒီ တစ် ခေါက် ကိစ္စ ကြောင့် ကျွန်တော့် ရဲ့ သိက္ခာ အဖတ်ဆယ် လို့ မ ရ အောင် ကျ သွား တယ်
  ဂီနီ နိုင်ငံ သည် ကမ္ဘာ ပေါ် တွင် ဘောက် ဆိုက် တင်ပို့ မှု အများဆုံး နိုင်ငံ ဖြစ် သည်
  ဘာ လုပ် ရ မလဲ ဟင်
```

```
$ cat test.my | bin/jitar-tag ./hmm.model > ./test.hmm.result
```
```
$ head -5 test.hmm.result
	B N N N E
	B O O O O O O N N E
	B O O O O O O O O N N N N N N E
	O O O O O O O O O O O N N E
	B N N N E
```
Make pair format with mk-pair.pl in program folder
```
$ perl mk-pair.pl test.my test.hmm.result > test.hmm.TAGGED
```
```
$ head -5 test.hmm.TAGGED
  အခု/B သန့်စင်ခန်း/N ကို/N သုံး/N ပါရစေ/E
  လူငယ်/B တွေ/O က/O ပုံစံတကျ/O ရှိ/O မှု/O ကို/O မ/N ကြိုက်/N ဘူး/E
  ဒီ/B တစ်/O ခေါက်/O ကိစ္စ/O ကြောင့်/O ကျွန်တော့်/O ရဲ့/O သိက္ခာ/O အဖတ်ဆယ်/O လို့/N မ/N ရ/N အောင်/N ကျ/N သွား/N တယ်/E
  ဂီနီ/O နိုင်ငံ/O သည်/O ကမ္ဘာ/O ပေါ်/O တွင်/O ဘောက်/O ဆိုက်/O တင်ပို့/O မှု/O အများဆုံး/O နိုင်ငံ/N ဖြစ်/N သည်/E
  ဘာ/B လုပ်/N ရ/N မလဲ/N ဟင်/E
```
