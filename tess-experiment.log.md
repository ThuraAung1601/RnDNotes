# Tesseract OCR Engine

- Tesseract Docs : https://tesseract-ocr.github.io/tessdoc/
- April 2022
- Thura Aung

## Installation
### Preparation and setup

```shell
sudo apt-get update
sudo apt-get install -y wget unzip bc vim python3-pip libleptonica-dev git
```
Packages to compile Tesseract Engine

```shell
sudo apt-get install -y --reinstall make 
sudo apt-get install -y g++ autoconf utomake libtool pkg-config libpng-dev libjpeg8-dev libtiff5-dev libicu-dev \
                        libpango1.0-dev autoconf-archive
```
Install Tesseract and make
```shell
git clone https://github.com/tesseract-ocr/tesseract
cd tesseract && ./autogen.sh && ./configure && make && make install && ldconfig 
```
```
Cloning into 'tesseract'...
remote: Enumerating objects: 45524, done.
remote: Counting objects: 100% (154/154), done.
remote: Compressing objects: 100% (93/93), done.
remote: Total 45524 (delta 72), reused 106 (delta 60), pack-reused 45370
Receiving objects: 100% (45524/45524), 50.99 MiB | 753.00 KiB/s, done.
Resolving deltas: 100% (35731/35731), done.
```

```
Running aclocal
Running /usr/bin/libtoolize
libtoolize: putting auxiliary files in AC_CONFIG_AUX_DIR, 'config'.
libtoolize: copying file 'config/ltmain.sh'
libtoolize: putting macros in AC_CONFIG_MACRO_DIRS, 'm4'.
libtoolize: copying file 'm4/libtool.m4'
libtoolize: copying file 'm4/ltoptions.m4'
libtoolize: copying file 'm4/ltsugar.m4'
libtoolize: copying file 'm4/ltversion.m4'
libtoolize: copying file 'm4/lt~obsolete.m4'
Running aclocal
Running autoconf
Running autoheader
Running automake --add-missing --copy
configure.ac:414: installing 'config/compile'
configure.ac:89: installing 'config/config.guess'
configure.ac:89: installing 'config/config.sub'
configure.ac:27: installing 'config/install-sh'
configure.ac:27: installing 'config/missing'
Makefile.am: installing 'config/depcomp'
parallel-tests: installing 'config/test-driver'

All done.
To build the software now, do something like:

$ ./configure [--enable-debug] [...other options]
checking for g++... g++
checking whether the C++ compiler works... yes
checking for C++ compiler default output file name... a.out
checking for suffix of executables... 
checking whether we are cross compiling... no
checking for suffix of object files... o
checking whether we are using the GNU C++ compiler... yes
checking whether g++ accepts -g... yes
checking for a BSD-compatible install... /usr/bin/install -c
checking whether build environment is sane... yes
checking for a thread-safe mkdir -p... /usr/bin/mkdir -p
checking for gawk... no
checking for mawk... mawk
checking whether make sets $(MAKE)... yes
checking whether make supports the include directive... yes (GNU style)
checking whether make supports nested variables... yes
checking dependency style of g++... gcc3
checking for a sed that does not truncate output... /usr/bin/sed
checking Major version... 5
checking Minor version... 1
checking Point version... 0-28-g5e053
checking whether make supports nested variables... (cached) yes
checking build system type... x86_64-pc-linux-gnu
checking host system type... x86_64-pc-linux-gnu
checking whether C++ compiler accepts -Werror=unused-command-line-argument... no
checking whether C++ compiler accepts -mavx... yes
checking whether C++ compiler accepts -mavx2... yes
checking whether C++ compiler accepts -mavx512f... yes
checking whether C++ compiler accepts -mfma... yes
checking whether C++ compiler accepts -msse4.1... yes
checking for feenableexcept... yes
checking whether C++ compiler accepts -fopenmp-simd... yes
checking --enable-float32 argument... 
checking --enable-graphics argument... 
checking --enable-legacy argument... 
checking for g++ option to support OpenMP... -fopenmp
checking how to run the C++ preprocessor... g++ -E
checking for grep that handles long lines and -e... /usr/bin/grep
checking for egrep... /usr/bin/grep -E
checking for ANSI C header files... yes
checking for sys/types.h... yes
checking for sys/stat.h... yes
checking for stdlib.h... yes
checking for string.h... yes
checking for memory.h... yes
checking for strings.h... yes
checking for inttypes.h... yes
checking for stdint.h... yes
checking for unistd.h... yes
checking tiffio.h usability... yes
checking tiffio.h presence... yes
checking for tiffio.h... yes
checking --enable-opencl argument... 
checking tensorflow/core/framework/graph.pb.h usability... no
checking tensorflow/core/framework/graph.pb.h presence... no
checking for tensorflow/core/framework/graph.pb.h... no
checking --enable-visibility argument... 
checking whether to use tessdata-prefix... yes
checking if compiling with clang... no
checking whether to enable debugging... 
checking how to print strings... printf
checking for gcc... gcc
checking whether we are using the GNU C compiler... yes
checking whether gcc accepts -g... yes
checking for gcc option to accept ISO C89... none needed
checking whether gcc understands -c and -o together... yes
checking dependency style of gcc... gcc3
checking for a sed that does not truncate output... (cached) /usr/bin/sed
checking for fgrep... /usr/bin/grep -F
checking for ld used by gcc... /usr/bin/ld
checking if the linker (/usr/bin/ld) is GNU ld... yes
checking for BSD- or MS-compatible name lister (nm)... /usr/bin/nm -B
checking the name lister (/usr/bin/nm -B) interface... BSD nm
checking whether ln -s works... yes
checking the maximum length of command line arguments... 1572864
checking how to convert x86_64-pc-linux-gnu file names to x86_64-pc-linux-gnu format... func_convert_file_noop
checking how to convert x86_64-pc-linux-gnu file names to toolchain format... func_convert_file_noop
checking for /usr/bin/ld option to reload object files... -r
checking for objdump... objdump
checking how to recognize dependent libraries... pass_all
checking for dlltool... no
checking how to associate runtime and link libraries... printf %s\n
checking for ar... ar
checking for archiver @FILE support... @
checking for strip... strip
checking for ranlib... ranlib
checking command to parse /usr/bin/nm -B output from gcc object... ok
checking for sysroot... no
checking for a working dd... /usr/bin/dd
checking how to truncate binary pipes... /usr/bin/dd bs=4096 count=1
checking for mt... mt
checking if mt is a manifest tool... no
checking for dlfcn.h... yes
checking for objdir... .libs
checking if gcc supports -fno-rtti -fno-exceptions... no
checking for gcc option to produce PIC... -fPIC -DPIC
checking if gcc PIC flag -fPIC -DPIC works... yes
checking if gcc static flag -static works... yes
checking if gcc supports -c -o file.o... yes
checking if gcc supports -c -o file.o... (cached) yes
checking whether the gcc linker (/usr/bin/ld -m elf_x86_64) supports shared libraries... yes
checking whether -lc should be explicitly linked in... no
checking dynamic linker characteristics... GNU/Linux ld.so
checking how to hardcode library paths into programs... immediate
checking whether stripping libraries is possible... yes
checking if libtool supports shared libraries... yes
checking whether to build shared libraries... yes
checking whether to build static libraries... yes
checking how to run the C++ preprocessor... g++ -E
checking for ld used by g++... /usr/bin/ld -m elf_x86_64
checking if the linker (/usr/bin/ld -m elf_x86_64) is GNU ld... yes
checking whether the g++ linker (/usr/bin/ld -m elf_x86_64) supports shared libraries... yes
checking for g++ option to produce PIC... -fPIC -DPIC
checking if g++ PIC flag -fPIC -DPIC works... yes
checking if g++ static flag -static works... yes
checking if g++ supports -c -o file.o... yes
checking if g++ supports -c -o file.o... (cached) yes
checking whether the g++ linker (/usr/bin/ld -m elf_x86_64) supports shared libraries... yes
checking dynamic linker characteristics... (cached) GNU/Linux ld.so
checking how to hardcode library paths into programs... immediate
checking whether C++ compiler accepts -std=c++17... yes
checking whether C++ compiler accepts -std=c++20... no
checking for library containing pthread_create... -lpthread
checking for brew... false
checking for asciidoc... true
checking for xsltproc... true
checking for wchar_t... yes
checking for long long int... yes
checking for pkg-config... /usr/bin/pkg-config
checking pkg-config is at least version 0.9.0... yes
checking for libcurl... no
checking for LEPTONICA... yes
checking for libarchive... no
checking for ICU_UC... yes
checking for ICU_I18N... yes
checking for pango... yes
checking for cairo... yes
checking for pangocairo... yes
checking for pangoft2... yes
checking that generated files are newer than configure... done
configure: creating ./config.status
config.status: creating include/tesseract/version.h
config.status: creating Makefile
config.status: creating tesseract.pc
config.status: creating tessdata/Makefile
config.status: creating tessdata/configs/Makefile
config.status: creating tessdata/tessconfigs/Makefile
config.status: creating java/Makefile
config.status: creating java/com/Makefile
config.status: creating java/com/google/Makefile
config.status: creating java/com/google/scrollview/Makefile
config.status: creating java/com/google/scrollview/events/Makefile
config.status: creating java/com/google/scrollview/ui/Makefile
config.status: creating include/config_auto.h
config.status: executing depfiles commands
config.status: executing libtool commands

Configuration is done.
You can now build and install tesseract by running:

$ make
$ sudo make install
$ sudo ldconfig

This will also build the documentation.

Training tools can be built and installed with:

$ make training
$ sudo make training-install

Making all in .
make[1]: Entering directory '/home/tra/tesseract'
  CXX      src/tesseract-tesseract.o
  CXX      src/api/libtesseract_la-baseapi.lo
  CXX      src/api/libtesseract_la-altorenderer.lo
  CXX      src/api/libtesseract_la-capi.lo
  CXX      src/api/libtesseract_la-hocrrenderer.lo
  CXX      src/api/libtesseract_la-lstmboxrenderer.lo
  CXX      src/api/libtesseract_la-pdfrenderer.lo
  CXX      src/api/libtesseract_la-renderer.lo
  CXX      src/api/libtesseract_la-wordstrboxrenderer.lo
  CXX      src/arch/libtesseract_la-intsimdmatrix.lo
  CXX      src/arch/libtesseract_la-simddetect.lo
  CXX      src/ccmain/libtesseract_la-applybox.lo
  CXX      src/ccmain/libtesseract_la-control.lo
  CXX      src/ccmain/libtesseract_la-linerec.lo
  CXX      src/ccmain/libtesseract_la-ltrresultiterator.lo
  CXX      src/ccmain/libtesseract_la-mutableiterator.lo
  CXX      src/ccmain/libtesseract_la-output.lo
  CXX      src/ccmain/libtesseract_la-pageiterator.lo
  CXX      src/ccmain/libtesseract_la-pagesegmain.lo
  CXX      src/ccmain/libtesseract_la-pagewalk.lo
  CXX      src/ccmain/libtesseract_la-paragraphs.lo
  CXX      src/ccmain/libtesseract_la-paramsd.lo
  CXX      src/ccmain/libtesseract_la-pgedit.lo
  CXX      src/ccmain/libtesseract_la-reject.lo
  CXX      src/ccmain/libtesseract_la-resultiterator.lo
  CXX      src/ccmain/libtesseract_la-tessedit.lo
  CXX      src/ccmain/libtesseract_la-tesseractclass.lo
  CXX      src/ccmain/libtesseract_la-tessvars.lo
  CXX      src/ccmain/libtesseract_la-thresholder.lo
  CXX      src/ccmain/libtesseract_la-werdit.lo
  CXX      src/ccmain/libtesseract_la-adaptions.lo
  CXX      src/ccmain/libtesseract_la-docqual.lo
  CXX      src/ccmain/libtesseract_la-equationdetect.lo
  CXX      src/ccmain/libtesseract_la-fixspace.lo
  CXX      src/ccmain/libtesseract_la-fixxht.lo
  CXX      src/ccmain/libtesseract_la-osdetect.lo
  CXX      src/ccmain/libtesseract_la-par_control.lo
  CXX      src/ccmain/libtesseract_la-recogtraining.lo
  CXX      src/ccmain/libtesseract_la-superscript.lo
  CXX      src/ccmain/libtesseract_la-tessbox.lo
  CXX      src/ccmain/libtesseract_la-tfacepp.lo
  CXX      src/ccstruct/libtesseract_la-blamer.lo
  CXX      src/ccstruct/libtesseract_la-blobbox.lo
  CXX      src/ccstruct/libtesseract_la-blobs.lo
  CXX      src/ccstruct/libtesseract_la-blread.lo
  CXX      src/ccstruct/libtesseract_la-boxread.lo
  CXX      src/ccstruct/libtesseract_la-boxword.lo
  CXX      src/ccstruct/libtesseract_la-ccstruct.lo
  CXX      src/ccstruct/libtesseract_la-coutln.lo
  CXX      src/ccstruct/libtesseract_la-detlinefit.lo
  CXX      src/ccstruct/libtesseract_la-dppoint.lo
  CXX      src/ccstruct/libtesseract_la-image.lo
  CXX      src/ccstruct/libtesseract_la-imagedata.lo
  CXX      src/ccstruct/libtesseract_la-linlsq.lo
  CXX      src/ccstruct/libtesseract_la-matrix.lo
  CXX      src/ccstruct/libtesseract_la-mod128.lo
  CXX      src/ccstruct/libtesseract_la-normalis.lo
  CXX      src/ccstruct/libtesseract_la-ocrblock.lo
  CXX      src/ccstruct/libtesseract_la-ocrpara.lo
  CXX      src/ccstruct/libtesseract_la-ocrrow.lo
  CXX      src/ccstruct/libtesseract_la-otsuthr.lo
  CXX      src/ccstruct/libtesseract_la-pageres.lo
  CXX      src/ccstruct/libtesseract_la-pdblock.lo
  CXX      src/ccstruct/libtesseract_la-points.lo
  CXX      src/ccstruct/libtesseract_la-polyaprx.lo
  CXX      src/ccstruct/libtesseract_la-polyblk.lo
  CXX      src/ccstruct/libtesseract_la-quadlsq.lo
  CXX      src/ccstruct/libtesseract_la-quspline.lo
  CXX      src/ccstruct/libtesseract_la-ratngs.lo
  CXX      src/ccstruct/libtesseract_la-rect.lo
  CXX      src/ccstruct/libtesseract_la-rejctmap.lo
  CXX      src/ccstruct/libtesseract_la-seam.lo
  CXX      src/ccstruct/libtesseract_la-split.lo
  CXX      src/ccstruct/libtesseract_la-statistc.lo
  CXX      src/ccstruct/libtesseract_la-stepblob.lo
  CXX      src/ccstruct/libtesseract_la-werd.lo
  CXX      src/ccstruct/libtesseract_la-fontinfo.lo
  CXX      src/ccstruct/libtesseract_la-params_training_featdef.lo
  CXX      src/classify/libtesseract_la-classify.lo
  CXX      src/classify/libtesseract_la-adaptive.lo
  CXX      src/classify/libtesseract_la-adaptmatch.lo
  CXX      src/classify/libtesseract_la-blobclass.lo
  CXX      src/classify/libtesseract_la-cluster.lo
  CXX      src/classify/libtesseract_la-clusttool.lo
  CXX      src/classify/libtesseract_la-cutoffs.lo
  CXX      src/classify/libtesseract_la-featdefs.lo
  CXX      src/classify/libtesseract_la-float2int.lo
  CXX      src/classify/libtesseract_la-fpoint.lo
  CXX      src/classify/libtesseract_la-intfeaturespace.lo
  CXX      src/classify/libtesseract_la-intfx.lo
  CXX      src/classify/libtesseract_la-intmatcher.lo
  CXX      src/classify/libtesseract_la-intproto.lo
  CXX      src/classify/libtesseract_la-kdtree.lo
  CXX      src/classify/libtesseract_la-mf.lo
  CXX      src/classify/libtesseract_la-mfoutline.lo
  CXX      src/classify/libtesseract_la-mfx.lo
  CXX      src/classify/libtesseract_la-normfeat.lo
  CXX      src/classify/libtesseract_la-normmatch.lo
  CXX      src/classify/libtesseract_la-ocrfeatures.lo
  CXX      src/classify/libtesseract_la-outfeat.lo
  CXX      src/classify/libtesseract_la-picofeat.lo
  CXX      src/classify/libtesseract_la-protos.lo
  CXX      src/classify/libtesseract_la-shapeclassifier.lo
  CXX      src/classify/libtesseract_la-shapetable.lo
  CXX      src/classify/libtesseract_la-tessclassifier.lo
  CXX      src/classify/libtesseract_la-trainingsample.lo
  CXX      src/cutil/libtesseract_la-oldlist.lo
  CXX      src/dict/libtesseract_la-context.lo
  CXX      src/dict/libtesseract_la-dawg.lo
  CXX      src/dict/libtesseract_la-dawg_cache.lo
  CXX      src/dict/libtesseract_la-dict.lo
  CXX      src/dict/libtesseract_la-stopper.lo
  CXX      src/dict/libtesseract_la-trie.lo
  CXX      src/dict/libtesseract_la-hyphen.lo
  CXX      src/dict/libtesseract_la-permdawg.lo
  CXX      src/textord/libtesseract_la-alignedblob.lo
  CXX      src/textord/libtesseract_la-baselinedetect.lo
  CXX      src/textord/libtesseract_la-bbgrid.lo
  CXX      src/textord/libtesseract_la-blkocc.lo
  CXX      src/textord/libtesseract_la-blobgrid.lo
  CXX      src/textord/libtesseract_la-ccnontextdetect.lo
  CXX      src/textord/libtesseract_la-cjkpitch.lo
  CXX      src/textord/libtesseract_la-colfind.lo
  CXX      src/textord/libtesseract_la-colpartition.lo
  CXX      src/textord/libtesseract_la-colpartitionset.lo
  CXX      src/textord/libtesseract_la-colpartitiongrid.lo
  CXX      src/textord/libtesseract_la-devanagari_processing.lo
  CXX      src/textord/libtesseract_la-drawtord.lo
  CXX      src/textord/libtesseract_la-edgblob.lo
  CXX      src/textord/libtesseract_la-edgloop.lo
  CXX      src/textord/libtesseract_la-fpchop.lo
  CXX      src/textord/libtesseract_la-gap_map.lo
  CXX      src/textord/libtesseract_la-imagefind.lo
  CXX      src/textord/libtesseract_la-linefind.lo
  CXX      src/textord/libtesseract_la-makerow.lo
  CXX      src/textord/libtesseract_la-oldbasel.lo
  CXX      src/textord/libtesseract_la-pithsync.lo
  CXX      src/textord/libtesseract_la-pitsync1.lo
  CXX      src/textord/libtesseract_la-scanedg.lo
  CXX      src/textord/libtesseract_la-sortflts.lo
  CXX      src/textord/libtesseract_la-strokewidth.lo
  CXX      src/textord/libtesseract_la-tabfind.lo
  CXX      src/textord/libtesseract_la-tablefind.lo
  CXX      src/textord/libtesseract_la-tabvector.lo
  CXX      src/textord/libtesseract_la-tablerecog.lo
  CXX      src/textord/libtesseract_la-textlineprojection.lo
  CXX      src/textord/libtesseract_la-textord.lo
  CXX      src/textord/libtesseract_la-topitch.lo
  CXX      src/textord/libtesseract_la-tordmain.lo
  CXX      src/textord/libtesseract_la-tospace.lo
  CXX      src/textord/libtesseract_la-tovars.lo
  CXX      src/textord/libtesseract_la-underlin.lo
  CXX      src/textord/libtesseract_la-wordseg.lo
  CXX      src/textord/libtesseract_la-workingpartset.lo
  CXX      src/textord/libtesseract_la-equationdetectbase.lo
  CXX      src/viewer/libtesseract_la-scrollview.lo
  CXX      src/viewer/libtesseract_la-svmnode.lo
  CXX      src/viewer/libtesseract_la-svutil.lo
  CXX      src/wordrec/libtesseract_la-tface.lo
  CXX      src/wordrec/libtesseract_la-wordrec.lo
  CXX      src/wordrec/libtesseract_la-associate.lo
  CXX      src/wordrec/libtesseract_la-chop.lo
  CXX      src/wordrec/libtesseract_la-chopper.lo
  CXX      src/wordrec/libtesseract_la-drawfx.lo
  CXX      src/wordrec/libtesseract_la-findseam.lo
  CXX      src/wordrec/libtesseract_la-gradechop.lo
  CXX      src/wordrec/libtesseract_la-language_model.lo
  CXX      src/wordrec/libtesseract_la-lm_consistency.lo
  CXX      src/wordrec/libtesseract_la-lm_pain_points.lo
  CXX      src/wordrec/libtesseract_la-lm_state.lo
  CXX      src/wordrec/libtesseract_la-outlines.lo
  CXX      src/wordrec/libtesseract_la-params_model.lo
  CXX      src/wordrec/libtesseract_la-pieces.lo
  CXX      src/wordrec/libtesseract_la-plotedges.lo
  CXX      src/wordrec/libtesseract_la-render.lo
  CXX      src/wordrec/libtesseract_la-segsearch.lo
  CXX      src/wordrec/libtesseract_la-wordclass.lo
  CXX      src/ccutil/libtesseract_ccutil_la-ccutil.lo
  CXX      src/ccutil/libtesseract_ccutil_la-clst.lo
  CXX      src/ccutil/libtesseract_ccutil_la-elst2.lo
  CXX      src/ccutil/libtesseract_ccutil_la-elst.lo
  CXX      src/ccutil/libtesseract_ccutil_la-errcode.lo
  CXX      src/ccutil/libtesseract_ccutil_la-serialis.lo
  CXX      src/ccutil/libtesseract_ccutil_la-scanutils.lo
  CXX      src/ccutil/libtesseract_ccutil_la-tessdatamanager.lo
  CXX      src/ccutil/libtesseract_ccutil_la-tprintf.lo
  CXX      src/ccutil/libtesseract_ccutil_la-unichar.lo
  CXX      src/ccutil/libtesseract_ccutil_la-unicharcompress.lo
  CXX      src/ccutil/libtesseract_ccutil_la-unicharmap.lo
  CXX      src/ccutil/libtesseract_ccutil_la-unicharset.lo
  CXX      src/ccutil/libtesseract_ccutil_la-params.lo
  CXX      src/ccutil/libtesseract_ccutil_la-ambigs.lo
  CXX      src/ccutil/libtesseract_ccutil_la-bitvector.lo
  CXX      src/ccutil/libtesseract_ccutil_la-indexmapbidi.lo
  CXXLD    libtesseract_ccutil.la
  CXX      src/lstm/libtesseract_lstm_la-convolve.lo
  CXX      src/lstm/libtesseract_lstm_la-fullyconnected.lo
  CXX      src/lstm/libtesseract_lstm_la-functions.lo
  CXX      src/lstm/libtesseract_lstm_la-input.lo
  CXX      src/lstm/libtesseract_lstm_la-lstm.lo
  CXX      src/lstm/libtesseract_lstm_la-lstmrecognizer.lo
  CXX      src/lstm/libtesseract_lstm_la-maxpool.lo
  CXX      src/lstm/libtesseract_lstm_la-network.lo
  CXX      src/lstm/libtesseract_lstm_la-networkio.lo
  CXX      src/lstm/libtesseract_lstm_la-parallel.lo
  CXX      src/lstm/libtesseract_lstm_la-plumbing.lo
  CXX      src/lstm/libtesseract_lstm_la-recodebeam.lo
  CXX      src/lstm/libtesseract_lstm_la-reconfig.lo
  CXX      src/lstm/libtesseract_lstm_la-reversed.lo
  CXX      src/lstm/libtesseract_lstm_la-series.lo
  CXX      src/lstm/libtesseract_lstm_la-stridemap.lo
  CXX      src/lstm/libtesseract_lstm_la-tfnetwork.lo
  CXX      src/lstm/libtesseract_lstm_la-weightmatrix.lo
  CXXLD    libtesseract_lstm.la
  CXX      src/arch/libtesseract_native_la-dotproduct.lo
  CXXLD    libtesseract_native.la
  CXX      src/arch/libtesseract_avx_la-dotproductavx.lo
  CXXLD    libtesseract_avx.la
  CXX      src/arch/libtesseract_avx2_la-intsimdmatrixavx2.lo
  CXXLD    libtesseract_avx2.la
  CXX      src/arch/libtesseract_avx512_la-dotproductavx512.lo
  CXXLD    libtesseract_avx512.la
  CXX      src/arch/libtesseract_fma_la-dotproductfma.lo
  CXXLD    libtesseract_fma.la
  CXX      src/arch/libtesseract_sse_la-dotproductsse.lo
  CXX      src/arch/libtesseract_sse_la-intsimdmatrixsse.lo
  CXXLD    libtesseract_sse.la
  CXXLD    libtesseract.la
  CXXLD    tesseract
asciidoc -b docbook -d manpage -o - doc/combine_lang_model.1.asc | \
xsltproc --nonet -o doc/combine_lang_model.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing combine_lang_model.1
asciidoc -b docbook -d manpage -o - doc/combine_tessdata.1.asc | \
xsltproc --nonet -o doc/combine_tessdata.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing combine_tessdata.1
asciidoc -b docbook -d manpage -o - doc/dawg2wordlist.1.asc | \
xsltproc --nonet -o doc/dawg2wordlist.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing dawg2wordlist.1
asciidoc -b docbook -d manpage -o - doc/lstmeval.1.asc | \
xsltproc --nonet -o doc/lstmeval.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing lstmeval.1
asciidoc -b docbook -d manpage -o - doc/lstmtraining.1.asc | \
xsltproc --nonet -o doc/lstmtraining.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing lstmtraining.1
asciidoc -b docbook -d manpage -o - doc/merge_unicharsets.1.asc | \
xsltproc --nonet -o doc/merge_unicharsets.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing merge_unicharsets.1
asciidoc -b docbook -d manpage -o - doc/set_unicharset_properties.1.asc | \
xsltproc --nonet -o doc/set_unicharset_properties.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing set_unicharset_properties.1
asciidoc -b docbook -d manpage -o - doc/tesseract.1.asc | \
xsltproc --nonet -o doc/tesseract.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing tesseract.1
asciidoc -b docbook -d manpage -o - doc/text2image.1.asc | \
xsltproc --nonet -o doc/text2image.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing text2image.1
asciidoc -b docbook -d manpage -o - doc/unicharset.5.asc | \
xsltproc --nonet -o doc/unicharset.5 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing unicharset.5
asciidoc -b docbook -d manpage -o - doc/unicharset_extractor.1.asc | \
xsltproc --nonet -o doc/unicharset_extractor.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing unicharset_extractor.1
asciidoc -b docbook -d manpage -o - doc/wordlist2dawg.1.asc | \
xsltproc --nonet -o doc/wordlist2dawg.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing wordlist2dawg.1
asciidoc -b docbook -d manpage -o - doc/ambiguous_words.1.asc | \
xsltproc --nonet -o doc/ambiguous_words.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing ambiguous_words.1
asciidoc -b docbook -d manpage -o - doc/classifier_tester.1.asc | \
xsltproc --nonet -o doc/classifier_tester.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing classifier_tester.1
asciidoc -b docbook -d manpage -o - doc/cntraining.1.asc | \
xsltproc --nonet -o doc/cntraining.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing cntraining.1
asciidoc -b docbook -d manpage -o - doc/mftraining.1.asc | \
xsltproc --nonet -o doc/mftraining.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing mftraining.1
asciidoc -b docbook -d manpage -o - doc/shapeclustering.1.asc | \
xsltproc --nonet -o doc/shapeclustering.1 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing shapeclustering.1
asciidoc -b docbook -d manpage -o - doc/unicharambigs.5.asc | \
xsltproc --nonet -o doc/unicharambigs.5 http://docbook.sourceforge.net/release/xsl/current/manpages/docbook.xsl -
Note: Writing unicharambigs.5
make[1]: Leaving directory '/home/tra/tesseract'
Making all in tessdata
make[1]: Entering directory '/home/tra/tesseract/tessdata'
Making all in configs
make[2]: Entering directory '/home/tra/tesseract/tessdata/configs'
make[2]: Nothing to be done for 'all'.
make[2]: Leaving directory '/home/tra/tesseract/tessdata/configs'
Making all in tessconfigs
make[2]: Entering directory '/home/tra/tesseract/tessdata/tessconfigs'
make[2]: Nothing to be done for 'all'.
make[2]: Leaving directory '/home/tra/tesseract/tessdata/tessconfigs'
make[2]: Entering directory '/home/tra/tesseract/tessdata'
make[2]: Nothing to be done for 'all-am'.
make[2]: Leaving directory '/home/tra/tesseract/tessdata'
make[1]: Leaving directory '/home/tra/tesseract/tessdata'
Making install in .
make[1]: Entering directory '/home/tra/tesseract'
make[2]: Entering directory '/home/tra/tesseract'
 /usr/bin/mkdir -p '/usr/local/lib'
 /bin/bash ./libtool   --mode=install /usr/bin/install -c   libtesseract.la '/usr/local/lib'
libtool: install: /usr/bin/install -c .libs/libtesseract.so.5.0.1 /usr/local/lib/libtesseract.so.5.0.1
/usr/bin/install: cannot remove '/usr/local/lib/libtesseract.so.5.0.1': Permission denied
make[2]: *** [Makefile:3203: install-libLTLIBRARIES] Error 1
make[2]: Leaving directory '/home/tra/tesseract'
make[1]: *** [Makefile:9060: install-am] Error 2
make[1]: Leaving directory '/home/tra/tesseract'
make: *** [Makefile:8166: install-recursive] Error 1
```

Training tools can be built and installed with:

```shell
make training && make training-install
```
Download tesseract trained model data
```shell
cd /usr/local/share/tessdata 
```
```shell
git clone https://github.com/tesseract-ocr/tessdata_best
```

## OCR with Tesseract

Run this command

```shell
tesseract test.png stdout --oem 1 --psm 7 -l mya
```

Result is

```
UZN file test loaded.
လွှတ်တော် အစည်းအဝေးသို့ တင ရောက်လာကြသောံ.ပြည်နယ်-ကောင်စိ အ့မတ်များစါ `
```

```shell
tesseract --help
```

```
Usage:
  tesseract --help | --help-extra | --version
  tesseract --list-langs
  tesseract imagename outputbase [options...] [configfile...]

OCR options:
  -l LANG[+LANG]        Specify language(s) used for OCR.
NOTE: These options must occur before any configfile.

Single options:
  --help                Show this help message.
  --help-extra          Show extra help for advanced users.
  --version             Show version information.
  --list-langs          List available languages for tesseract engine.
```

Tesseract parameters

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

## Training 

Clone tesstrain repository for Makefile

```shell
git clone https://github.com/tesseract-ocr/tesstrain
```

Change Directory

```shell
cd tesstrain
```

Setup

```shell
make leptonica tesseract
```
You can run

```shell
make help
```

to see

```
  Targets

    unicharset       Create unicharset
    lists            Create lists of lstmf filenames for training and eval
    training         Start training
    traineddata      Create best and fast .traineddata files from each .checkpoint file
    proto-model      Build the proto model
    leptonica        Build leptonica
    tesseract        Build tesseract
    tesseract-langs  Download tesseract-langs
    clean            Clean all generated files

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

### Dataset preparation

Dataset folder

```shell
mkdir ./data/<MODEL-NAME>-ground-truth
```

- Image files -> .tif or .png 
- Text files -> .gt.txt extensions
- tif နဲ့ tiff အတူတူ ဆို​ပေမဲ့ ဒီမှာ tiff မရ tif ပဲ
- .gt.txt နဲ့ ဆုံးရမယ် 
- Image file name နဲ့ txt file name တူရပါမယ်။ box file နဲ့ lstmf ဆိုတဲ့ file ​တွေ ထုတ်လို့ ဖြစ်ပါတယ်။
- Variable ​တွေ ​ပြောင်း Tune ကြည့်ရဦးမယ်။ ခုက အကုန် default mode ​တွေချည်းပဲ။

```shell
make training MODEL_NAME=mmfoo START_MODEL=mya PSM=7 TESSDATA=/usr/local/share/tessdata/tessdata_best
```

```
find -L data/mmfoo-ground-truth -name '*.gt.txt' | xargs paste -s > "data/mmfoo/all-gt"
combine_tessdata -u /home/tra/Desktop/HTR-ThuraAung-March-2022/htr-segmentation/tessOCR/TesseractOCRMM/tessdata/LSTM/mya.traineddata  data/mya/mmfoo
Extracting tessdata components from /home/tra/Desktop/HTR-ThuraAung-March-2022/htr-segmentation/tessOCR/TesseractOCRMM/tessdata/LSTM/mya.traineddata
Wrote data/mya/mmfoo.lstm
Wrote data/mya/mmfoo.lstm-punc-dawg
Wrote data/mya/mmfoo.lstm-word-dawg
Wrote data/mya/mmfoo.lstm-number-dawg
Wrote data/mya/mmfoo.lstm-unicharset
Wrote data/mya/mmfoo.lstm-recoder
Wrote data/mya/mmfoo.version
Version string:4.00.00alpha:mya:synth20170629
17:lstm:size=11836843, offset=192
18:lstm-punc-dawg:size=3314, offset=11837035
19:lstm-word-dawg:size=3119562, offset=11840349
20:lstm-number-dawg:size=58, offset=14959911
21:lstm-unicharset:size=9730, offset=14959969
22:lstm-recoder:size=1327, offset=14969699
23:version:size=30, offset=14971026
unicharset_extractor --output_unicharset "data/mmfoo/my.unicharset" --norm_mode 2 "data/mmfoo/all-gt"
Bad box coordinates in boxfile string! ပေါက်ရွာချနေလေသည်။ ။
Extracting unicharset from plain text file data/mmfoo/all-gt
Two grapheme links in a row:0x103a 0x1039
Invalid start of Myanmar syllable:0x103a
Normalization failed for string 'ဝင်းစာချီ၊ ယခု အင်္ဂလိပ်မင်း အစိုးရ လက်ထက်မှာ။ '
Two grapheme links in a row:0x103a 0x1039
Invalid start of Myanmar syllable:0x103a
Normalization failed for string 'အင်္ဂလိပ် အစိုးရမင်းတို့က ခန့်ထားသူကောင်းပြုတော်မူ '
Wrote unicharset file data/mmfoo/my.unicharset
merge_unicharsets data/mya/mmfoo.lstm-unicharset data/mmfoo/my.unicharset  "data/mmfoo/unicharset"
Loaded unicharset of size 147 from file data/mya/mmfoo.lstm-unicharset
Loaded unicharset of size 64 from file data/mmfoo/my.unicharset
Wrote unicharset file data/mmfoo/unicharset.
Remainder of file ignored
+ tesseract data/mmfoo-ground-truth/44.tif data/mmfoo-ground-truth/44 --psm 7 lstm.train
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Page 1
find -L data/mmfoo-ground-truth -name '*.lstmf' | python3 shuffle.py 0 > "data/mmfoo/all-lstmf"
Error processing line 1 of /home/tra/miniconda3/envs/mllearn/lib/python3.7/site-packages/vision-1.0.0-py3.7-nspkg.pth:

  Traceback (most recent call last):
    File "/home/tra/miniconda3/envs/mllearn/lib/python3.7/site.py", line 168, in addpackage
      exec(line)
    File "<string>", line 1, in <module>
    File "<frozen importlib._bootstrap>", line 580, in module_from_spec
  AttributeError: 'NoneType' object has no attribute 'loader'

Remainder of file ignored
+ head -n 45 data/mmfoo/all-lstmf
+ tail -n 5 data/mmfoo/all-lstmf
combine_lang_model \
  --input_unicharset data/mmfoo/unicharset \
  --script_dir data \
  --numbers data/mmfoo/mmfoo.numbers \
  --puncs data/mmfoo/mmfoo.punc \
  --words data/mmfoo/mmfoo.wordlist \
  --output_dir data \
   \
  --lang mmfoo
Failed to read data from: data/mmfoo/mmfoo.wordlist
Failed to read data from: data/mmfoo/mmfoo.punc
Failed to read data from: data/mmfoo/mmfoo.numbers
Loaded unicharset of size 152 from file data/mmfoo/unicharset
Setting unichar properties
Setting script properties
Failed to load script unicharset from:data/Myanmar.unicharset
Failed to load script unicharset from:data/Latin.unicharset
Warning: properties incomplete for index 3 = 3
Warning: properties incomplete for index 4 = မ
Warning: properties incomplete for index 5 = ွ
Warning: properties incomplete for index 6 = ီ
Warning: properties incomplete for index 7 = ခ
Warning: properties incomplete for index 8 = ိ
Warning: properties incomplete for index 9 = ု
Warning: properties incomplete for index 10 = ဖ
Warning: properties incomplete for index 11 = ့
Warning: properties incomplete for index 12 = လ
Warning: properties incomplete for index 13 = ပ
Warning: properties incomplete for index 14 = ါ
Warning: properties incomplete for index 15 = ၂
Warning: properties incomplete for index 16 = န
Warning: properties incomplete for index 17 = ဲ
Warning: properties incomplete for index 18 = စ
Warning: properties incomplete for index 19 = ာ
Warning: properties incomplete for index 20 = း
Warning: properties incomplete for index 21 = သ
Warning: properties incomplete for index 22 = ေ
Warning: properties incomplete for index 23 = တ
Warning: properties incomplete for index 24 = ည
Warning: properties incomplete for index 25 = ်
Warning: properties incomplete for index 26 = ယ
Warning: properties incomplete for index 27 = က
Warning: properties incomplete for index 28 = ူ
Warning: properties incomplete for index 29 = ျ
Warning: properties incomplete for index 30 = ှ
Warning: properties incomplete for index 31 = င
Warning: properties incomplete for index 32 = ြ
Warning: properties incomplete for index 33 = ရ
Warning: properties incomplete for index 34 = ဆ
Warning: properties incomplete for index 35 = ္ဆ
Warning: properties incomplete for index 36 = ္ပ
Warning: properties incomplete for index 37 = ံ
Warning: properties incomplete for index 38 = 0
Warning: properties incomplete for index 39 = 9
Warning: properties incomplete for index 40 = 6
Warning: properties incomplete for index 41 = 1
Warning: properties incomplete for index 42 = 7
Warning: properties incomplete for index 43 = ဇ
Warning: properties incomplete for index 44 = ထ
Warning: properties incomplete for index 45 = ္တ
Warning: properties incomplete for index 46 = ္မ
Warning: properties incomplete for index 47 = 5
Warning: properties incomplete for index 48 = 4
Warning: properties incomplete for index 49 = 2
Warning: properties incomplete for index 50 = အ
Warning: properties incomplete for index 51 = ဗ
Warning: properties incomplete for index 52 = ဟ
Warning: properties incomplete for index 53 = ္က
Warning: properties incomplete for index 54 = ဘ
Warning: properties incomplete for index 55 = ္လ
Warning: properties incomplete for index 56 = ဝ
Warning: properties incomplete for index 57 = 8
Warning: properties incomplete for index 58 = ဧ
Warning: properties incomplete for index 59 = ္အ
Warning: properties incomplete for index 60 = ္သ
Warning: properties incomplete for index 61 = ဉ
Warning: properties incomplete for index 62 = ၇
Warning: properties incomplete for index 63 = ဦ
Warning: properties incomplete for index 64 = ္ဟ
Warning: properties incomplete for index 65 = ဥ
Warning: properties incomplete for index 66 = ္ရ
Warning: properties incomplete for index 67 = ္ခ
Warning: properties incomplete for index 68 = ဂ
Warning: properties incomplete for index 69 = ဒ
Warning: properties incomplete for index 70 = ၀
Warning: properties incomplete for index 71 = ၉
Warning: properties incomplete for index 72 = ၄
Warning: properties incomplete for index 73 = ၅
Warning: properties incomplete for index 74 = ္ထ
Warning: properties incomplete for index 75 = ႕
Warning: properties incomplete for index 76 = ႔
Warning: properties incomplete for index 77 = ဏ
Warning: properties incomplete for index 78 = ္ဘ
Warning: properties incomplete for index 79 = ႈ
Warning: properties incomplete for index 80 = ၁
Warning: properties incomplete for index 81 = ၆
Warning: properties incomplete for index 82 = ဓ
Warning: properties incomplete for index 83 = ္န
Warning: properties incomplete for index 84 = ္စ
Warning: properties incomplete for index 85 = ဋ
Warning: properties incomplete for index 86 = ္ဓ
Warning: properties incomplete for index 87 = ္ဦ
Warning: properties incomplete for index 88 = ၃
Warning: properties incomplete for index 89 = ္ဒ
Warning: properties incomplete for index 90 = ၤ
Warning: properties incomplete for index 91 = ္ဗ
Warning: properties incomplete for index 92 = ဌ
Warning: properties incomplete for index 93 = ္ယ
Warning: properties incomplete for index 94 = ၱ
Warning: properties incomplete for index 95 = .
Warning: properties incomplete for index 96 = ႆ
Warning: properties incomplete for index 97 = ၈
Warning: properties incomplete for index 98 = ၿ
Warning: properties incomplete for index 99 = ္ဖ
Warning: properties incomplete for index 100 = ဤ
Warning: properties incomplete for index 101 = ္ဂ
Warning: properties incomplete for index 102 = င်္
Warning: properties incomplete for index 103 = ္ဍ
Warning: properties incomplete for index 104 = ၢ
Warning: properties incomplete for index 105 = ဠ
Warning: properties incomplete for index 106 = ဿ
Warning: properties incomplete for index 107 = ္ဝ
Warning: properties incomplete for index 108 = ဃ
Warning: properties incomplete for index 109 = ္ဇ
Warning: properties incomplete for index 110 = ္ဌ
Warning: properties incomplete for index 111 = ၾ
Warning: properties incomplete for index 112 = ္င
Warning: properties incomplete for index 113 = ္ည
Warning: properties incomplete for index 114 = ၠ
Warning: properties incomplete for index 115 = _
Warning: properties incomplete for index 116 = «
Warning: properties incomplete for index 117 = &
Warning: properties incomplete for index 118 = '
Warning: properties incomplete for index 119 = *
Warning: properties incomplete for index 120 = ”
Warning: properties incomplete for index 121 = ;
Warning: properties incomplete for index 122 = “
Warning: properties incomplete for index 123 = [
Warning: properties incomplete for index 124 = #
Warning: properties incomplete for index 125 = >
Warning: properties incomplete for index 126 = %
Warning: properties incomplete for index 127 = {
Warning: properties incomplete for index 128 = "
Warning: properties incomplete for index 129 = )
Warning: properties incomplete for index 130 = +
Warning: properties incomplete for index 131 = !
Warning: properties incomplete for index 132 = ]
Warning: properties incomplete for index 133 = /
Warning: properties incomplete for index 134 = `
Warning: properties incomplete for index 135 = }
Warning: properties incomplete for index 136 = ,
Warning: properties incomplete for index 137 = »
Warning: properties incomplete for index 138 = =
Warning: properties incomplete for index 139 = |
Warning: properties incomplete for index 140 = ?
Warning: properties incomplete for index 141 = (
Warning: properties incomplete for index 142 = @
Warning: properties incomplete for index 143 = ~
Warning: properties incomplete for index 144 = :
Warning: properties incomplete for index 145 = <
Warning: properties incomplete for index 146 = -
Warning: properties incomplete for index 147 = ။
Warning: properties incomplete for index 148 = ၏
Warning: properties incomplete for index 149 = ၊
Warning: properties incomplete for index 150 = ၎
Warning: properties incomplete for index 151 = ၍
Config file is optional, continuing...
Failed to read data from: data/mmfoo/mmfoo.config
Null char=2
lstmtraining \
  --debug_interval 0 \
  --traineddata data/mmfoo/mmfoo.traineddata \
  --old_traineddata /home/tra/Desktop/HTR-ThuraAung-March-2022/htr-segmentation/tessOCR/TesseractOCRMM/tessdata/LSTM/mya.traineddata \
  --continue_from data/mya/mmfoo.lstm \
  --learning_rate 0.0001 \
  --model_output data/mmfoo/checkpoints/mmfoo \
  --train_listfile data/mmfoo/list.train \
  --eval_listfile data/mmfoo/list.eval \
  --max_iterations 10000 \
  --target_error_rate 0.01
Loaded file data/mya/mmfoo.lstm, unpacking...
Warning: LSTMTrainer deserialized an LSTMRecognizer!
Code range changed from 147 to 124!
Num (Extended) outputs,weights in Series:
  1,48,0,1:1, 0
Num (Extended) outputs,weights in Series:
  C3,3:9, 0
  Ft16:16, 160
Total weights = 160
  [C3,3Ft16]:16, 160
  Mp3,3:16, 0
  Lfys64:64, 20736
  Lfx96:96, 61824
  Lrx96:96, 74112
  Lfx512:512, 1247232
  Fc124:124, 63612
Total weights = 1467676
Previous null char=2 mapped to 123
Continuing from data/mya/mmfoo.lstm
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/6.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/18.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/22.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/43.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/27.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/16.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/46.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/28.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/39.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/21.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/9.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/40.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/45.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/2.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/36.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/48.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/3.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/11.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/38.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/5.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/51.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/4.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/8.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/1.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/42.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/33.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/17.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/19.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/41.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/34.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/37.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/44.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/50.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/20.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/29.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/26.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/49.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/30.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/35.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/12.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/7.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/52.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/32.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/23.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/10.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/15.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/13.lstmf
2 Percent improvement time=100, best error was 100 @ 0
At iteration 100/100/100, Mean rms=5.51%, delta=54.456%, char train=99.234%, word train=72.671%, skip ratio=0%,  New best char error = 99.234 wrote checkpoint.

2 Percent improvement time=100, best error was 99.234 @ 100
At iteration 200/200/200, Mean rms=5.234%, delta=51.286%, char train=82.046%, word train=74.167%, skip ratio=0%,  New best char error = 82.046 wrote checkpoint.

2 Percent improvement time=100, best error was 82.046 @ 200
At iteration 300/300/300, Mean rms=4.96%, delta=47.002%, char train=70.06%, word train=75.984%, skip ratio=0%,  New best char error = 70.06 wrote best model:data/mmfoo/checkpoints/mmfoo70.06_300.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 70.06 @ 300
At iteration 400/400/400, Mean rms=4.768%, delta=44.124%, char train=62.798%, word train=77.41%, skip ratio=0%,  New best char error = 62.798 wrote best model:data/mmfoo/checkpoints/mmfoo62.798_400.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 62.798 @ 400
At iteration 500/500/500, Mean rms=4.628%, delta=41.982%, char train=58.016%, word train=78.225%, skip ratio=0%,  New best char error = 58.016 wrote best model:data/mmfoo/checkpoints/mmfoo58.016_500.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 58.016 @ 500
At iteration 600/600/600, Mean rms=4.522%, delta=40.426%, char train=54.873%, word train=78.426%, skip ratio=0%,  New best char error = 54.873 wrote best model:data/mmfoo/checkpoints/mmfoo54.873_600.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 54.873 @ 600
At iteration 700/700/700, Mean rms=4.416%, delta=38.861%, char train=52.37%, word train=78.185%, skip ratio=0%,  New best char error = 52.37 wrote best model:data/mmfoo/checkpoints/mmfoo52.37_700.checkpoint wrote checkpoint.

2 Percent improvement time=200, best error was 54.873 @ 600
At iteration 800/800/800, Mean rms=4.325%, delta=37.479%, char train=50.587%, word train=78.043%, skip ratio=0%,  New best char error = 50.587 wrote best model:data/mmfoo/checkpoints/mmfoo50.587_800.checkpoint wrote checkpoint.

2 Percent improvement time=200, best error was 52.37 @ 700
At iteration 900/900/900, Mean rms=4.246%, delta=36.269%, char train=49.119%, word train=78.04%, skip ratio=0%,  New best char error = 49.119 wrote checkpoint.

2 Percent improvement time=200, best error was 50.587 @ 800
At iteration 1000/1000/1000, Mean rms=4.18%, delta=35.319%, char train=48.293%, word train=78.035%, skip ratio=0%,  New best char error = 48.293 wrote best model:data/mmfoo/checkpoints/mmfoo48.293_1000.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 48.293 @ 1000
At iteration 1100/1100/1100, Mean rms=3.976%, delta=32.367%, char train=42.061%, word train=78.505%, skip ratio=0%,  New best char error = 42.061 wrote best model:data/mmfoo/checkpoints/mmfoo42.061_1100.checkpoint wrote checkpoint.

2 Percent improvement time=100, best error was 42.061 @ 1100
At iteration 1200/1200/1200, Mean rms=3.842%, delta=30.368%, char train=39.672%, word train=78.735%, skip ratio=0%,  New best char error = 39.672 wrote best model:data/mmfoo/checkpoints/mmfoo39.672_1200.checkpoint wrote checkpoint.

At iteration 1300/1300/1300, Mean rms=3.792%, delta=29.805%, char train=40.063%, word train=78.903%, skip ratio=0%,  New worst char error = 40.063 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=4.065%, delta=35.08%, char train=47.493%, word train=80.203%, skip ratio=0%,  New worst char error = 47.493 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=4.375%, delta=40.007%, char train=59.152%, word train=82.054%, skip ratio=0%,  New worst char error = 59.152 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=4.514%, delta=41.008%, char train=66.088%, word train=84.111%, skip ratio=0%,  New worst char error = 66.088 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=4.601%, delta=40.696%, char train=71.822%, word train=86.437%, skip ratio=0%,  New worst char error = 71.822 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=4.668%, delta=39.976%, char train=76.773%, word train=88.732%, skip ratio=0%,  New worst char error = 76.773 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=4.726%, delta=39.137%, char train=81.897%, word train=90.931%, skip ratio=0%,  New worst char error = 81.897 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=4.782%, delta=38.214%, char train=86.38%, word train=93.131%, skip ratio=0%,  New worst char error = 86.38 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=4.839%, delta=37.261%, char train=91.176%, word train=95.394%, skip ratio=0%,  New worst char error = 91.176 wrote checkpoint.

Layer 2=ConvNL: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
Layer 4=Lfys64: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
Layer 5=Lfx96: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
Layer 6=Lrx96: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
Layer 7=Lfx512: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
Layer 8=Output: lr 6.25e-05->-nan%, lr 8.83884e-05->-nan% SAME
At iteration 2200/2200/2200, Mean rms=4.873%, delta=35.818%, char train=95.547%, word train=97.599%, skip ratio=0%,  New worst char error = 95.547
Divergence! Reverted to iteration 1200/1200/1200
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1300/1300/1300, Mean rms=3.784%, delta=29.712%, char train=39.767%, word train=78.784%, skip ratio=0%,  New worst char error = 39.767 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.782%, delta=29.859%, char train=41.233%, word train=78.974%, skip ratio=0%,  New worst char error = 41.233 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=4.12%, delta=36.216%, char train=50.927%, word train=80.703%, skip ratio=0%,  New worst char error = 50.927 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=4.529%, delta=43.378%, char train=61.988%, word train=82.76%, skip ratio=0%,  New worst char error = 61.988 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=4.754%, delta=46.224%, char train=69.096%, word train=85.086%, skip ratio=0%,  New worst char error = 69.096 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=4.932%, delta=47.818%, char train=76.068%, word train=87.382%, skip ratio=0%,  New worst char error = 76.068 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=5.073%, delta=48.539%, char train=82.854%, word train=89.58%, skip ratio=0%,  New worst char error = 82.854 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=5.181%, delta=48.473%, char train=88.621%, word train=91.781%, skip ratio=0%,  New worst char error = 88.621 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=5.271%, delta=48.034%, char train=94.203%, word train=94.044%, skip ratio=0%,  New worst char error = 94.203 wrote checkpoint.

Layer 2=ConvNL: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
Layer 4=Lfys64: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
Layer 5=Lfx96: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
Layer 6=Lrx96: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
Layer 7=Lfx512: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
Layer 8=Output: lr 4.41942e-05->-nan%, lr 6.25e-05->-nan% SAME
At iteration 2200/2200/2200, Mean rms=5.337%, delta=47.191%, char train=98.918%, word train=96.248%, skip ratio=0%,  New worst char error = 98.918
Divergence! Reverted to iteration 1200/1200/1200
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1300/1300/1300, Mean rms=3.782%, delta=29.637%, char train=39.764%, word train=78.841%, skip ratio=0%,  New worst char error = 39.764 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.768%, delta=29.596%, char train=40.968%, word train=78.994%, skip ratio=0%,  New worst char error = 40.968 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.816%, delta=30.493%, char train=43.751%, word train=79.641%, skip ratio=0%,  New worst char error = 43.751 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=4.22%, delta=38.167%, char train=54.52%, word train=81.638%, skip ratio=0%,  New worst char error = 54.52 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=4.77%, delta=48.286%, char train=68.182%, word train=83.964%, skip ratio=0%,  New worst char error = 68.182 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=5.179%, delta=55.15%, char train=79.448%, word train=86.259%, skip ratio=0%,  New worst char error = 79.448 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=5.448%, delta=58.945%, char train=89.393%, word train=88.458%, skip ratio=0%,  New worst char error = 89.393 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=5.658%, delta=61.254%, char train=96.732%, word train=90.658%, skip ratio=0%,  New worst char error = 96.732 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=5.844%, delta=62.937%, char train=103.588%, word train=92.921%, skip ratio=0%,  New worst char error = 103.588 wrote checkpoint.

Layer 2=ConvNL: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
Layer 4=Lfys64: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
Layer 5=Lfx96: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
Layer 6=Lrx96: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
Layer 7=Lfx512: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
Layer 8=Output: lr 3.125e-05->-nan%, lr 4.41942e-05->-nan% SAME
At iteration 2200/2200/2200, Mean rms=5.986%, delta=63.557%, char train=109.659%, word train=95.126%, skip ratio=0%,  New worst char error = 109.659
Divergence! Reverted to iteration 1200/1200/1200
Reduced learning rate on layers: 6
 wrote checkpoint.

2 Percent improvement time=200, best error was 42.061 @ 1100
At iteration 1300/1300/1300, Mean rms=3.778%, delta=29.606%, char train=39.619%, word train=78.675%, skip ratio=0%,  New best char error = 39.619 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.751%, delta=29.375%, char train=40.488%, word train=78.577%, skip ratio=0%,  New worst char error = 40.488 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.756%, delta=29.647%, char train=42.558%, word train=78.868%, skip ratio=0%,  New worst char error = 42.558 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.787%, delta=30.215%, char train=44.972%, word train=79.591%, skip ratio=0%,  New worst char error = 44.972 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=4.001%, delta=34.058%, char train=50.208%, word train=81.304%, skip ratio=0%,  New worst char error = 50.208 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=4.514%, delta=43.801%, char train=62.386%, word train=83.599%, skip ratio=0%,  New worst char error = 62.386 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=5.09%, delta=54.498%, char train=76.81%, word train=85.798%, skip ratio=0%,  New worst char error = 76.81 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=5.577%, delta=63.059%, char train=90.072%, word train=87.998%, skip ratio=0%,  New worst char error = 90.072 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=5.998%, delta=70.004%, char train=100.841%, word train=90.261%, skip ratio=0%,  New worst char error = 100.841 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=6.3%, delta=74.423%, char train=109.981%, word train=92.466%, skip ratio=0%,  New worst char error = 109.981 wrote checkpoint.

Layer 2=ConvNL: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
Layer 4=Lfys64: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
Layer 5=Lfx96: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
Layer 6=Lrx96: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
Layer 7=Lfx512: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
Layer 8=Output: lr 2.20971e-05->-nan%, lr 3.125e-05->-nan% SAME
At iteration 2300/2300/2300, Mean rms=6.534%, delta=77.42%, char train=117.754%, word train=94.564%, skip ratio=0%,  New worst char error = 117.754
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.752%, delta=29.378%, char train=40.429%, word train=78.555%, skip ratio=0%,  New worst char error = 40.429 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.753%, delta=29.583%, char train=42.384%, word train=78.722%, skip ratio=0%,  New worst char error = 42.384 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.755%, delta=29.711%, char train=44.015%, word train=79.322%, skip ratio=0%,  New worst char error = 44.015 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.807%, delta=30.593%, char train=46.205%, word train=80.162%, skip ratio=0%,  New worst char error = 46.205 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.945%, delta=33.065%, char train=49.512%, word train=81.528%, skip ratio=0%,  New worst char error = 49.512 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=4.316%, delta=39.909%, char train=58.833%, word train=83.601%, skip ratio=0%,  New worst char error = 58.833 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=4.84%, delta=49.757%, char train=71.877%, word train=85.801%, skip ratio=0%,  New worst char error = 71.877 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=5.44%, delta=61.029%, char train=85.278%, word train=88.065%, skip ratio=0%,  New worst char error = 85.278 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=6.003%, delta=71.376%, char train=98.865%, word train=90.269%, skip ratio=0%,  New worst char error = 98.865 wrote checkpoint.

Layer 2=ConvNL: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
Layer 4=Lfys64: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
Layer 5=Lfx96: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
Layer 6=Lrx96: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
Layer 7=Lfx512: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
Layer 8=Output: lr 1.5625e-05->-nan%, lr 2.20971e-05->-nan% SAME
At iteration 2300/2300/2300, Mean rms=6.496%, delta=79.937%, char train=112.321%, word train=92.367%, skip ratio=0%,  New worst char error = 112.321
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.747%, delta=29.299%, char train=40.36%, word train=78.461%, skip ratio=0%,  New worst char error = 40.36 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.736%, delta=29.314%, char train=41.812%, word train=78.561%, skip ratio=0%,  New worst char error = 41.812 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.727%, delta=29.342%, char train=43.127%, word train=78.828%, skip ratio=0%,  New worst char error = 43.127 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.757%, delta=29.919%, char train=45.14%, word train=79.553%, skip ratio=0%,  New worst char error = 45.14 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.807%, delta=30.761%, char train=47.182%, word train=80.492%, skip ratio=0%,  New worst char error = 47.182 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.891%, delta=32.157%, char train=49.895%, word train=81.377%, skip ratio=0%,  New worst char error = 49.895 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=4.028%, delta=34.412%, char train=53.331%, word train=82.536%, skip ratio=0%,  New worst char error = 53.331 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=4.299%, delta=39.218%, char train=59.22%, word train=84.453%, skip ratio=0%,  New worst char error = 59.22 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=4.697%, delta=46.488%, char train=68.579%, word train=86.607%, skip ratio=0%,  New worst char error = 68.579 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=5.17%, delta=55.213%, char train=80.545%, word train=88.693%, skip ratio=0%,  New worst char error = 80.545 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/24.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/14.lstmf
Loaded 1/1 lines (1-1) of document data/mmfoo-ground-truth/31.lstmf
Layer 2=ConvNL: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
Layer 4=Lfys64: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
Layer 5=Lfx96: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
Layer 6=Lrx96: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
Layer 7=Lfx512: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
Layer 8=Output: lr 1.10485e-05->-nan%, lr 1.5625e-05->-nan% SAME
At iteration 2400/2400/2400, Mean rms=5.695%, delta=65.124%, char train=94.574%, word train=90.738%, skip ratio=0%,  New worst char error = 94.574
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.746%, delta=29.288%, char train=40.333%, word train=78.426%, skip ratio=0%,  New worst char error = 40.333 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.731%, delta=29.209%, char train=41.693%, word train=78.442%, skip ratio=0%,  New worst char error = 41.693 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.718%, delta=29.203%, char train=42.906%, word train=78.621%, skip ratio=0%,  New worst char error = 42.906 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.737%, delta=29.67%, char train=44.51%, word train=79.147%, skip ratio=0%,  New worst char error = 44.51 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.773%, delta=30.367%, char train=46.125%, word train=79.912%, skip ratio=0%,  New worst char error = 46.125 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.823%, delta=31.282%, char train=48.172%, word train=80.668%, skip ratio=0%,  New worst char error = 48.172 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.879%, delta=32.177%, char train=50.167%, word train=81.547%, skip ratio=0%,  New worst char error = 50.167 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.959%, delta=33.413%, char train=52.262%, word train=82.413%, skip ratio=0%,  New worst char error = 52.262 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=4.046%, delta=34.777%, char train=54.25%, word train=83.326%, skip ratio=0%,  New worst char error = 54.25 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=4.143%, delta=36.332%, char train=56.402%, word train=84.352%, skip ratio=0%,  New worst char error = 56.402 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=4.294%, delta=38.819%, char train=59.884%, word train=85.843%, skip ratio=0%,  New worst char error = 59.884At iteration 2300, stage 0, Eval Char error rate=190.36031, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2500/2500/2500, Mean rms=4.559%, delta=43.494%, char train=65%, word train=87.428%, skip ratio=0%,  New worst char error = 65At iteration 2300, stage 0, Eval Char error rate=97.336617, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=4.913%, delta=49.858%, char train=72.239%, word train=89.212%, skip ratio=0%,  New worst char error = 72.239At iteration 2400, stage 0, Eval Char error rate=109.86258, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=5.331%, delta=57.652%, char train=81.149%, word train=91.012%, skip ratio=0%,  New worst char error = 81.149At iteration 2500, stage 0, Eval Char error rate=134.10777, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
Layer 4=Lfys64: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
Layer 5=Lfx96: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
Layer 6=Lrx96: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
Layer 7=Lfx512: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
Layer 8=Output: lr 7.8125e-06->-nan%, lr 1.10485e-05->-nan% SAME
At iteration 2800/2800/2800, Mean rms=5.79%, delta=66.294%, char train=92.058%, word train=92.542%, skip ratio=0%,  New worst char error = 92.058At iteration 2600, stage 0, Eval Char error rate=144.52383, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.744%, delta=29.241%, char train=40.244%, word train=78.426%, skip ratio=0%,  New worst char error = 40.244 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.726%, delta=29.168%, char train=41.644%, word train=78.367%, skip ratio=0%,  New worst char error = 41.644 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.71%, delta=29.081%, char train=42.631%, word train=78.436%, skip ratio=0%,  New worst char error = 42.631 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.723%, delta=29.484%, char train=44.054%, word train=78.807%, skip ratio=0%,  New worst char error = 44.054 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.757%, delta=30.154%, char train=45.473%, word train=79.395%, skip ratio=0%,  New worst char error = 45.473 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.799%, delta=30.965%, char train=47.415%, word train=79.934%, skip ratio=0%,  New worst char error = 47.415 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.837%, delta=31.626%, char train=49.003%, word train=80.524%, skip ratio=0%,  New worst char error = 49.003 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.896%, delta=32.632%, char train=50.809%, word train=81.213%, skip ratio=0%,  New worst char error = 50.809 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.945%, delta=33.407%, char train=52.205%, word train=81.956%, skip ratio=0%,  New worst char error = 52.205 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.987%, delta=33.992%, char train=53.465%, word train=82.75%, skip ratio=0%,  New worst char error = 53.465 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=4.03%, delta=34.603%, char train=55.333%, word train=83.532%, skip ratio=0%,  New worst char error = 55.333At iteration 2700, stage 0, Eval Char error rate=161.65226, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2500/2500/2500, Mean rms=4.073%, delta=35.187%, char train=56.093%, word train=84.117%, skip ratio=0%,  New worst char error = 56.093At iteration 2300, stage 0, Eval Char error rate=79.369325, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=4.153%, delta=36.519%, char train=57.768%, word train=84.659%, skip ratio=0%,  New worst char error = 57.768At iteration 2400, stage 0, Eval Char error rate=91.001652, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=4.239%, delta=37.807%, char train=59.465%, word train=85.494%, skip ratio=0%,  New worst char error = 59.465At iteration 2500, stage 0, Eval Char error rate=93.453479, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=4.365%, delta=39.957%, char train=62.721%, word train=86.33%, skip ratio=0%,  New worst char error = 62.721At iteration 2600, stage 0, Eval Char error rate=102.47178, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=4.533%, delta=42.854%, char train=65.843%, word train=87.442%, skip ratio=0%,  New worst char error = 65.843At iteration 2700, stage 0, Eval Char error rate=99.213878, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3000/3000/3000, Mean rms=4.785%, delta=47.264%, char train=70.342%, word train=88.82%, skip ratio=0%,  New worst char error = 70.342At iteration 2800, stage 0, Eval Char error rate=100.1603, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=5.086%, delta=52.694%, char train=76.188%, word train=90.298%, skip ratio=0%,  New worst char error = 76.188At iteration 2900, stage 0, Eval Char error rate=113.18726, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=5.43%, delta=59.022%, char train=84.044%, word train=91.71%, skip ratio=0%,  New worst char error = 84.044At iteration 3000, stage 0, Eval Char error rate=121.6601, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
Layer 4=Lfys64: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
Layer 5=Lfx96: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
Layer 6=Lrx96: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
Layer 7=Lfx512: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
Layer 8=Output: lr 5.52427e-06->-nan%, lr 7.8125e-06->-nan% SAME
At iteration 3300/3300/3300, Mean rms=5.808%, delta=66.169%, char train=93.449%, word train=92.98%, skip ratio=0%,  New worst char error = 93.449At iteration 3100, stage 0, Eval Char error rate=160.82979, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.743%, delta=29.227%, char train=40.265%, word train=78.426%, skip ratio=0%,  New worst char error = 40.265 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.725%, delta=29.152%, char train=41.59%, word train=78.342%, skip ratio=0%,  New worst char error = 41.59 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.705%, delta=28.996%, char train=42.474%, word train=78.375%, skip ratio=0%,  New worst char error = 42.474 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.713%, delta=29.322%, char train=43.782%, word train=78.66%, skip ratio=0%,  New worst char error = 43.782 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.738%, delta=29.862%, char train=45.083%, word train=79.151%, skip ratio=0%,  New worst char error = 45.083 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.774%, delta=30.589%, char train=46.821%, word train=79.566%, skip ratio=0%,  New worst char error = 46.821 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.812%, delta=31.302%, char train=48.206%, word train=79.931%, skip ratio=0%,  New worst char error = 48.206 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.864%, delta=32.251%, char train=49.705%, word train=80.315%, skip ratio=0%,  New worst char error = 49.705 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.902%, delta=32.846%, char train=50.776%, word train=80.82%, skip ratio=0%,  New worst char error = 50.776 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.927%, delta=33.219%, char train=51.497%, word train=81.327%, skip ratio=0%,  New worst char error = 51.497 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=3.953%, delta=33.634%, char train=52.718%, word train=81.927%, skip ratio=0%,  New worst char error = 52.718At iteration 3200, stage 0, Eval Char error rate=168.06606, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2500/2500/2500, Mean rms=3.966%, delta=33.776%, char train=53.051%, word train=82.404%, skip ratio=0%,  New worst char error = 53.051At iteration 2300, stage 0, Eval Char error rate=91.582408, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=4%, delta=34.318%, char train=53.828%, word train=82.888%, skip ratio=0%,  New worst char error = 53.828At iteration 2400, stage 0, Eval Char error rate=88.261664, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=4.028%, delta=34.605%, char train=54.443%, word train=83.56%, skip ratio=0%,  New worst char error = 54.443At iteration 2500, stage 0, Eval Char error rate=83.523433, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=4.055%, delta=34.935%, char train=55.597%, word train=84.106%, skip ratio=0%,  New worst char error = 55.597At iteration 2600, stage 0, Eval Char error rate=87.417216, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=4.08%, delta=35.236%, char train=56.21%, word train=84.565%, skip ratio=0%,  New worst char error = 56.21At iteration 2700, stage 0, Eval Char error rate=83.128532, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3000/3000/3000, Mean rms=4.117%, delta=35.716%, char train=56.578%, word train=85.033%, skip ratio=0%,  New worst char error = 56.578At iteration 2800, stage 0, Eval Char error rate=84.057825, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=4.159%, delta=36.33%, char train=57.368%, word train=85.621%, skip ratio=0%,  New worst char error = 57.368At iteration 2900, stage 0, Eval Char error rate=88.111111, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=4.22%, delta=37.269%, char train=58.657%, word train=86.151%, skip ratio=0%,  New worst char error = 58.657At iteration 3000, stage 0, Eval Char error rate=89.739402, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3300/3300/3300, Mean rms=4.292%, delta=38.417%, char train=60.734%, word train=86.682%, skip ratio=0%,  New worst char error = 60.734At iteration 3100, stage 0, Eval Char error rate=97.108518, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3400/3400/3400, Mean rms=4.359%, delta=39.515%, char train=61.674%, word train=87.08%, skip ratio=0%,  New worst char error = 61.674At iteration 3200, stage 0, Eval Char error rate=95.18529, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3500/3500/3500, Mean rms=4.473%, delta=41.406%, char train=63.782%, word train=87.485%, skip ratio=0%,  New worst char error = 63.782At iteration 3300, stage 0, Eval Char error rate=99.186168, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3600/3600/3600, Mean rms=4.615%, delta=43.728%, char train=66.231%, word train=88.383%, skip ratio=0%,  New worst char error = 66.231At iteration 3400, stage 0, Eval Char error rate=102.35674, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3700/3700/3700, Mean rms=4.817%, delta=47.405%, char train=70.382%, word train=89.402%, skip ratio=0%,  New worst char error = 70.382At iteration 3500, stage 0, Eval Char error rate=100.83445, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3800/3800/3800, Mean rms=5.051%, delta=51.634%, char train=74.701%, word train=90.402%, skip ratio=0%,  New worst char error = 74.701At iteration 3600, stage 0, Eval Char error rate=110.57551, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3900/3900/3900, Mean rms=5.312%, delta=56.453%, char train=79.465%, word train=91.6%, skip ratio=0%,  New worst char error = 79.465At iteration 3700, stage 0, Eval Char error rate=93.595416, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4000/4000/4000, Mean rms=5.605%, delta=61.882%, char train=85.801%, word train=92.916%, skip ratio=0%,  New worst char error = 85.801At iteration 3800, stage 0, Eval Char error rate=119.98977, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
Layer 4=Lfys64: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
Layer 5=Lfx96: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
Layer 6=Lrx96: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
Layer 7=Lfx512: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
Layer 8=Output: lr 3.90625e-06->-nan%, lr 5.52427e-06->-nan% SAME
At iteration 4100/4100/4100, Mean rms=5.935%, delta=68.152%, char train=93.328%, word train=94.175%, skip ratio=0%,  New worst char error = 93.328At iteration 3900, stage 0, Eval Char error rate=130.66027, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.743%, delta=29.222%, char train=40.262%, word train=78.426%, skip ratio=0%,  New worst char error = 40.262 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.722%, delta=29.115%, char train=41.59%, word train=78.303%, skip ratio=0%,  New worst char error = 41.59 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.698%, delta=28.883%, char train=42.394%, word train=78.252%, skip ratio=0%,  New worst char error = 42.394 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.703%, delta=29.159%, char train=43.62%, word train=78.52%, skip ratio=0%,  New worst char error = 43.62 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.727%, delta=29.702%, char train=44.857%, word train=78.975%, skip ratio=0%,  New worst char error = 44.857 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.759%, delta=30.393%, char train=46.373%, word train=79.375%, skip ratio=0%,  New worst char error = 46.373 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.788%, delta=30.945%, char train=47.404%, word train=79.596%, skip ratio=0%,  New worst char error = 47.404 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.832%, delta=31.75%, char train=48.745%, word train=79.878%, skip ratio=0%,  New worst char error = 48.745 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.864%, delta=32.279%, char train=49.848%, word train=80.203%, skip ratio=0%,  New worst char error = 49.848 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.885%, delta=32.608%, char train=50.486%, word train=80.566%, skip ratio=0%,  New worst char error = 50.486 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=3.901%, delta=32.853%, char train=51.423%, word train=80.929%, skip ratio=0%,  New worst char error = 51.423At iteration 4000, stage 0, Eval Char error rate=162.66135, Word error rate=100 wrote checkpoint.

At iteration 2500/2500/2500, Mean rms=3.903%, delta=32.847%, char train=51.387%, word train=81.191%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=3.929%, delta=33.307%, char train=52.067%, word train=81.566%, skip ratio=0%,  New worst char error = 52.067At iteration 2300, stage 0, Eval Char error rate=91.041157, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=3.949%, delta=33.535%, char train=52.471%, word train=82.109%, skip ratio=0%,  New worst char error = 52.471At iteration 2400, stage 0, Eval Char error rate=90.370161, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=3.963%, delta=33.733%, char train=53.304%, word train=82.462%, skip ratio=0%,  New worst char error = 53.304At iteration 2600, stage 0, Eval Char error rate=83.519167, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=3.973%, delta=33.811%, char train=53.854%, word train=82.833%, skip ratio=0%,  New worst char error = 53.854At iteration 2700, stage 0, Eval Char error rate=92.201644, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3000/3000/3000, Mean rms=3.992%, delta=34.068%, char train=54.215%, word train=83.279%, skip ratio=0%,  New worst char error = 54.215At iteration 2800, stage 0, Eval Char error rate=84.908714, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=4.012%, delta=34.299%, char train=54.708%, word train=83.816%, skip ratio=0%,  New worst char error = 54.708At iteration 2900, stage 0, Eval Char error rate=87.604241, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=4.035%, delta=34.607%, char train=55.16%, word train=84.404%, skip ratio=0%,  New worst char error = 55.16At iteration 3000, stage 0, Eval Char error rate=82.466194, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3300/3300/3300, Mean rms=4.056%, delta=34.89%, char train=56.194%, word train=84.837%, skip ratio=0%,  New worst char error = 56.194At iteration 3100, stage 0, Eval Char error rate=89.678315, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3400/3400/3400, Mean rms=4.068%, delta=34.988%, char train=56.392%, word train=85.21%, skip ratio=0%,  New worst char error = 56.392At iteration 3200, stage 0, Eval Char error rate=90.259552, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3500/3500/3500, Mean rms=4.102%, delta=35.487%, char train=57.174%, word train=85.47%, skip ratio=0%,  New worst char error = 57.174At iteration 3300, stage 0, Eval Char error rate=86.789322, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3600/3600/3600, Mean rms=4.131%, delta=35.815%, char train=57.796%, word train=85.932%, skip ratio=0%,  New worst char error = 57.796At iteration 3400, stage 0, Eval Char error rate=81.792982, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3700/3700/3700, Mean rms=4.163%, delta=36.26%, char train=58.652%, word train=86.199%, skip ratio=0%,  New worst char error = 58.652At iteration 3500, stage 0, Eval Char error rate=79.684945, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3800/3800/3800, Mean rms=4.189%, delta=36.606%, char train=58.918%, word train=86.462%, skip ratio=0%,  New worst char error = 58.918At iteration 3600, stage 0, Eval Char error rate=86.284271, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3900/3900/3900, Mean rms=4.222%, delta=37.087%, char train=59.084%, word train=86.528%, skip ratio=0%,  New worst char error = 59.084At iteration 3700, stage 0, Eval Char error rate=91.27417, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4000/4000/4000, Mean rms=4.265%, delta=37.735%, char train=59.652%, word train=86.652%, skip ratio=0%,  New worst char error = 59.652At iteration 3800, stage 0, Eval Char error rate=88.172198, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4100/4100/4100, Mean rms=4.312%, delta=38.428%, char train=60.52%, word train=86.935%, skip ratio=0%,  New worst char error = 60.52At iteration 3900, stage 0, Eval Char error rate=87.804818, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4200/4200/4200, Mean rms=4.372%, delta=39.428%, char train=62.077%, word train=87.034%, skip ratio=0%,  New worst char error = 62.077At iteration 4000, stage 0, Eval Char error rate=91.602213, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4300/4300/4300, Mean rms=4.427%, delta=40.327%, char train=62.69%, word train=87.182%, skip ratio=0%,  New worst char error = 62.69At iteration 4100, stage 0, Eval Char error rate=95.029822, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4400/4400/4400, Mean rms=4.516%, delta=41.977%, char train=64.113%, word train=87.329%, skip ratio=0%,  New worst char error = 64.113At iteration 4200, stage 0, Eval Char error rate=99.786729, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4500/4500/4500, Mean rms=4.61%, delta=43.537%, char train=65.924%, word train=87.914%, skip ratio=0%,  New worst char error = 65.924At iteration 4300, stage 0, Eval Char error rate=99.434469, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4600/4600/4600, Mean rms=4.736%, delta=45.826%, char train=68.441%, word train=88.474%, skip ratio=0%,  New worst char error = 68.441At iteration 4400, stage 0, Eval Char error rate=92.459335, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4700/4700/4700, Mean rms=4.888%, delta=48.642%, char train=71.411%, word train=89.237%, skip ratio=0%,  New worst char error = 71.411At iteration 4500, stage 0, Eval Char error rate=106.31382, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4800/4800/4800, Mean rms=5.065%, delta=51.878%, char train=74.411%, word train=90.066%, skip ratio=0%,  New worst char error = 74.411At iteration 4600, stage 0, Eval Char error rate=126.46519, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4900/4900/4900, Mean rms=5.272%, delta=55.641%, char train=78.019%, word train=91.186%, skip ratio=0%,  New worst char error = 78.019At iteration 4700, stage 0, Eval Char error rate=110.42755, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5000/5000/5000, Mean rms=5.498%, delta=59.878%, char train=82.673%, word train=92.462%, skip ratio=0%,  New worst char error = 82.673At iteration 4800, stage 0, Eval Char error rate=121.64707, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5100/5100/5100, Mean rms=5.741%, delta=64.454%, char train=88.102%, word train=93.543%, skip ratio=0%,  New worst char error = 88.102At iteration 4900, stage 0, Eval Char error rate=127.9619, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
Layer 4=Lfys64: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
Layer 5=Lfx96: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
Layer 6=Lrx96: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
Layer 7=Lfx512: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
Layer 8=Output: lr 2.76214e-06->-nan%, lr 3.90625e-06->-nan% SAME
At iteration 5200/5200/5200, Mean rms=5.984%, delta=68.985%, char train=92.883%, word train=94.654%, skip ratio=0%,  New worst char error = 92.883At iteration 5000, stage 0, Eval Char error rate=119.80743, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.742%, delta=29.222%, char train=40.237%, word train=78.426%, skip ratio=0%,  New worst char error = 40.237 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.722%, delta=29.118%, char train=41.568%, word train=78.289%, skip ratio=0%,  New worst char error = 41.568 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.698%, delta=28.896%, char train=42.36%, word train=78.166%, skip ratio=0%,  New worst char error = 42.36 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.701%, delta=29.147%, char train=43.449%, word train=78.386%, skip ratio=0%,  New worst char error = 43.449 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.723%, delta=29.67%, char train=44.548%, word train=78.815%, skip ratio=0%,  New worst char error = 44.548 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.752%, delta=30.29%, char train=45.853%, word train=79.112%, skip ratio=0%,  New worst char error = 45.853 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.777%, delta=30.819%, char train=46.909%, word train=79.333%, skip ratio=0%,  New worst char error = 46.909 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.819%, delta=31.605%, char train=48.09%, word train=79.59%, skip ratio=0%,  New worst char error = 48.09 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.846%, delta=32.046%, char train=48.968%, word train=79.858%, skip ratio=0%,  New worst char error = 48.968 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.863%, delta=32.351%, char train=49.656%, word train=80.168%, skip ratio=0%,  New worst char error = 49.656 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=3.876%, delta=32.549%, char train=50.264%, word train=80.438%, skip ratio=0%,  New worst char error = 50.264At iteration 5100, stage 0, Eval Char error rate=124.28921, Word error rate=100 wrote checkpoint.

At iteration 2500/2500/2500, Mean rms=3.876%, delta=32.549%, char train=50.215%, word train=80.581%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=3.899%, delta=32.997%, char train=50.611%, word train=80.848%, skip ratio=0%,  New worst char error = 50.611At iteration 2300, stage 0, Eval Char error rate=86.108121, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=3.92%, delta=33.274%, char train=51.153%, word train=81.28%, skip ratio=0%,  New worst char error = 51.153At iteration 2400, stage 0, Eval Char error rate=94.397683, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=3.928%, delta=33.407%, char train=51.808%, word train=81.501%, skip ratio=0%,  New worst char error = 51.808At iteration 2600, stage 0, Eval Char error rate=95.172386, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=3.93%, delta=33.381%, char train=52.207%, word train=81.738%, skip ratio=0%,  New worst char error = 52.207At iteration 2700, stage 0, Eval Char error rate=89.358241, Word error rate=95 wrote checkpoint.

At iteration 3000/3000/3000, Mean rms=3.941%, delta=33.545%, char train=52.166%, word train=81.971%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=3.95%, delta=33.608%, char train=52.49%, word train=82.303%, skip ratio=0%,  New worst char error = 52.49At iteration 2800, stage 0, Eval Char error rate=87.984503, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=3.967%, delta=33.847%, char train=52.846%, word train=82.7%, skip ratio=0%,  New worst char error = 52.846At iteration 2900, stage 0, Eval Char error rate=92.529958, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3300/3300/3300, Mean rms=3.976%, delta=33.957%, char train=53.623%, word train=82.865%, skip ratio=0%,  New worst char error = 53.623At iteration 3100, stage 0, Eval Char error rate=93.265784, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3400/3400/3400, Mean rms=3.977%, delta=33.952%, char train=53.67%, word train=83.094%, skip ratio=0%,  New worst char error = 53.67At iteration 3200, stage 0, Eval Char error rate=86.592969, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3500/3500/3500, Mean rms=3.99%, delta=34.107%, char train=53.941%, word train=83.269%, skip ratio=0%,  New worst char error = 53.941At iteration 3300, stage 0, Eval Char error rate=86.496769, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3600/3600/3600, Mean rms=3.995%, delta=34.025%, char train=54.198%, word train=83.728%, skip ratio=0%,  New worst char error = 54.198At iteration 3400, stage 0, Eval Char error rate=87.095803, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3700/3700/3700, Mean rms=4.004%, delta=34.165%, char train=54.629%, word train=84.003%, skip ratio=0%,  New worst char error = 54.629At iteration 3500, stage 0, Eval Char error rate=81.493109, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3800/3800/3800, Mean rms=4.008%, delta=34.185%, char train=54.853%, word train=84.18%, skip ratio=0%,  New worst char error = 54.853At iteration 3600, stage 0, Eval Char error rate=85.097727, Word error rate=95 wrote checkpoint.

At iteration 3900/3900/3900, Mean rms=4.023%, delta=34.38%, char train=54.807%, word train=84.292%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4000/4000/4000, Mean rms=4.033%, delta=34.486%, char train=55.203%, word train=84.478%, skip ratio=0%,  New worst char error = 55.203At iteration 3700, stage 0, Eval Char error rate=86.451262, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4100/4100/4100, Mean rms=4.049%, delta=34.699%, char train=55.488%, word train=84.8%, skip ratio=0%,  New worst char error = 55.488At iteration 3800, stage 0, Eval Char error rate=88.862915, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4200/4200/4200, Mean rms=4.06%, delta=34.812%, char train=56.352%, word train=84.994%, skip ratio=0%,  New worst char error = 56.352At iteration 4000, stage 0, Eval Char error rate=89.741619, Word error rate=95 wrote checkpoint.

At iteration 4300/4300/4300, Mean rms=4.062%, delta=34.773%, char train=56.074%, word train=85.288%, skip ratio=0%,  wrote checkpoint.

At iteration 4400/4400/4400, Mean rms=4.082%, delta=35.068%, char train=56.352%, word train=85.426%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4500/4500/4500, Mean rms=4.102%, delta=35.287%, char train=56.631%, word train=85.823%, skip ratio=0%,  New worst char error = 56.631At iteration 4100, stage 0, Eval Char error rate=87.828576, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4600/4600/4600, Mean rms=4.128%, delta=35.666%, char train=57.528%, word train=86.076%, skip ratio=0%,  New worst char error = 57.528At iteration 4200, stage 0, Eval Char error rate=88.749211, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4700/4700/4700, Mean rms=4.134%, delta=35.635%, char train=57.823%, word train=86.173%, skip ratio=0%,  New worst char error = 57.823At iteration 4500, stage 0, Eval Char error rate=84.07581, Word error rate=95 wrote checkpoint.

At iteration 4800/4800/4800, Mean rms=4.155%, delta=35.885%, char train=57.808%, word train=86.382%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4900/4900/4900, Mean rms=4.177%, delta=36.184%, char train=58.381%, word train=86.552%, skip ratio=0%,  New worst char error = 58.381At iteration 4600, stage 0, Eval Char error rate=94.202418, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5000/5000/5000, Mean rms=4.208%, delta=36.579%, char train=58.889%, word train=86.825%, skip ratio=0%,  New worst char error = 58.889At iteration 4700, stage 0, Eval Char error rate=93.427714, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5100/5100/5100, Mean rms=4.245%, delta=37.21%, char train=59.873%, word train=86.915%, skip ratio=0%,  New worst char error = 59.873At iteration 4900, stage 0, Eval Char error rate=95.852709, Word error rate=95 wrote checkpoint.

At iteration 5200/5200/5200, Mean rms=4.269%, delta=37.62%, char train=59.783%, word train=86.997%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5300/5300/5300, Mean rms=4.318%, delta=38.424%, char train=60.394%, word train=87.004%, skip ratio=0%,  New worst char error = 60.394At iteration 5000, stage 0, Eval Char error rate=87.611352, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5400/5400/5400, Mean rms=4.363%, delta=39.154%, char train=61.423%, word train=87.325%, skip ratio=0%,  New worst char error = 61.423At iteration 5100, stage 0, Eval Char error rate=85.445114, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5500/5500/5500, Mean rms=4.425%, delta=40.34%, char train=62.668%, word train=87.455%, skip ratio=0%,  New worst char error = 62.668At iteration 5300, stage 0, Eval Char error rate=92.97595, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5600/5600/5600, Mean rms=4.473%, delta=41.189%, char train=63.535%, word train=87.565%, skip ratio=0%,  New worst char error = 63.535At iteration 5400, stage 0, Eval Char error rate=89.383567, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5700/5700/5700, Mean rms=4.538%, delta=42.311%, char train=64.289%, word train=87.76%, skip ratio=0%,  New worst char error = 64.289At iteration 5500, stage 0, Eval Char error rate=96.985967, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5800/5800/5800, Mean rms=4.607%, delta=43.599%, char train=65.736%, word train=87.926%, skip ratio=0%,  New worst char error = 65.736At iteration 5600, stage 0, Eval Char error rate=97.234247, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5900/5900/5900, Mean rms=4.683%, delta=44.902%, char train=67.279%, word train=88.371%, skip ratio=0%,  New worst char error = 67.279At iteration 5700, stage 0, Eval Char error rate=101.09467, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6000/6000/6000, Mean rms=4.78%, delta=46.696%, char train=69.713%, word train=88.861%, skip ratio=0%,  New worst char error = 69.713At iteration 5800, stage 0, Eval Char error rate=101.08671, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6100/6100/6100, Mean rms=4.867%, delta=48.173%, char train=71.309%, word train=89.43%, skip ratio=0%,  New worst char error = 71.309At iteration 5900, stage 0, Eval Char error rate=103.41885, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6200/6200/6200, Mean rms=4.997%, delta=50.536%, char train=73.612%, word train=90.084%, skip ratio=0%,  New worst char error = 73.612At iteration 6000, stage 0, Eval Char error rate=98.385386, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6300/6300/6300, Mean rms=5.136%, delta=53.121%, char train=75.862%, word train=90.886%, skip ratio=0%,  New worst char error = 75.862At iteration 6100, stage 0, Eval Char error rate=111.64007, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6400/6400/6400, Mean rms=5.28%, delta=55.78%, char train=79.189%, word train=91.664%, skip ratio=0%,  New worst char error = 79.189At iteration 6200, stage 0, Eval Char error rate=110.51174, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6500/6500/6500, Mean rms=5.416%, delta=58.12%, char train=81.968%, word train=92.46%, skip ratio=0%,  New worst char error = 81.968At iteration 6300, stage 0, Eval Char error rate=134.73867, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6600/6600/6600, Mean rms=5.58%, delta=61.164%, char train=84.497%, word train=93.249%, skip ratio=0%,  New worst char error = 84.497At iteration 6400, stage 0, Eval Char error rate=127.09346, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6700/6700/6700, Mean rms=5.758%, delta=64.434%, char train=87.781%, word train=94.157%, skip ratio=0%,  New worst char error = 87.781At iteration 6500, stage 0, Eval Char error rate=128.03342, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
Layer 4=Lfys64: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
Layer 5=Lfx96: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
Layer 6=Lrx96: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
Layer 7=Lfx512: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
Layer 8=Output: lr 1.95313e-06->-nan%, lr 2.76214e-06->-nan% SAME
At iteration 6800/6800/6800, Mean rms=5.948%, delta=67.939%, char train=91.571%, word train=95.129%, skip ratio=0%,  New worst char error = 91.571At iteration 6600, stage 0, Eval Char error rate=122.21959, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.742%, delta=29.219%, char train=40.231%, word train=78.426%, skip ratio=0%,  New worst char error = 40.231 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.721%, delta=29.119%, char train=41.542%, word train=78.289%, skip ratio=0%,  New worst char error = 41.542 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.695%, delta=28.9%, char train=42.31%, word train=78.161%, skip ratio=0%,  New worst char error = 42.31 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.698%, delta=29.122%, char train=43.386%, word train=78.36%, skip ratio=0%,  New worst char error = 43.386 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.716%, delta=29.583%, char train=44.412%, word train=78.765%, skip ratio=0%,  New worst char error = 44.412 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.744%, delta=30.217%, char train=45.769%, word train=79.071%, skip ratio=0%,  New worst char error = 45.769 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.764%, delta=30.642%, char train=46.688%, word train=79.214%, skip ratio=0%,  New worst char error = 46.688 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.803%, delta=31.394%, char train=47.81%, word train=79.415%, skip ratio=0%,  New worst char error = 47.81 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.826%, delta=31.794%, char train=48.554%, word train=79.654%, skip ratio=0%,  New worst char error = 48.554 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.841%, delta=32.044%, char train=49.026%, word train=79.879%, skip ratio=0%,  New worst char error = 49.026 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=3.85%, delta=32.157%, char train=49.694%, word train=80.063%, skip ratio=0%,  New worst char error = 49.694At iteration 6700, stage 0, Eval Char error rate=125.05335, Word error rate=100 wrote checkpoint.

At iteration 2500/2500/2500, Mean rms=3.846%, delta=32.06%, char train=49.536%, word train=79.999%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2600/2600/2600, Mean rms=3.865%, delta=32.397%, char train=49.804%, word train=80.243%, skip ratio=0%,  New worst char error = 49.804At iteration 2300, stage 0, Eval Char error rate=95.063764, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=3.879%, delta=32.607%, char train=50.016%, word train=80.547%, skip ratio=0%,  New worst char error = 50.016At iteration 2400, stage 0, Eval Char error rate=90.478784, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=3.887%, delta=32.703%, char train=50.555%, word train=80.615%, skip ratio=0%,  New worst char error = 50.555At iteration 2600, stage 0, Eval Char error rate=90.478784, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=3.887%, delta=32.66%, char train=50.734%, word train=80.663%, skip ratio=0%,  New worst char error = 50.734At iteration 2700, stage 0, Eval Char error rate=89.569693, Word error rate=95 wrote checkpoint.

At iteration 3000/3000/3000, Mean rms=3.899%, delta=32.838%, char train=50.647%, word train=80.779%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=3.907%, delta=32.943%, char train=50.906%, word train=81.015%, skip ratio=0%,  New worst char error = 50.906At iteration 2800, stage 0, Eval Char error rate=90.034339, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=3.921%, delta=33.16%, char train=51.249%, word train=81.212%, skip ratio=0%,  New worst char error = 51.249At iteration 2900, stage 0, Eval Char error rate=91.051174, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3300/3300/3300, Mean rms=3.929%, delta=33.277%, char train=51.891%, word train=81.342%, skip ratio=0%,  New worst char error = 51.891At iteration 3100, stage 0, Eval Char error rate=88.034046, Word error rate=95 wrote checkpoint.

At iteration 3400/3400/3400, Mean rms=3.925%, delta=33.202%, char train=51.671%, word train=81.416%, skip ratio=0%,  wrote checkpoint.

At iteration 3500/3500/3500, Mean rms=3.936%, delta=33.383%, char train=51.706%, word train=81.573%, skip ratio=0%,  wrote checkpoint.

At iteration 3600/3600/3600, Mean rms=3.942%, delta=33.364%, char train=51.889%, word train=81.885%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3700/3700/3700, Mean rms=3.946%, delta=33.396%, char train=52.419%, word train=82.027%, skip ratio=0%,  New worst char error = 52.419At iteration 3200, stage 0, Eval Char error rate=88.550453, Word error rate=95 wrote checkpoint.

At iteration 3800/3800/3800, Mean rms=3.944%, delta=33.354%, char train=52.305%, word train=82.083%, skip ratio=0%,  wrote checkpoint.

At iteration 3900/3900/3900, Mean rms=3.951%, delta=33.475%, char train=52.134%, word train=82.208%, skip ratio=0%,  wrote checkpoint.

At iteration 4000/4000/4000, Mean rms=3.957%, delta=33.596%, char train=52.34%, word train=82.438%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4100/4100/4100, Mean rms=3.968%, delta=33.724%, char train=52.636%, word train=82.636%, skip ratio=0%,  New worst char error = 52.636At iteration 3300, stage 0, Eval Char error rate=89.809126, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4200/4200/4200, Mean rms=3.974%, delta=33.777%, char train=53.309%, word train=82.847%, skip ratio=0%,  New worst char error = 53.309At iteration 3700, stage 0, Eval Char error rate=95.824707, Word error rate=95 wrote checkpoint.

At iteration 4300/4300/4300, Mean rms=3.97%, delta=33.737%, char train=53.059%, word train=82.958%, skip ratio=0%,  wrote checkpoint.

At iteration 4400/4400/4400, Mean rms=3.983%, delta=33.927%, char train=53.179%, word train=83.048%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4500/4500/4500, Mean rms=3.993%, delta=34.051%, char train=53.533%, word train=83.521%, skip ratio=0%,  New worst char error = 53.533At iteration 4100, stage 0, Eval Char error rate=80.268858, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4600/4600/4600, Mean rms=4.005%, delta=34.243%, char train=54.197%, word train=83.721%, skip ratio=0%,  New worst char error = 54.197At iteration 4200, stage 0, Eval Char error rate=92.899931, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4700/4700/4700, Mean rms=4.007%, delta=34.3%, char train=54.245%, word train=83.845%, skip ratio=0%,  New worst char error = 54.245At iteration 4500, stage 0, Eval Char error rate=86.483008, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4800/4800/4800, Mean rms=4.023%, delta=34.564%, char train=54.563%, word train=84.086%, skip ratio=0%,  New worst char error = 54.563At iteration 4600, stage 0, Eval Char error rate=81.608173, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4900/4900/4900, Mean rms=4.028%, delta=34.606%, char train=54.893%, word train=84.213%, skip ratio=0%,  New worst char error = 54.893At iteration 4700, stage 0, Eval Char error rate=80.699082, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5000/5000/5000, Mean rms=4.047%, delta=34.871%, char train=55.381%, word train=84.544%, skip ratio=0%,  New worst char error = 55.381At iteration 4800, stage 0, Eval Char error rate=87.495619, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5100/5100/5100, Mean rms=4.055%, delta=34.973%, char train=56.064%, word train=84.773%, skip ratio=0%,  New worst char error = 56.064At iteration 4900, stage 0, Eval Char error rate=86.566326, Word error rate=95 wrote checkpoint.

At iteration 5200/5200/5200, Mean rms=4.051%, delta=34.907%, char train=55.804%, word train=84.949%, skip ratio=0%,  wrote checkpoint.

At iteration 5300/5300/5300, Mean rms=4.061%, delta=34.967%, char train=55.858%, word train=85.005%, skip ratio=0%,  wrote checkpoint.

At iteration 5400/5400/5400, Mean rms=4.072%, delta=35.074%, char train=56.056%, word train=85.407%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5500/5500/5500, Mean rms=4.089%, delta=35.356%, char train=56.868%, word train=85.517%, skip ratio=0%,  New worst char error = 56.868At iteration 5000, stage 0, Eval Char error rate=82.35113, Word error rate=95 wrote checkpoint.

At iteration 5600/5600/5600, Mean rms=4.089%, delta=35.303%, char train=56.852%, word train=85.586%, skip ratio=0%,  wrote checkpoint.

At iteration 5700/5700/5700, Mean rms=4.098%, delta=35.377%, char train=56.799%, word train=85.725%, skip ratio=0%,  wrote checkpoint.

At iteration 5800/5800/5800, Mean rms=4.103%, delta=35.374%, char train=56.86%, word train=85.891%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5900/5900/5900, Mean rms=4.122%, delta=35.594%, char train=57.197%, word train=86.204%, skip ratio=0%,  New worst char error = 57.197At iteration 5100, stage 0, Eval Char error rate=88.674363, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6000/6000/6000, Mean rms=4.133%, delta=35.682%, char train=57.78%, word train=86.309%, skip ratio=0%,  New worst char error = 57.78At iteration 5500, stage 0, Eval Char error rate=86.784219, Word error rate=95 wrote checkpoint.

At iteration 6100/6100/6100, Mean rms=4.136%, delta=35.701%, char train=57.486%, word train=86.39%, skip ratio=0%,  wrote checkpoint.

At iteration 6200/6200/6200, Mean rms=4.153%, delta=35.946%, char train=57.558%, word train=86.383%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6300/6300/6300, Mean rms=4.176%, delta=36.231%, char train=57.964%, word train=86.714%, skip ratio=0%,  New worst char error = 57.964At iteration 5900, stage 0, Eval Char error rate=86.61453, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6400/6400/6400, Mean rms=4.193%, delta=36.46%, char train=58.682%, word train=86.8%, skip ratio=0%,  New worst char error = 58.682At iteration 6000, stage 0, Eval Char error rate=83.718908, Word error rate=95 wrote checkpoint.

At iteration 6500/6500/6500, Mean rms=4.196%, delta=36.405%, char train=58.518%, word train=86.838%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6600/6600/6600, Mean rms=4.22%, delta=36.746%, char train=58.749%, word train=86.833%, skip ratio=0%,  New worst char error = 58.749At iteration 6300, stage 0, Eval Char error rate=90.727545, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6700/6700/6700, Mean rms=4.244%, delta=37.093%, char train=59.081%, word train=86.923%, skip ratio=0%,  New worst char error = 59.081At iteration 6400, stage 0, Eval Char error rate=89.489073, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6800/6800/6800, Mean rms=4.267%, delta=37.413%, char train=59.588%, word train=87.043%, skip ratio=0%,  New worst char error = 59.588At iteration 6600, stage 0, Eval Char error rate=81.410084, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6900/6900/6900, Mean rms=4.293%, delta=37.829%, char train=60.375%, word train=87.136%, skip ratio=0%,  New worst char error = 60.375At iteration 6700, stage 0, Eval Char error rate=87.329276, Word error rate=95 wrote checkpoint.

At iteration 7000/7000/7000, Mean rms=4.313%, delta=38.191%, char train=60.356%, word train=87.201%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7100/7100/7100, Mean rms=4.352%, delta=38.883%, char train=60.819%, word train=87.187%, skip ratio=0%,  New worst char error = 60.819At iteration 6800, stage 0, Eval Char error rate=90.837694, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7200/7200/7200, Mean rms=4.395%, delta=39.621%, char train=61.696%, word train=87.436%, skip ratio=0%,  New worst char error = 61.696At iteration 6900, stage 0, Eval Char error rate=101.15777, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7300/7300/7300, Mean rms=4.434%, delta=40.351%, char train=62.767%, word train=87.479%, skip ratio=0%,  New worst char error = 62.767At iteration 7100, stage 0, Eval Char error rate=93.716691, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7400/7400/7400, Mean rms=4.47%, delta=41.018%, char train=63.378%, word train=87.548%, skip ratio=0%,  New worst char error = 63.378At iteration 7200, stage 0, Eval Char error rate=93.825313, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7500/7500/7500, Mean rms=4.51%, delta=41.745%, char train=63.761%, word train=87.61%, skip ratio=0%,  New worst char error = 63.761At iteration 7300, stage 0, Eval Char error rate=93.649643, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7600/7600/7600, Mean rms=4.552%, delta=42.573%, char train=64.75%, word train=87.766%, skip ratio=0%,  New worst char error = 64.75At iteration 7400, stage 0, Eval Char error rate=97.921701, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7700/7700/7700, Mean rms=4.609%, delta=43.619%, char train=65.972%, word train=88.009%, skip ratio=0%,  New worst char error = 65.972At iteration 7500, stage 0, Eval Char error rate=97.916787, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7800/7800/7800, Mean rms=4.661%, delta=44.544%, char train=67.429%, word train=88.217%, skip ratio=0%,  New worst char error = 67.429At iteration 7600, stage 0, Eval Char error rate=99.71987, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7900/7900/7900, Mean rms=4.703%, delta=45.297%, char train=68.01%, word train=88.409%, skip ratio=0%,  New worst char error = 68.01At iteration 7700, stage 0, Eval Char error rate=98.074492, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8000/8000/8000, Mean rms=4.764%, delta=46.416%, char train=69.353%, word train=88.496%, skip ratio=0%,  New worst char error = 69.353At iteration 7800, stage 0, Eval Char error rate=98.754334, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8100/8100/8100, Mean rms=4.823%, delta=47.361%, char train=70.694%, word train=88.974%, skip ratio=0%,  New worst char error = 70.694At iteration 7900, stage 0, Eval Char error rate=95.431374, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8200/8200/8200, Mean rms=4.892%, delta=48.535%, char train=72.624%, word train=89.346%, skip ratio=0%,  New worst char error = 72.624At iteration 8000, stage 0, Eval Char error rate=103.69956, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8300/8300/8300, Mean rms=4.963%, delta=49.799%, char train=74.178%, word train=89.859%, skip ratio=0%,  New worst char error = 74.178At iteration 8100, stage 0, Eval Char error rate=95.712383, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8400/8400/8400, Mean rms=5.057%, delta=51.519%, char train=75.601%, word train=90.338%, skip ratio=0%,  New worst char error = 75.601At iteration 8200, stage 0, Eval Char error rate=106.85003, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8500/8500/8500, Mean rms=5.168%, delta=53.583%, char train=77.582%, word train=90.891%, skip ratio=0%,  New worst char error = 77.582At iteration 8300, stage 0, Eval Char error rate=109.6591, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8600/8600/8600, Mean rms=5.28%, delta=55.584%, char train=79.306%, word train=91.593%, skip ratio=0%,  New worst char error = 79.306At iteration 8400, stage 0, Eval Char error rate=107.2112, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8700/8700/8700, Mean rms=5.39%, delta=57.566%, char train=82.049%, word train=92.257%, skip ratio=0%,  New worst char error = 82.049At iteration 8500, stage 0, Eval Char error rate=113.37984, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8800/8800/8800, Mean rms=5.512%, delta=59.909%, char train=84.139%, word train=92.933%, skip ratio=0%,  New worst char error = 84.139At iteration 8600, stage 0, Eval Char error rate=116.14781, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8900/8900/8900, Mean rms=5.646%, delta=62.418%, char train=86.465%, word train=93.528%, skip ratio=0%,  New worst char error = 86.465At iteration 8700, stage 0, Eval Char error rate=114.78071, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9000/9000/9000, Mean rms=5.787%, delta=64.966%, char train=89.057%, word train=94.334%, skip ratio=0%,  New worst char error = 89.057At iteration 8800, stage 0, Eval Char error rate=99.579543, Word error rate=100 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
Layer 2=ConvNL: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
Layer 4=Lfys64: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
Layer 5=Lfx96: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
Layer 6=Lrx96: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
Layer 7=Lfx512: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
Layer 8=Output: lr 1.38107e-06->-nan%, lr 1.95313e-06->-nan% SAME
At iteration 9100/9100/9100, Mean rms=5.935%, delta=67.809%, char train=92.079%, word train=95.01%, skip ratio=0%,  New worst char error = 92.079At iteration 8900, stage 0, Eval Char error rate=118.01439, Word error rate=100
Divergence! Reverted to iteration 1300/1300/1300
Reduced learning rate on layers: 6
 wrote checkpoint.

At iteration 1400/1400/1400, Mean rms=3.741%, delta=29.212%, char train=40.269%, word train=78.426%, skip ratio=0%,  New worst char error = 40.269 wrote checkpoint.

At iteration 1500/1500/1500, Mean rms=3.72%, delta=29.089%, char train=41.544%, word train=78.289%, skip ratio=0%,  New worst char error = 41.544 wrote checkpoint.

At iteration 1600/1600/1600, Mean rms=3.694%, delta=28.866%, char train=42.303%, word train=78.161%, skip ratio=0%,  New worst char error = 42.303 wrote checkpoint.

At iteration 1700/1700/1700, Mean rms=3.696%, delta=29.094%, char train=43.397%, word train=78.347%, skip ratio=0%,  New worst char error = 43.397 wrote checkpoint.

At iteration 1800/1800/1800, Mean rms=3.713%, delta=29.546%, char train=44.374%, word train=78.726%, skip ratio=0%,  New worst char error = 44.374 wrote checkpoint.

At iteration 1900/1900/1900, Mean rms=3.742%, delta=30.2%, char train=45.65%, word train=78.971%, skip ratio=0%,  New worst char error = 45.65 wrote checkpoint.

At iteration 2000/2000/2000, Mean rms=3.761%, delta=30.59%, char train=46.514%, word train=79.128%, skip ratio=0%,  New worst char error = 46.514 wrote checkpoint.

At iteration 2100/2100/2100, Mean rms=3.801%, delta=31.387%, char train=47.698%, word train=79.315%, skip ratio=0%,  New worst char error = 47.698 wrote checkpoint.

At iteration 2200/2200/2200, Mean rms=3.822%, delta=31.758%, char train=48.393%, word train=79.5%, skip ratio=0%,  New worst char error = 48.393 wrote checkpoint.

At iteration 2300/2300/2300, Mean rms=3.833%, delta=31.935%, char train=48.797%, word train=79.696%, skip ratio=0%,  New worst char error = 48.797 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2400/2400/2400, Mean rms=3.84%, delta=32.036%, char train=49.3%, word train=79.855%, skip ratio=0%,  New worst char error = 49.3At iteration 9000, stage 0, Eval Char error rate=126.20921, Word error rate=100 wrote checkpoint.

At iteration 2500/2500/2500, Mean rms=3.833%, delta=31.905%, char train=48.964%, word train=79.752%, skip ratio=0%,  wrote checkpoint.

At iteration 2600/2600/2600, Mean rms=3.847%, delta=32.13%, char train=49.108%, word train=79.899%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2700/2700/2700, Mean rms=3.858%, delta=32.274%, char train=49.336%, word train=80.152%, skip ratio=0%,  New worst char error = 49.336At iteration 2300, stage 0, Eval Char error rate=84.773909, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2800/2800/2800, Mean rms=3.864%, delta=32.356%, char train=49.819%, word train=80.23%, skip ratio=0%,  New worst char error = 49.819At iteration 2400, stage 0, Eval Char error rate=88.854717, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 2900/2900/2900, Mean rms=3.858%, delta=32.226%, char train=49.977%, word train=80.347%, skip ratio=0%,  New worst char error = 49.977At iteration 2700, stage 0, Eval Char error rate=88.700128, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3000/3000/3000, Mean rms=3.87%, delta=32.422%, char train=50.001%, word train=80.369%, skip ratio=0%,  New worst char error = 50.001At iteration 2800, stage 0, Eval Char error rate=89.279838, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3100/3100/3100, Mean rms=3.871%, delta=32.392%, char train=50.137%, word train=80.559%, skip ratio=0%,  New worst char error = 50.137At iteration 2900, stage 0, Eval Char error rate=91.09802, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3200/3200/3200, Mean rms=3.883%, delta=32.569%, char train=50.494%, word train=80.739%, skip ratio=0%,  New worst char error = 50.494At iteration 3000, stage 0, Eval Char error rate=87.461656, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3300/3300/3300, Mean rms=3.889%, delta=32.658%, char train=51.09%, word train=80.912%, skip ratio=0%,  New worst char error = 51.09At iteration 3100, stage 0, Eval Char error rate=88.370747, Word error rate=95 wrote checkpoint.

At iteration 3400/3400/3400, Mean rms=3.883%, delta=32.624%, char train=50.9%, word train=80.883%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3500/3500/3500, Mean rms=3.895%, delta=32.831%, char train=51.247%, word train=81.028%, skip ratio=0%,  New worst char error = 51.247At iteration 3200, stage 0, Eval Char error rate=86.552565, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3600/3600/3600, Mean rms=3.902%, delta=32.875%, char train=51.532%, word train=81.336%, skip ratio=0%,  New worst char error = 51.532At iteration 3300, stage 0, Eval Char error rate=88.660602, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3700/3700/3700, Mean rms=3.906%, delta=32.903%, char train=51.922%, word train=81.444%, skip ratio=0%,  New worst char error = 51.922At iteration 3500, stage 0, Eval Char error rate=87.307067, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 3800/3800/3800, Mean rms=3.901%, delta=32.87%, char train=51.991%, word train=81.41%, skip ratio=0%,  New worst char error = 51.991At iteration 3600, stage 0, Eval Char error rate=85.488885, Word error rate=95 wrote checkpoint.

At iteration 3900/3900/3900, Mean rms=3.907%, delta=32.979%, char train=51.822%, word train=81.348%, skip ratio=0%,  wrote checkpoint.

At iteration 4000/4000/4000, Mean rms=3.913%, delta=33.063%, char train=51.904%, word train=81.531%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4100/4100/4100, Mean rms=3.924%, delta=33.243%, char train=52.144%, word train=81.633%, skip ratio=0%,  New worst char error = 52.144At iteration 3700, stage 0, Eval Char error rate=88.660602, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 4200/4200/4200, Mean rms=3.928%, delta=33.278%, char train=52.671%, word train=81.746%, skip ratio=0%,  New worst char error = 52.671At iteration 3800, stage 0, Eval Char error rate=89.569693, Word error rate=95 wrote checkpoint.

At iteration 4300/4300/4300, Mean rms=3.921%, delta=33.147%, char train=52.24%, word train=81.651%, skip ratio=0%,  wrote checkpoint.

At iteration 4400/4400/4400, Mean rms=3.931%, delta=33.301%, char train=52.311%, word train=81.754%, skip ratio=0%,  wrote checkpoint.

At iteration 4500/4500/4500, Mean rms=3.94%, delta=33.407%, char train=52.208%, word train=82.012%, skip ratio=0%,  wrote checkpoint.

At iteration 4600/4600/4600, Mean rms=3.947%, delta=33.509%, char train=52.587%, word train=82.054%, skip ratio=0%,  wrote checkpoint.

At iteration 4700/4700/4700, Mean rms=3.945%, delta=33.466%, char train=52.484%, word train=82.086%, skip ratio=0%,  wrote checkpoint.

At iteration 4800/4800/4800, Mean rms=3.948%, delta=33.441%, char train=52.238%, word train=82.141%, skip ratio=0%,  wrote checkpoint.

At iteration 4900/4900/4900, Mean rms=3.952%, delta=33.502%, char train=52.334%, word train=82.301%, skip ratio=0%,  wrote checkpoint.

At iteration 5000/5000/5000, Mean rms=3.958%, delta=33.587%, char train=52.407%, word train=82.475%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5100/5100/5100, Mean rms=3.96%, delta=33.563%, char train=52.804%, word train=82.68%, skip ratio=0%,  New worst char error = 52.804At iteration 4100, stage 0, Eval Char error rate=89.407784, Word error rate=95 wrote checkpoint.

At iteration 5200/5200/5200, Mean rms=3.953%, delta=33.449%, char train=52.438%, word train=82.713%, skip ratio=0%,  wrote checkpoint.

At iteration 5300/5300/5300, Mean rms=3.963%, delta=33.62%, char train=52.487%, word train=82.79%, skip ratio=0%,  wrote checkpoint.

At iteration 5400/5400/5400, Mean rms=3.968%, delta=33.632%, char train=52.579%, word train=83.081%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5500/5500/5500, Mean rms=3.974%, delta=33.702%, char train=53.225%, word train=83.195%, skip ratio=0%,  New worst char error = 53.225At iteration 4200, stage 0, Eval Char error rate=92.135057, Word error rate=95 wrote checkpoint.

At iteration 5600/5600/5600, Mean rms=3.973%, delta=33.711%, char train=53.221%, word train=83.253%, skip ratio=0%,  wrote checkpoint.

At iteration 5700/5700/5700, Mean rms=3.977%, delta=33.745%, char train=53.18%, word train=83.394%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5800/5800/5800, Mean rms=3.984%, delta=33.887%, char train=53.327%, word train=83.6%, skip ratio=0%,  New worst char error = 53.327At iteration 5100, stage 0, Eval Char error rate=90.828366, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 5900/5900/5900, Mean rms=3.997%, delta=34.078%, char train=53.876%, word train=83.785%, skip ratio=0%,  New worst char error = 53.876At iteration 5500, stage 0, Eval Char error rate=79.263567, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6000/6000/6000, Mean rms=4.006%, delta=34.177%, char train=54.603%, word train=83.96%, skip ratio=0%,  New worst char error = 54.603At iteration 5800, stage 0, Eval Char error rate=80.617103, Word error rate=95 wrote checkpoint.

At iteration 6100/6100/6100, Mean rms=4.005%, delta=34.19%, char train=54.447%, word train=83.987%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6200/6200/6200, Mean rms=4.017%, delta=34.384%, char train=54.639%, word train=84.078%, skip ratio=0%,  New worst char error = 54.639At iteration 5900, stage 0, Eval Char error rate=79.823075, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6300/6300/6300, Mean rms=4.03%, delta=34.555%, char train=54.842%, word train=84.383%, skip ratio=0%,  New worst char error = 54.842At iteration 6000, stage 0, Eval Char error rate=78.934187, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6400/6400/6400, Mean rms=4.041%, delta=34.721%, char train=55.389%, word train=84.502%, skip ratio=0%,  New worst char error = 55.389At iteration 6200, stage 0, Eval Char error rate=81.937554, Word error rate=95 wrote checkpoint.

At iteration 6500/6500/6500, Mean rms=4.032%, delta=34.581%, char train=55.231%, word train=84.613%, skip ratio=0%,  wrote checkpoint.

At iteration 6600/6600/6600, Mean rms=4.04%, delta=34.656%, char train=55.327%, word train=84.701%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6700/6700/6700, Mean rms=4.048%, delta=34.776%, char train=55.432%, word train=84.853%, skip ratio=0%,  New worst char error = 55.432At iteration 6300, stage 0, Eval Char error rate=85.109271, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6800/6800/6800, Mean rms=4.06%, delta=34.921%, char train=55.654%, word train=85.066%, skip ratio=0%,  New worst char error = 55.654At iteration 6400, stage 0, Eval Char error rate=83.259343, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 6900/6900/6900, Mean rms=4.067%, delta=34.977%, char train=56.034%, word train=85.233%, skip ratio=0%,  New worst char error = 56.034At iteration 6700, stage 0, Eval Char error rate=79.663383, Word error rate=95 wrote checkpoint.

At iteration 7000/7000/7000, Mean rms=4.06%, delta=34.855%, char train=55.437%, word train=85.233%, skip ratio=0%,  wrote checkpoint.

At iteration 7100/7100/7100, Mean rms=4.076%, delta=35.075%, char train=55.514%, word train=85.276%, skip ratio=0%,  wrote checkpoint.

At iteration 7200/7200/7200, Mean rms=4.086%, delta=35.144%, char train=55.523%, word train=85.501%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7300/7300/7300, Mean rms=4.098%, delta=35.3%, char train=56.151%, word train=85.645%, skip ratio=0%,  New worst char error = 56.151At iteration 6800, stage 0, Eval Char error rate=90.821925, Word error rate=95 wrote checkpoint.

At iteration 7400/7400/7400, Mean rms=4.094%, delta=35.152%, char train=56.077%, word train=85.835%, skip ratio=0%,  wrote checkpoint.

At iteration 7500/7500/7500, Mean rms=4.112%, delta=35.383%, char train=56.084%, word train=85.833%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7600/7600/7600, Mean rms=4.117%, delta=35.405%, char train=56.296%, word train=85.88%, skip ratio=0%,  New worst char error = 56.296At iteration 6900, stage 0, Eval Char error rate=89.407784, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7700/7700/7700, Mean rms=4.13%, delta=35.524%, char train=56.538%, word train=86.051%, skip ratio=0%,  New worst char error = 56.538At iteration 7300, stage 0, Eval Char error rate=89.677625, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 7800/7800/7800, Mean rms=4.141%, delta=35.673%, char train=57.232%, word train=86.202%, skip ratio=0%,  New worst char error = 57.232At iteration 7600, stage 0, Eval Char error rate=88.297447, Word error rate=95 wrote checkpoint.

At iteration 7900/7900/7900, Mean rms=4.141%, delta=35.673%, char train=57.123%, word train=86.242%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8000/8000/8000, Mean rms=4.154%, delta=35.885%, char train=57.656%, word train=86.326%, skip ratio=0%,  New worst char error = 57.656At iteration 7700, stage 0, Eval Char error rate=90.612021, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8100/8100/8100, Mean rms=4.164%, delta=35.952%, char train=57.984%, word train=86.516%, skip ratio=0%,  New worst char error = 57.984At iteration 7800, stage 0, Eval Char error rate=88.908903, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8200/8200/8200, Mean rms=4.183%, delta=36.245%, char train=58.984%, word train=86.671%, skip ratio=0%,  New worst char error = 58.984At iteration 8000, stage 0, Eval Char error rate=84.996445, Word error rate=95 wrote checkpoint.

At iteration 8300/8300/8300, Mean rms=4.182%, delta=36.177%, char train=58.926%, word train=86.627%, skip ratio=0%,  wrote checkpoint.

At iteration 8400/8400/8400, Mean rms=4.2%, delta=36.474%, char train=58.939%, word train=86.53%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8500/8500/8500, Mean rms=4.211%, delta=36.681%, char train=59.21%, word train=86.594%, skip ratio=0%,  New worst char error = 59.21At iteration 8100, stage 0, Eval Char error rate=87.279273, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8600/8600/8600, Mean rms=4.224%, delta=36.866%, char train=59.286%, word train=86.828%, skip ratio=0%,  New worst char error = 59.286At iteration 8200, stage 0, Eval Char error rate=84.964699, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 8700/8700/8700, Mean rms=4.243%, delta=37.158%, char train=60.171%, word train=86.838%, skip ratio=0%,  New worst char error = 60.171At iteration 8500, stage 0, Eval Char error rate=96.632976, Word error rate=95 wrote checkpoint.

At iteration 8800/8800/8800, Mean rms=4.244%, delta=37.139%, char train=59.929%, word train=86.816%, skip ratio=0%,  wrote checkpoint.

At iteration 8900/8900/8900, Mean rms=4.26%, delta=37.362%, char train=59.818%, word train=86.637%, skip ratio=0%,  wrote checkpoint.

At iteration 9000/9000/9000, Mean rms=4.279%, delta=37.555%, char train=59.899%, word train=86.825%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9100/9100/9100, Mean rms=4.302%, delta=38.011%, char train=60.453%, word train=86.966%, skip ratio=0%,  New worst char error = 60.453At iteration 8600, stage 0, Eval Char error rate=96.025284, Word error rate=95 wrote checkpoint.

At iteration 9200/9200/9200, Mean rms=4.306%, delta=38.086%, char train=60.188%, word train=86.976%, skip ratio=0%,  wrote checkpoint.

At iteration 9300/9300/9300, Mean rms=4.33%, delta=38.471%, char train=60.218%, word train=87.072%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9400/9400/9400, Mean rms=4.345%, delta=38.721%, char train=60.539%, word train=87.182%, skip ratio=0%,  New worst char error = 60.539At iteration 8700, stage 0, Eval Char error rate=91.91273, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9500/9500/9500, Mean rms=4.377%, delta=39.272%, char train=61.235%, word train=87.404%, skip ratio=0%,  New worst char error = 61.235At iteration 9100, stage 0, Eval Char error rate=92.034423, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9600/9600/9600, Mean rms=4.41%, delta=39.929%, char train=62.235%, word train=87.504%, skip ratio=0%,  New worst char error = 62.235At iteration 9400, stage 0, Eval Char error rate=87.61269, Word error rate=95 wrote checkpoint.

At iteration 9700/9700/9700, Mean rms=4.427%, delta=40.288%, char train=61.949%, word train=87.542%, skip ratio=0%,  wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9800/9800/9800, Mean rms=4.46%, delta=40.893%, char train=62.296%, word train=87.484%, skip ratio=0%,  New worst char error = 62.296At iteration 9500, stage 0, Eval Char error rate=94.214422, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 9900/9900/9900, Mean rms=4.498%, delta=41.601%, char train=62.952%, word train=87.845%, skip ratio=0%,  New worst char error = 62.952At iteration 9600, stage 0, Eval Char error rate=90.524772, Word error rate=95 wrote checkpoint.

Warning: LSTMTrainer deserialized an LSTMRecognizer!
At iteration 10000/10000/10000, Mean rms=4.537%, delta=42.37%, char train=64.074%, word train=87.848%, skip ratio=0%,  New worst char error = 64.074At iteration 9800, stage 0, Eval Char error rate=97.468118, Word error rate=95 wrote checkpoint.

Finished! Error rate = 39.619
lstmtraining \
--stop_training \
--continue_from data/mmfoo/checkpoints/mmfoo_checkpoint \
--traineddata data/mmfoo/mmfoo.traineddata \
--model_output data/mmfoo.traineddata
Loaded file data/mmfoo/checkpoints/mmfoo_checkpoint, unpacking...
```
