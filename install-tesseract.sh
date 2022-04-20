# Install and build tesseract 
# Author : Thura Aung

# Update and install dependencies

sudo apt-get update 
sudo apt-get install -y wget unzip bc vim python3-pip libleptonica-dev git

# Packages to compile Tesseract Engine

sudo apt-get install -y --reinstall make 
sudo apt-get install -y g++ autoconf utomake libtool pkg-config libpng-dev libjpeg8-dev libtiff5-dev libicu-dev \
                        libpango1.0-dev autoconf-archive

# Getting tesstrain: beware the source might change or not being available
# Complie Tesseract with training options (also feel free to update Tesseract versions and such!)
# Getting data: beware the source might change or not being available

mkdir src && cd /app/src 
wget https://github.com/tesseract-ocr/tesseract/archive/4.1.0.zip 
unzip 4.1.0.zip 
cd /app/src/tesseract-4.1.0 && ./autogen.sh && ./configure && make && make install && ldconfig 
make training && make training-install 
cd /usr/local/share/tessdata 
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/mya.traineddata
