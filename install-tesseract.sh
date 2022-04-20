# Install and build tesseract 
# Author : Thura Aung

# Update and install dependencies

apt-get update 
apt-get install -y wget unzip bc vim python3-pip libleptonica-dev git

# Packages to compile Tesseract Engine

apt-get install -y --reinstall make 
apt-get install -y g++ autoconf utomake libtool pkg-config libpng-dev libjpeg8-dev libtiff5-dev libicu-dev \
                        libpango1.0-dev autoconf-archive

# Getting tesstrain: beware the source might change or not being available
# Complie Tesseract with training options (also feel free to update Tesseract versions and such!)
# Getting data: beware the source might change or not being available

mkdir src && cd /app/src 
git clone https://github.com/tesseract-ocr/tesseract
cd /app/src/tesseract && ./autogen.sh && ./configure && make && make install && ldconfig 
make training && make training-install 
cd /usr/local/share/tessdata 
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/mya.traineddata
