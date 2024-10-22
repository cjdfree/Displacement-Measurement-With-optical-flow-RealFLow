#!/bin/bash
mkdir RF_dataset
cd RF_dataset

# RF-Ktrain
mkdir RF-Ktrain
cd RF-Ktrain
wget https://data.megengine.org.cn/research/realflow/RF-Ktrain-flow.zip
wget https://data.megengine.org.cn/research/realflow/RF-Ktrain-img.zip
wget https://data.megengine.org.cn/research/realflow/RF-Ktrain-flo.zip
cd ..

# RF-KTest
mkdir RF-KTest
cd RF-KTest
wget https://data.megengine.org.cn/research/realflow/RF-KTest-flow.zip
wget https://data.megengine.org.cn/research/realflow/RF-KTest-img.zip
wget https://data.megengine.org.cn/research/realflow/RF-KTest-flo.zip
cd ..

# RF-Sintel
mkdir RF-Sintel
cd RF-Sintel
wget https://data.megengine.org.cn/research/realflow/RFAB-sintel-flow.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-sintel-img.zip
cd ..

# RF-DAVIS
mkdir RF-DAVIS
cd RF-DAVIS
wget https://data.megengine.org.cn/research/realflow/RF-Davis-flow.zip
wget https://data.megengine.org.cn/research/realflow/RF-Davis-img.zip
wget https://data.megengine.org.cn/research/realflow/RF-Davis-flo.zip
cd ..

# RF-AB
mkdir RF-AB
cd RF-AB
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Apart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Apart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Apart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Apart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Bpart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Bpart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Bpart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-img-Bpart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Apart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Apart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Apart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Apart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Bpart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Bpart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Bpart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flo-Bpart3.zip
cd ..
