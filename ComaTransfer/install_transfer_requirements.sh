#!/bin/bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch --yes
pip install torch-scatter==1.3.2 torch-sparse==0.4.3 torch-cluster==1.4.5 torch-geometric==1.3.2
conda install -c conda-forge matplotlib==3.3.4 --yes
pip install cached-property==1.5.2
pip install certifi==2020.12.5
pip install chardet==4.0.0
pip install cycler==0.10.0
pip install decorator==4.4.2
pip install googledrivedownloader==0.4
pip install h5py==3.1.0
pip install idna==2.10
pip install isodate==0.6.0
pip install joblib==1.0.1
pip install kiwisolver==1.3.1
pip install networkx==2.5
pip install numpy==1.17.3
pip install opencv-python==4.1.1.26
pip install pandas==1.1.5
pip install Pillow==8.1.0
pip install plyfile==0.7.3
pip install PyOpenGL==3.1.5
pip install pyparsing==2.4.7
pip install python-dateutil==2.8.1
pip install pytz==2021.1
pip install PyYAML==5.4.1
pip install pyzmq==22.0.3
pip install rdflib==5.0.0
pip install requests==2.25.1
pip install scikit-learn==0.21.3
pip install scipy==1.3.1
pip install six==1.15.0
pip install threadpoolctl==2.1.0
pip install tqdm==4.58.0
pip install urllib3==1.26.3
pip install torch==1.3.0
pip uninstall torchvision --yes
pip install hdf5storage==0.1.18

