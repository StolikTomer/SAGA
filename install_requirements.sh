#!/bin/bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch --yes
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3
pip install torch-summary==1.4.5 --yes
pip install openmesh==1.1.6
conda install -c conda-forge matplotlib==3.3.4 --yes
pip install plotly==5.5.0
pip install ipywidgets==7.6.5
conda install -c anaconda jupyter==1.0.0 --yes
conda install -c conda-forge jupyter_client==7.1.2 --yes
conda install -c conda-forge jupyter_console==5.2.0 --yes
conda install -c conda-forge jupyterlab_widgets==1.0.2 --yes
pip install open3d==0.9.0.0
pip install hdf5storage==0.1.16
pip install seaborn==0.11.2
pip install prompt-toolkit==1.0.18

