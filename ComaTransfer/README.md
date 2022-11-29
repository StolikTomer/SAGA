# Spectral Adversarial Geometric Attack on CoMA
This submodule is used to apply SAGA's adversarial shapes on [CoMA](https://arxiv.org/abs/1807.10267).
The code is inspired by CoMA's [public implementation](https://github.com/pixelite1201/pytorch_coma/). Please follow the license rights of the authors if you use the code.

## Installation

This code is tested on Python 3.6 and Pytorch version 1.3. Requirements can be installed by running:

```bash
conda create --name saga_transfer python=3.6 --yes
conda activate saga_transfer
bash install_transfer_requirements.sh
```

Please also install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh).
The steps are:

```bash
git clone https://github.com/MPI-IS/mesh.git
sudo apt-get install libboost-dev
cd mesh
pip install --upgrade pip
make all
cd ../
```

## Prepare the Data

* Please copy the same training dataset as used for the victim autoencoder:

```bash
mkdir raw_data
cp ../coma/data/raw/coma_FEM.mat raw_data/
cp ../coma/data/raw/mesh_faces.npy raw_data/
```

* To test the trained model on SAGA's adversarial examples, please copy the desired results file from ```coma/results/``` and place it under ```ComaTransfer/attack_data/```. Then, rename it to ```attack_data_in.pickle``` .

## Train the Model

To train the model please use:

```bash
python main.py
```

## Test the Model

To evaluate the model please set the ```eval_flag``` to ```true``` and provide a checkpoint file. For example:

```bash
python main.py --eval_flag true --checkpoint_file_name checkpoint_sgd_0.008_199.pt 
```

## Transfer SAGA

To evaluate and save the outputs of SAGA's adversarial shapes, please use:

```bash
python main.py --eval_attack_flag true --checkpoint_file_name checkpoint_sgd_0.008_199.pt 
```

The output is a dictionary, saved as ```attack_data_out.pickle``` under ```ComaTransfer/attack_data/```.





