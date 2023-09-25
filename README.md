# SAGA: Spectral Adversarial Geometric Attack on 3D Meshes
Created by [Tomer Stolik](https://scholar.google.com/citations?pli=1&authuser=2&user=DGrmAqkAAAAJ), [Itai Lang](https://itailang.github.io/), and [Shai Avidan](http://www.eng.tau.ac.il/~avidan/) from Tel Aviv University.

[[Paper]](https://arxiv.org/abs/2211.13775) [[Project Page]](https://stoliktomer.github.io/SAGA/) [[Video]](https://www.youtube.com/watch?v=qOtEIj8hEcU)

![figure_1_video](./doc/figure_1_video.gif)

![figure_2_video](./doc/figure_2_video.gif)

## Abstract
A triangular mesh is one of the most popular 3D data representations. As such, the deployment of deep neural networks for mesh processing is widely spread and is increasingly attracting more attention. However, neural networks are prone to adversarial attacks, where carefully crafted inputs impair the model's functionality. The need to explore these vulnerabilities is a fundamental factor in future development of 3D-based applications. Recently, mesh attacks were studied on the semantic level, where classifiers are misled to produce wrong predictions. Nevertheless, mesh surfaces possess complex geometric attributes beyond their semantic meaning, and their analysis often includes the need to encode and reconstruct the geometric shape.

We propose a novel framework for a geometric adversarial attack on a 3D mesh autoencoder. In this setting, an adversarial input mesh deceives an autoencoder by forcing it to reconstruct a different geometric shape at its output. The malicious input is produced by perturbing a clean shape in a spectral domain. Our method leverages the spectral decomposition of the mesh along with additional mesh-related properties to obtain visually credible results that consider the delicacy of surface distortions. Our code is publicly available.

## Citation
If you find our work useful in your research, please consider citing:

	@article{stolik2022saga,
	  author = {Stolik, Tomer and Lang, Itai and Avidan, Shai},
	  title = {{SAGA: Spectral Adversarial Geometric Attack on 3D Meshes}},
	  journal = {arXiv preprint arXiv:2211.13775},
	  year = {2022}
	}

## Overview
The code has the following structure:

```
src/
coma(or smal)/
├── models/
│   ├── autoencoders/
│   ├── classifiers/
│   ├── detectors/
├── data/
│   ├── raw/
│   ├── spectral/
│   ├── index/
├── logs/
├── results/
├── images/
```

* All the directories are automatically generated by our code.
* The directory ```../data/raw/``` stores the raw data files.
* The directory ```../data/spectral/``` stores the shared spectral basis of eigenvectors and the spectral coefficients of each shape.
* The directory ```../data/index/``` stores the matrices according which the source-target pairs are arranged.
* The results of each attack are stored as a dictionary in a ```.pickle``` file under ```../results/```.
* The log of every script is automatically saved and stored under ```../logs/```.
* Visualizations and plots are stored under ```../images/```.
* The official pre-trained models and future training results are saved under ```../models/autoencoders/```, ```../models/classifiers/```, ```../models/detectors/```.

## Installation

The code was tested with Python 3.6, CUDA 10.2, cuDNN 7.6.5 on Ubuntu 16.04.
To install and activate the conda environment of the project please run:

```bash
conda create --name saga python=3.6 --yes
conda activate saga
bash install_requirements.sh
```

The Chamfer Distance compilation is implemented according to the [point cloud geometric attack](https://github.com/itailang/geometric_adv) by Lang *et al*. The relevant code is located under src/ChamferDistancePytorch/chamfer3D folder. The sh compilation script uses CUDA 10.2 path. If needed, modify the script to point to your CUDA path. Then, use:

```bash
cd src/
sh compile_op_pt.sh
cd ../
```

## Prepare the Data

### Download Datasets and Models
To download the datasets and the pre-trained models please run:

```bash
bash download_data_and_models.sh
```

You may also manually download the data. The datasets and models are available in the following links:

* [CoMA data and models](https://drive.google.com/drive/u/4/folders/1zdreTZdhBvjlUDdHQqGyp42_5yTqSM7C)
* [SMAL data and models](https://drive.google.com/drive/u/4/folders/1D_skXub4s_PuK_HOXYvVCA8lHkmG1xI9)

Please extract and place the content of each .zip file according the specified steps in ```download_data_and_models.sh``` 

### Arrange the Attacked Pairs
The next step is to prepare the source-target pairs for the attack:

```bash
cd src/
python prepare_indices_for_attack.py --dataset coma
python prepare_indices_for_attack.py --dataset smal
```

## Attack
The main attack experiments are obtained by:

```bash
python run_attack_wrapper.py --dataset coma --w_recon_mse 1 --w_reg_bary 100 --w_reg_area 500 --w_reg_edge 2

python run_attack_wrapper.py --dataset smal --w_recon_mse 1 --w_reg_bary 50 --w_reg_normals 0.5 --w_reg_edge 5
```

Additional attack variants are obtained by different flags.

Different regularization losses can be explored by changing:
```bash
--w_reg_bary
--w_reg_area
--w_reg_edge
--w_reg_normals
```

For a Euclidean attack please add:
```bash
--adversary_type delta
```

To avoid the use of a shared basis add:
```bash
--use_self_evects true
```

To pick the target shapes randomly first run:
```bash
python prepare_indices_for_attack.py --dataset coma --random_targets_mode true
python prepare_indices_for_attack.py --dataset smal --random_targets_mode true
```
and then add the following flag to the main attack experiment:
```bash
--random_targets_mode true
```

Our comparison to Lang *et al.*'s [point cloud geometric attack](https://github.com/itailang/geometric_adv) is obtained by:

```bash
python run_attack_wrapper.py --adversary_type delta --dataset coma --w_recon_mse 1 --w_reg_chamfer 0.5

python run_attack_wrapper.py --adversary_type delta --dataset smal --w_recon_mse 1 --w_reg_chamfer 10
```

## Evaluation
To evaluate different attack variants, please insert the experiment's file name (stored under ```../results/```) to the method ```get_results_path``` (in ```src/utils_attack.py```) under the relevant ```result_type```.

### Geometric Evaluation and Analysis

The relevant script is ```evaluate_attack.py```.

To calculate the geometric measures use ```--evaluation_type curv_dist```.

To explore visual results use ```--evaluation_type visual --visualize true```.

For a spectral analysis use ```--evaluation_type beta  --visualize true```.

For a latent space visualization use ```--evaluation_type tsne  --visualize true```

You may control the dataset, the experiment's type, and other settings using the different flags.
The plots are saved under ```../images/``` when using ```--save_images true```.

### Semantic Evaluation
For a semantic evaluation of the reconstructed shapes use the script:

```bash
python evaluate_classifier.py
```

You may control the dataset, the experiment's type, and other settings using the different flags.

To train a detector to distinguish adversarial shapes use:

```bash
python train_evaluate_detector.py --purpose train --detector_test_class 0
```
Recall that we exclude attacked shapes from one class for validation and testing (class 0 here is an example).
To test the detector's accuracy run:

```bash
python train_evaluate_detector.py --purpose test --detector_test_class 0 
```

### Transferability
To produce results on the "other_mlp" autoencoder (Figure 6 in the paper) run:

```bash
python ae_transfer.py --dataset coma --result_type saga
```

To obtain results on the "CoMA" model, please refer to our submodule [ComaTransfer](./ComaTransfer).

## License

This project is licensed under the terms of the MIT license (see the [LICENSE](./LICENSE) file for more details).

## Acknowledgment

We were inspired by the work of [Lang *et al.*](https://github.com/itailang/geometric_adv), [Rampini *et al.*](https://github.com/AriannaRampini/SpectralAdversarialAttacks), and [Marin *et al*](https://github.com/riccardomarin/InstantRecoveryFromSpectrum). We thank the authors for sharing their code.


