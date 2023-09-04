![Title
Badge](https://img.shields.io/badge/Song_Type_Validation-k?style=for-the-badge&labelColor=d99c2b&color=d99c2b)
![R
Badge](https://img.shields.io/badge/≥3.8-4295B3?style=for-the-badge&logo=python&logoColor=white)
#

![version](https://img.shields.io/badge/v0.1.0-orange?style=flat-square)
![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg?style=flat-square)
![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)


This repository contains the code to to train a classifier to check the
robustness of a manual classification of Great Tit song types following
[McGregor & Krebs (1982)](https://doi.org/10.1163/156853982X00210). For more
information on the preprocessing of the data, see [this
paper](https://www.biorxiv.org/content/10.1101/2023.07.03.547484v1). Model
training follows the steps described in [this
article](https://onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.14155).

A narrative code notebook including outputs can be found [here](https://github.com/nilomr/wytham-songtype-validation/blob/main/notebooks/4_train-model.ipynb).


<!-- add a table of  -->

### Installation

1. Create a new environment, e.g. using miniconda:
```bash
conda create -n wytham-songtype-validation python=3.9
```

2. Clone this repository to your local machine, navigate to its root and install
using pip:

```bash
git clone https://github.com/nilomr/wytham-songtype-validation.git
cd wytham-songtype-validation
pip install .
  ```
<br>


##### GPU installation

One of the steps to reproduce this example involves training a deep neural
network, which requires compatible GPU resources.

If you want to reatrain the model, you will need a few more libraries
that are not installed automatically with pykanto. The reason for this is that the are a bit finicky: which exact installation you need depends on which version of
CUDA you have and the like.

I recommend that, if this is the case, you first create a fresh environment with conda:

```bash
conda create -n wytham-songtype-validation python=3.9
```         
And then install torch, pykanto and this example including the extra libraries.

```bash
conda install -c pytorch pytorch torchvision   
pip install pykanto
# Navigate to the root of this repository, then:
pip install ."[torch]" # see the pyproject.toml file for other options
```


### User guide


First, make sure that you have activated this project's environment (`conda
activate wytham-songtype-validation` if you followed the instructions above). Then,
navigate to ```/notebooks```. This is where the scripts are located. They can
all be run from the terminal, `python <script-name>`.

<details>
  <summary>Expand user guide</summary>
<br>

| Script                      | Description                                                 | Use                                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1_prepare-dataset.py`      | Ingests, creates spectrograms, and segments the dataset[^1] | To run: `python 1_prepare-dataset.py`.<br>Requires the output of [this repository](https://github.com/nilomr/great-tit-hits-setup).                                         |
| `3_export-training-data.py` | Exports the data required to train the deep learning model  | `python 3_export-training-data.py`                                                                                                                                               |
| `4_train-model.ipynb`       | Model definition and training step                          | A separate, self-contained jupyter notebook. This is to make it easier to run interactively on a GPU-enabled HPC. If you don't want to retrain the model, you can skip this step. |
| `5_save_labels.py`          | Saves the checked labels to a csv file                      | `python 5_save_labels.py`                                                                                                                                                        |


[^1]: If you want to run this in a HPC you can use `pykanto`'s tool for this,
    which makes it very easy (see
    [Docs](https://nilomr.github.io/pykanto/_build/html/contents/hpc.html) for
    more info).

</details>

<br>

#
<sub>© Nilo M. Recalde, 2021-present</sub>

