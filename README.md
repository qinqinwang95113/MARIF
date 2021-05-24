# MARIF:User-Item Mutual Aware Representation with Instant Feedback for Click-Through Prediction

This repository is the official implementation of [MARIF:User-Item Mutual Aware Representation with Instant Feedback for Click-Through Prediction](). 

<p align="center">
<img src="img/marif.jpg" height=400>
</p>

## Installation

To install requirements, download and install the Python 3.7 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/#download-section).

Then run the following commands to create a conda environment named `marif` with Python 3.7, and clone this repository into a folder named `marif` your machine.

```bash
git clone git@github.com:qinqinwang95113/MARIF.git
cd auto-rec
conda create -n marif python=3.7
conda env update --name marif --file environment.yml
```

<!-- > 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Training
## Quick start please check 
For a quick walkthrough on how to run `MARIF model`, take a look at the [run.py](deepTrans/run.py).
For train the model from scratch
```bash
python run.py
```
<!-- 
```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation

<!-- To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
 -->

We use three common datasets in CTR task for evaluation, whereas not all users/items are involved in training because of the we conduct flexible up-down sampling strategy.
<p align="center">
<img src="img/table2.jpg">
<img src="img/table1.jpg">

</p>


## Results
Our model achieves the following performance on different datasets comparing with different models:

<p align="center">
<img src="img/results.jpg" height=277>
</p>

<!-- Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
 -->

## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 