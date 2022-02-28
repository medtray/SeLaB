# SeLaB: Semantic Labeling with BERT

This repository contains source code for the `SeLaB` model ([IJCNN](https://ieeexplore.ieee.org/abstract/document/9534408), [arXiv](https://arxiv.org/abs/2010.16037)), a new context-aware semantic labeling method using both the column values and contextual information. `SeLaB` s based on a new setting for semantic labeling, where we sequentially predict labels for an input table with missing headers. We incorporate both the values and context of each data column using the pre-trained contextualized language model, BERT. To our knowledge, we are the first to successfully apply BERT to solve the semantic labeling task. 

## Installation

First, install the conda environment `selab` with supporting libraries.

```bash
conda create --name selab python=3.7
conda activate selab
pip install -r requirements.txt
```

## Data

We use `WikiTables` collection for evaluation. This dataset is composed of the `WikiTables` corpus which contains over 1.6𝑀 tables that are extracted from Wikipedia. Since a lot of tables have unexpected formats, we preprocess tables so that we only keep tables that have enough content with at least 3 columns and 50 rows. We further filter the columns whose schema labels appear less than 10 times in the table corpus, as there are not enough data tables that can be used to train the
model to recognize these labels. We experiment on 15, 252 data tables, with a total number of columns equal to 82, 981. The total number of schema labels is equal to 1088.


## Training with SeLaB

To train the semantic labeling model with `SeLaB`:

Split the data to train and test splits:
```bash
python train_test_split.py
```
Train `SeLaB` model with the default parameters
```bash
python selab_train.py
```
Test `SeLaB` model with the default parameters:
```bash
python selab_test.py
```

## Reference

If you plan to use `SeLaB` in your project, please consider citing [our paper](https://ieeexplore.ieee.org/abstract/document/9534408):

```bash
@INPROCEEDINGS{trabelsi21ijcnn,
  author={Trabelsi, Mohamed and Cao, Jin and Heflin, Jeff},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={SeLaB: Semantic Labeling with BERT}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN52387.2021.9534408}}
```
 ## Contact
  
  if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu
