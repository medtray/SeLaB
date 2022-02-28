import string
from gensim.models.doc2vec import Doc2Vec
from gensim.models import FastText
import numpy as np
import os
from transformers import *
import json
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
import torch

stop_words = set(stopwords.words('english'))
cwd = os.getcwd()

sys_home = os.path.join(cwd,'outputs')
if not os.path.exists(sys_home):
    os.mkdir(sys_home)

data_loc=os.path.join(cwd,'data')

output_folder=os.path.join(sys_home,'column_predict_WikiTablesCollection')
data_folder=os.path.join(data_loc,'WikiTablesCollection')
data_folder_debug=os.path.join(data_loc,'WikiTablesCollectionDebug')


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(data_folder_debug):
    os.mkdir(data_folder_debug)

train_features_path =os.path.join(output_folder,'train_features.json')
test_features_path =os.path.join(output_folder,'test_features.json')

word2int_path = os.path.join(output_folder,'word2int.npy')

train_files=os.path.join(output_folder,'train_files.npy')
test_files=os.path.join(output_folder,'test_files.npy')

test_debug=os.path.join(output_folder,'test_debug.npy')
train_debug=os.path.join(output_folder,'train_debug.npy')

seen_labels_file=os.path.join(output_folder,'seen_labels.npy')
freq_seen_labels_file=os.path.join(output_folder,'freq_seen_labels.npy')

model_file = os.path.join(output_folder,'selab_model.pt')
