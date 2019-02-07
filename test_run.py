""" Demo code of GIT and its variants.
Note: Very raw code, will continue to sort these days.
"""
import numpy as np
import pickle
import os
import random
import argparse

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import load_dataset, split_dataset, evaluate
from models import GIT

__author__ = 'Yifeng Tao'

#TODO:
# single ' to double "
#TODO:
# change:
# pickle.dump(w, open("a.pkl","wb"), protocol=2)
#TODO:
# prepare data

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', help='directory of input files', type=str, default='data/input')
parser.add_argument('--output_dir', help='directory of output files', type=str, default='data/output')

parser.add_argument('--max_fscore', help='Max F1 score when model will stop training', type=float, default=0.7)

parser.add_argument('--embedding_size', help='embedding dimension of genes and tumors', type=int, default=512)
parser.add_argument('--hidden_size', help='hidden layer dimension of MLP decoder', type=int, default=1024)
parser.add_argument('--attention_size', help='size of attention parameter beta_j', type=int, default=400)
parser.add_argument('--attention_head', help='number of attention heads', type=int, default=128)

parser.add_argument('--learning_rate', help='learning rate for Adam', type=float, default=1e-4)
parser.add_argument('--max_iter', help='maximum number of training iterations', type=int, default=3072*20)
parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
parser.add_argument('--test_batch_size', help='test batch size', type=int, default=512)
parser.add_argument('--test_inc_size', help='increment interval size between log outputs', type=int, default=256)
parser.add_argument('--dropout_rate', help='dropout rate', type=float, default=0.5)
parser.add_argument('--weight_decay', help='coefficient of l2 regularizer', type=float, default=1e-5)

# GIT variants
parser.add_argument('--initializtion', help = 'whether to use pre-trained gene embeddings or not', type=bool, default=True)
parser.add_argument('--attention', help = 'whether to use attention mechanism or not', type=bool, default=True)
parser.add_argument('--cancer_type', help = 'whether to use cancer type or not', type=bool, default=True)
parser.add_argument('--deg_shuffle', help = 'whether to shuffle DEGs or not', type=bool, default=False)


args = parser.parse_args()

#The above default parameters are tuned hyperparameters of full GIT model.
#To run ablated GIT variants, the tuned hyperparams are (if not listed, it means
#the same with full GIT model):
if args.cancer_type == False:
  args.max_iter = 3072*40
elif args.attention == False:
  args.max_iter = 3072*40
  args.learning_rate = 0.0003

dataset = load_dataset(path_input="data", deg_shuffle=args.deg_shuffle)
train_set, test_set = split_dataset(dataset, train_ratio=0.66)

args.can_size = dataset['can'].max() # cancer type dimension
args.sga_size = dataset['sga'].max() # SGA dimension
args.deg_size = dataset['deg'].shape[1] # DEG output dimension
args.num_max_sga = dataset['sga'].shape[1] # maximum number of SGAs in a tumor

print(args)

# Train from scratch.

model = GIT(args) #Initialize GIT model
model.build() #Build GIT model
#model.load_model() #Load pretrained model
model.train(
    train_set, test_set,
    batch_size=args.batch_size,
    test_batch_size=args.test_batch_size,
    max_iter=args.max_iter,
    max_fscore=args.max_fscore,
    test_inc_size=args.test_inc_size)


gene_emb_matrix = model.layer_sga_emb.weight.data.numpy()[1:]
model.load_model()
preds, labels, emb_tmr, hid_tmr, attn_wt, emb_sga, tmr = model.test(test_set, test_batch_size=args.test_batch_size)
precision, recall, f1score, accuracy = evaluate(labels, preds)

# Load trained model and extract features.

# show how to get A, emb_tmr, hid_tmr, refined SGA embedding

#'GIT', # full GIT
#'model_ccle', # GIT on CCLE dataset
#'GIT_fake_deg', # GIT w/ DEG shuffled data
#'GIT_wo_init', # GIT - init
#'GIT_wo_attn', # GIT - attn
#'GIT_wo_can', # GIT - can

# Hyperparameters and configurations of all models, will be instantiated or
# revised according to model_name.

# Fine-tuned hyperparamters for each model

