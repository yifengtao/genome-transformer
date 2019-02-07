""" Shared utilities for experiment.py and model.py.
Note: Very raw code, will continue to sort these days.
"""
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import random

__author__ = 'Yifeng Tao'

#TODO:
#pickle.dump(new_data, open("dataset.pkl", "wb"), protocol=2)

def load_dataset(path_input="data",deg_shuffle=False):
  """ Load data shuffle, and modify to adapt according to the `self.config`.

  """
  #TODO: comment the definitions of dataset format
  #new_data = {
  #    'sga':sga_index,
  #    'idx2sga':idx2sga,
  #    'can':can_index,
  #    'idx2can':idx2can,
  #    'deg':deg,
  #    'idx2deg':idx2deg,
  #    'tmr':barcode}

  data = pickle.load( open(os.path.join(path_input, "dataset.pkl"), "rb") )
  can_index = data["can"]
  sga_index = data["sga"]
  deg = data["deg"]
  tmr = data["tmr"]

  # Shift the index of cancer type and sga by +1, since 0 will be for padding.
  can = np.asarray([[x+1] for x in can_index], dtype=int)

  #MAX_DIM_EMB = 1000 #SGAs in a tumor: 4 ~ 997, 997+16=1013 < 1100
  num_max_sga = max([len(s) for s in sga_index])
  # (num_tumor, MAX_DIM_EMB) = (4468, 1000), value draw from 1 to 19781
  sga = np.zeros( (len(sga_index), num_max_sga), dtype=int )

  for idx, line in enumerate(sga_index):
    line = [s+1 for s in line]
    sga[idx,0:len(line)] = line

  if deg_shuffle:
    rng = list(range(deg.shape[1]))
    for idx in range(deg.shape[0]):
      random.shuffle(rng)
      deg[idx] = deg[idx][rng]

  rng = list(range(len(can)))
  #random.seed(2019)
  random.Random(2019).shuffle(rng)

  can = can[rng]
  sga = sga[rng]
  deg = deg[rng]
  tmr = [tmr[idx] for idx in rng]

  dataset = {'can':can, 'sga':sga, 'deg':deg, "tmr":tmr}

  return dataset


def split_dataset(dataset, train_ratio=0.66):
  """ Split the dataset according to the ratio of training/test set.

  Parameters
  ----------
  dataset: dict
  train_ratio: float
    train_set/test_set

  """
  num_sample = len(dataset['can'])
  num_train_sample = int(num_sample*train_ratio)

  train_set = {'sga':dataset['sga'][:num_train_sample],
               'can':dataset['can'][:num_train_sample],
               'deg':dataset['deg'][:num_train_sample],
               "tmr":dataset['tmr'][:num_train_sample]}
  test_set = {'sga':dataset['sga'][num_train_sample:],
              'can':dataset['can'][num_train_sample:],
              'deg':dataset['deg'][num_train_sample:],
              'tmr':dataset['tmr'][num_train_sample:]}

  return train_set, test_set


def wrap_dataset(sga, can, deg, tmr):
  """ Wrap default list or np data into PyTorch Variables.

  """
  dataset = {'sga': Variable(torch.LongTensor( sga )),
             'can': Variable(torch.LongTensor( can )),
             'deg': Variable(torch.FloatTensor( deg )),
             "tmr": tmr}
  return dataset


def get_minibatch_dataset(dataset, index, batch_size, batch_type='train'):
  """ Get a mini-batch of dataset for training and test.

  Parameters
  ----------
  dataset: dict
    Dict of lists, including SGAs, cancer types, DEGs, patient barcode etc.
  index: int
    Starting index of current mini-batch
  batch_size: int
  batch_type: string
    Batch strategy will be slightly different during training and test.
    'train': will return to beginning of the queue when index out of range
    'test': will not return to beginning of the queue when index out of range

  Returns
  -------
  batch_dataset: dict
    A mini-batch of the input `dataset'.
    Dict of lists, including SGAs, cancer types, DEGs, patient barcode etc.

  """

  sga = dataset['sga']
  can = dataset['can']
  deg = dataset['deg']
  tmr = dataset['tmr']

  if batch_type == 'train':
    batch_sga = [ sga[idx%len(sga)]
      for idx in range(index,index+batch_size) ]
    batch_can = [ can[idx%len(can)]
      for idx in range(index,index+batch_size) ]
    batch_deg = [ deg[idx%len(deg)]
      for idx in range(index,index+batch_size) ]
    batch_tmr = [ tmr[idx%len(tmr)]
      for idx in range(index,index+batch_size) ]
  elif batch_type == 'test':
    batch_sga = sga[index:index+batch_size]
    batch_can = can[index:index+batch_size]
    batch_deg = deg[index:index+batch_size]
    batch_tmr = tmr[index:index+batch_size]

  batch_dataset = wrap_dataset(
      batch_sga,
      batch_can,
      batch_deg,
      batch_tmr)

  return batch_dataset


def evaluate(labels, preds):
  """ Calculate performance metrics given two list of ground truth labels and
  prediction results.

  Parameters
  ----------
  labels: matrix of 0/1
    List of ground truth labels.
  preds: matrix of 0/1
    List of predicted labels.

  Returns
  -------
  precision: float
  recall: float
  f1score: float
  accuracy: float

  """
  EPSILON = 1e-4

  flat_labels = np.reshape(labels,-1)
  flat_preds = np.reshape(np.around(preds),-1)

  accuracy = np.mean( flat_labels == flat_preds )
  true_pos = np.dot(flat_labels, flat_preds)
  precision = 1.0*true_pos/flat_preds.sum()
  recall = 1.0*true_pos/flat_labels.sum()

  f1score = 2*precision*recall/(precision+recall+EPSILON)

  return precision, recall, f1score, accuracy

