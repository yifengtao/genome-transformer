""" Shared utilities for models.py and test_run.py.

"""
import os
import random
import numpy as np
import pickle

import torch
from torch.autograd import Variable

__author__ = "Yifeng Tao"


def bool_ext(rbool):
  """ Solve the problem that raw bool type is always True.

  Parameters
  ----------
  rbool: str
    should be True of False.

  """

  if rbool not in ["True", "False"]:
    raise ValueError("Not a valid boolean string")

  return rbool == "True"


def load_dataset(input_dir="data", deg_shuffle=False):
  """ Load dataset, modify, and shuffle`.

  Parameters
  ----------
  input_dir: str
    input directory of dataset
  deg_shuffle: bool
    whether to shuffle DEG or not

  Returns
  -------
  dataset: dict
    dict of lists, including SGAs, cancer types, DEGs, patient barcodes
  """

  # load dataset
  data = pickle.load( open(os.path.join(input_dir, "dataset.pkl"), "rb") )
  can_r = data["can"] # cancer type index of tumors: list of int
  sga_r = data["sga"] # SGA index of tumors: list of list
  deg = data["deg"]   # DEG binary matrix of tumors: 2D array of 0/1
  tmr = data["tmr"]   # barcodes of tumors: list of str

  # shift the index of cancer type by +1, 0 is for padding
  can = np.asarray([[x+1] for x in can_r], dtype=int)

  # shift the index of SGAs by +1, 0 is for padding
  num_max_sga = max([len(s) for s in sga_r])
  sga = np.zeros( (len(sga_r), num_max_sga), dtype=int )
  for idx, line in enumerate(sga_r):
    line = [s+1 for s in line]
    sga[idx,0:len(line)] = line

  # shuffle DEGs
  if deg_shuffle:
    rng = list(range(deg.shape[1]))
    for idx in range(deg.shape[0]):
      random.shuffle(rng)
      deg[idx] = deg[idx][rng]

  # shuffle whole dataset
  rng = list(range(len(can)))
  random.Random(2019).shuffle(rng)
  can = can[rng]
  sga = sga[rng]
  deg = deg[rng]
  tmr = [tmr[idx] for idx in rng]

  dataset = {"can":can, "sga":sga, "deg":deg, "tmr":tmr}

  return dataset


def split_dataset(dataset, ratio=0.66):
  """ Split the dataset according to the ratio of training/test sets.

  Parameters
  ----------
  dataset: dict
    dict of lists, including SGAs, cancer types, DEGs, patient barcodes
  ratio: float
    size(train_set)/size(train_set+test_set)

  Returns
  -------
  train_set, test_set: dict

  """

  num_sample = len(dataset["can"])
  num_train_sample = int(num_sample*ratio)

  train_set = {"sga":dataset["sga"][:num_train_sample],
               "can":dataset["can"][:num_train_sample],
               "deg":dataset["deg"][:num_train_sample],
               "tmr":dataset["tmr"][:num_train_sample]}
  test_set = {"sga":dataset["sga"][num_train_sample:],
              "can":dataset["can"][num_train_sample:],
              "deg":dataset["deg"][num_train_sample:],
              "tmr":dataset["tmr"][num_train_sample:]}

  return train_set, test_set


def wrap_dataset(sga, can, deg, tmr):
  """ Wrap default numpy or list data into PyTorch variables.

  """

  dataset = {"sga": Variable(torch.LongTensor(sga)),
             "can": Variable(torch.LongTensor(can)),
             "deg": Variable(torch.FloatTensor(deg)),
             "tmr": tmr}

  return dataset


def get_minibatch(dataset, index, batch_size, batch_type="train"):
  """ Get a mini-batch dataset for training or test.

  Parameters
  ----------
  dataset: dict
    dict of lists, including SGAs, cancer types, DEGs, patient barcodes
  index: int
    starting index of current mini-batch
  batch_size: int
  batch_type: str
    batch strategy is slightly different for training and test
    "train": will return to beginning of the queue when `index` out of range
    "test": will not return to beginning of the queue when `index` out of range

  Returns
  -------
  batch_dataset: dict
    a mini-batch of the input `dataset`.

  """

  sga = dataset["sga"]
  can = dataset["can"]
  deg = dataset["deg"]
  tmr = dataset["tmr"]

  if batch_type == "train":
    batch_sga = [ sga[idx%len(sga)]
      for idx in range(index,index+batch_size) ]
    batch_can = [ can[idx%len(can)]
      for idx in range(index,index+batch_size) ]
    batch_deg = [ deg[idx%len(deg)]
      for idx in range(index,index+batch_size) ]
    batch_tmr = [ tmr[idx%len(tmr)]
      for idx in range(index,index+batch_size) ]
  elif batch_type == "test":
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


def evaluate(labels, preds, epsilon=1e-4):
  """ Calculate performance metrics given ground truths and prediction results.

  Parameters
  ----------
  labels: matrix of 0/1
    ground truth labels
  preds: matrix of float in [0,1]
    predicted labels
  epsilon: float
    a small Laplacian smoothing term to avoid zero denominator

  Returns
  -------
  precision: float
  recall: float
  f1score: float
  accuracy: float

  """

  flat_labels = np.reshape(labels,-1)
  flat_preds = np.reshape(np.around(preds),-1)

  accuracy = np.mean(flat_labels == flat_preds)
  true_pos = np.dot(flat_labels, flat_preds)
  precision = 1.0*true_pos/flat_preds.sum()
  recall = 1.0*true_pos/flat_labels.sum()

  f1score = 2*precision*recall/(precision+recall+epsilon)

  return precision, recall, f1score, accuracy

