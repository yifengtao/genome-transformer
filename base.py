""" Base model.
Note: Very raw code, will continue to sort these days.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import random
import pickle
from collections import Counter
import os
from sklearn.metrics import f1_score

from utils import get_minibatch_dataset, evaluate

__author__ = 'Yifeng Tao'


class ModelBase(nn.Module):
  """ Base models for all models.

  """
  def __init__(self, args):
    """ Initialize the model.

    Parameters
    ----------
    args: arguments for initializing the model.

    """
    super(ModelBase, self).__init__()

    self.EPSILON = 1e-4                                # Numerical stability

    self.input_dir = args.input_dir
    self.output_dir = args.output_dir

    self.sga_size = args.sga_size
    self.deg_size = args.deg_size
    self.can_size = args.can_size

    self.num_max_sga = args.num_max_sga

    self.embedding_size = args.embedding_size
    self.hidden_size = args.hidden_size
    self.attention_size = args.attention_size
    self.attention_head = args.attention_head

    self.learning_rate = args.learning_rate
    self.dropout_rate = args.dropout_rate
    self.weight_decay = args.weight_decay

    self.initializtion = args.initializtion
    self.attention = args.attention
    self.cancer_type = args.cancer_type
    self.deg_shuffle = args.deg_shuffle

  def build(self):
    raise NotImplementedError

  def forward(self):
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError

  def load_model(self, path="trained_model.pth"):
    """ Load pretrained parameters of model.

    """
    print('Loading model from '+os.path.join(self.output_dir, path))
    self.load_state_dict(torch.load( os.path.join(self.output_dir, path) ))

  def save_model(self, path='trained_model.pth'):
    """ Save learnable parameters of the trained model.

    """
    print('Saving model to '+os.path.join(self.output_dir, path))
    torch.save(self.state_dict(), os.path.join(self.output_dir, path))


#62.7 78.7
