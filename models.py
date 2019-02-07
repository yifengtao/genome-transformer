""" GIT model and its variants.
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

from base import ModelBase

__author__ = 'Yifeng Tao'

class GIT(ModelBase):
  """ GIT model and its variants.

  """
  def __init__(self, args, **kwargs):
    """ Initialize the model.

    Parameters
    ----------
    args: arguments for initializing the model.

    """
    super(GIT, self).__init__(args, **kwargs)


  def build(self):
    """ Build the model.
    Parameters
    ----------
    sga: list of list
      SGA index in each tumors

    """

    self.NUM_CAN_TYPE = 16

    self.layer_sga_emb = nn.Embedding(
        num_embeddings=self.sga_size+1,
        embedding_dim=self.embedding_size,
        padding_idx=0)

    self.layer_can_emb = nn.Embedding(
        num_embeddings=self.can_size+1,
        embedding_dim=self.embedding_size,
        padding_idx=0)

    self.layer_w_0 = nn.Linear(
        in_features=self.embedding_size,
        out_features=self.attention_size,
        bias=True)

    self.layer_beta = nn.Linear(
        in_features=self.attention_size,
        out_features=self.attention_head,
        bias=True)

    self.layer_dropout_1 = nn.Dropout(p=self.dropout_rate)

    self.layer_w_1 = nn.Linear(
        in_features=self.embedding_size,
        out_features=self.hidden_size,
        bias=True)

    self.layer_dropout_2 = nn.Dropout(p=self.dropout_rate)

    self.layer_w_2 = nn.Linear(
        in_features=self.hidden_size,
        out_features=self.deg_size,
        bias=True)

    if self.initializtion:
      pretrained_gene_emb = np.loadtxt(
          open(os.path.join(self.input_dir, 'init_emb_new.csv')),
          delimiter=",")
      self.layer_sga_emb.weight.data.copy_(torch.from_numpy(pretrained_gene_emb))

    self.optimizer = optim.Adam(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay)


  def forward(self, sga_index, can_index):
    """Forward process.

    Parameters
    ----------
    sga_index: list of SGA index vectors.
    can_index: list of cancer type index vectors.

    Returns
    -------
    Loss of this process, a pytorch variable.
    """
    # Embedding of specific cancer type
    # (batch_size, 1, self.emb_dim)
    emb_can = self.layer_can_emb(can_index)
    # (batch_size, self.emb_dim)
    emb_can = emb_can.view(-1,self.embedding_size)

    # Embeddings of mutations
    # (batch_size, max_len=1000, self.emb_dim)
    E_t = self.layer_sga_emb(sga_index)

    # Flattened E_t
    # (batch_size * max_len=1000, self.emb_dim)
    E_t_flatten = E_t.view(-1, self.embedding_size)
    # E_t1_flatten
    # (batch_size * max_len=1000, self.attention_size)
    E_t1_flatten = torch.tanh( self.layer_w_0(E_t_flatten) )
    # E_t2_flatten
    # (batch_size * max_len=1000, self.attention_head)
    E_t2_flatten = self.layer_beta(E_t1_flatten)
    # E_t2
    # (batch_size, max_len=1000, self.attention_head)


    E_t2 = E_t2_flatten.view(-1, self.num_max_sga, self.attention_head)

    # The code bellow doesn't work in some version of PyTorch:
    # F.softmax(E_t2,dim=1)
    # So we switch the first and second dim of tensor E_t2
    E_t2 = E_t2.permute(1,0,2)
    # (max_len=1000, batch_size, self.attention_head)
    A = F.softmax(E_t2)
    # A: attention matrix
    # (batch_size, max_len=1000, self.attention_head)
    A = A.permute(1,0,2)

    if self.attention:
      # Multi-head attention weighted sga embedding:
      # emb_sga = sum(A^T * E_t, dim=1)
      # where A^T * E_t: (batch_size, self.attention_head, self.emb_dim)
      emb_sga = torch.sum( torch.bmm( A.permute(0,2,1), E_t ), dim=1)
      # (batch_size, self.emb_dim)
      emb_sga = emb_sga.view(-1,self.embedding_size)
    else:
      #E_t = self.layer_sga_emb(sga_index)
      emb_sga = torch.sum(E_t, dim=1)
      emb_sga = emb_sga.view(-1, self.embedding_size)

    # Embedding of tumor:
    # (batch_size, self.emb_dim)
    if self.cancer_type:
      emb_tmr = emb_can+emb_sga
    else:
      emb_tmr = emb_sga

    emb_tmr_relu = self.layer_dropout_1(F.relu(emb_tmr))

    hid_tmr = self.layer_w_1(emb_tmr_relu)
    hid_tmr_relu = self.layer_dropout_2(F.relu(hid_tmr))

    attn_wt = torch.sum(A, dim=2)
    attn_wt = attn_wt.view(-1, self.num_max_sga)

    return F.sigmoid(self.layer_w_2(hid_tmr_relu)), emb_tmr, hid_tmr, emb_sga, attn_wt


  def train(
      self, train_set, test_set,
      batch_size=None, test_batch_size=None,
      max_iter=None, max_fscore=None,
      test_inc_size=1, **kwargs):
    """
    Parameters
    ----------
    train_set: dict
    test_set: dict
    test_batch_size: int
    max_iter: int
    test_inc_size: int
      interval of running an evaluation

    """

    for iter_train in range(0, max_iter+1, batch_size):
      batch_set = get_minibatch_dataset(
          train_set, iter_train, batch_size,batch_type='train')

      preds, _, _, _, _ = self.forward(
          batch_set['sga'],
          batch_set['can']
          )
      labels = batch_set['deg']

      self.optimizer.zero_grad()

      loss = -torch.log( self.EPSILON +1-torch.abs(preds-labels) ).mean()

      loss.backward()
      self.optimizer.step()

      if test_inc_size and (iter_train % test_inc_size == 0):
        preds, labels, _, _, _, _, _ = self.test(test_set, test_batch_size)

        precision, recall, f1score, accuracy = evaluate(
            labels, preds)

        print('[%d,%d], f1_score: %.3f, acc: %.3f'% (iter_train//len(train_set['can']),
              iter_train % len(train_set['can']), f1score, accuracy)
          )

        if f1score >= max_fscore:
          break
    self.save_model()


  def test(self, test_set, test_batch_size, **kwargs):
    preds = []
    labels = []

    emb_tmr = []
    hid_tmr = []
    emb_sga = []

    attn_wt = []
    tmr = []

    for iter_test in range(0, len(test_set['can']), test_batch_size):
      batch_set = get_minibatch_dataset(
          test_set, iter_test, test_batch_size,batch_type='test')
      batch_preds, batch_emb_tmr, batch_hid_tmr, batch_emb_sga, batch_attn_wt = self.forward(
          batch_set['sga'],
          batch_set['can'])
      batch_labels = batch_set['deg']

      attn_wt.append(batch_attn_wt.data.numpy())

      tmr = tmr + batch_set["tmr"]
      emb_tmr.append(batch_emb_tmr.data.numpy())
      hid_tmr.append(batch_hid_tmr.data.numpy())
      emb_sga.append(batch_emb_sga.data.numpy())

      preds.append(batch_preds.data.numpy())
      labels.append(batch_labels.data.numpy())


    emb_tmr = np.concatenate(emb_tmr,axis=0)
    hid_tmr = np.concatenate(hid_tmr,axis=0)
    attn_wt = np.concatenate(attn_wt,axis=0)
    emb_sga = np.concatenate(emb_sga,axis=0)

    preds = np.concatenate(preds,axis=0)
    labels = np.concatenate(labels,axis=0)

    return preds, labels, emb_tmr, hid_tmr, attn_wt, emb_sga, tmr


