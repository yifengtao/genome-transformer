""" Base model of GIT and its variants.

"""
import torch
import torch.nn as nn

__author__ = "Yifeng Tao"


class ModelBase(nn.Module):
  """ Base models for all models.

  """

  def __init__(self, args):
    """ Initialize the hyperparameters of model.

    Parameters
    ----------
    args: arguments for initializing the model.

    """

    super(ModelBase, self).__init__()

    self.epsilon = 1e-4

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
    """ Define modules of the model.

    """

    raise NotImplementedError


  def forward(self):
    """ Define the data flow across modules of the model.

    """

    raise NotImplementedError


  def train(self):
    """ Train the model using training set.

    """

    raise NotImplementedError


  def test(self):
    """ Test the model using test set.

    """
    raise NotImplementedError


  def load_model(self, path="data/trained_model.pth"):
    """ Load trained parameters of the model.

    """

    print("Loading model from "+path)
    self.load_state_dict(torch.load(path))


  def save_model(self, path="data/trained_model.pth"):
    """ Save learnable parameters of the trained model.

    """

    print("Saving model to "+path)
    torch.save(self.state_dict(), path)

