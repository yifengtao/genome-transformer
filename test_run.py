""" Demo of training and evaluating GIT model and its variants.

"""
import os
import argparse

from utils import bool_ext, load_dataset, split_dataset, evaluate
from models import GIT

__author__ = "Yifeng Tao"


parser = argparse.ArgumentParser()

parser.add_argument("--train_model", help="whether to train model or load model", type=bool_ext, default=True)

parser.add_argument("--input_dir", help="directory of input files", type=str, default="data")
parser.add_argument("--output_dir", help="directory of output files", type=str, default="data")

parser.add_argument("--embedding_size", help="embedding dimension of genes and tumors", type=int, default=512)
parser.add_argument("--hidden_size", help="hidden layer dimension of MLP decoder", type=int, default=1024)
parser.add_argument("--attention_size", help="size of attention parameter beta_j", type=int, default=400)
parser.add_argument("--attention_head", help="number of attention heads", type=int, default=128)

parser.add_argument("--learning_rate", help="learning rate for Adam", type=float, default=1e-4)
parser.add_argument("--max_iter", help="maximum number of training iterations", type=int, default=3072*20)
parser.add_argument("--max_fscore", help="Max F1 score to early stop model from training", type=float, default=0.7)
parser.add_argument("--batch_size", help="training batch size", type=int, default=16)
parser.add_argument("--test_batch_size", help="test batch size", type=int, default=512)
parser.add_argument("--test_inc_size", help="increment interval size between log outputs", type=int, default=256)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.5)
parser.add_argument("--weight_decay", help="coefficient of l2 regularizer", type=float, default=1e-5)

# GIT variants:
# args.initializtion=False -> GIT-init
parser.add_argument("--initializtion", help="whether to use pre-trained gene embeddings or not", type=bool_ext, default=True)
# args.attention=False -> GIT-attn
parser.add_argument("--attention", help="whether to use attention mechanism or not", type=bool_ext, default=True)
# args.cancer_type=False -> GIT-can
parser.add_argument("--cancer_type", help="whether to use cancer type or not", type=bool_ext, default=True)
# args.deg_shuffle=False -> DEG-shuffled
parser.add_argument("--deg_shuffle", help="whether to shuffle DEGs or not", type=bool_ext, default=False)

args = parser.parse_args()

# Above default hyper parameters tuned in full GIT model;
# Ablated GIT variants have slight different tuned hyper parameters:

#if args.cancer_type == False:
#  args.max_iter = 3072*40
#elif args.attention == False:
#  args.max_iter = 3072*40
#  args.learning_rate = 0.0003

print("Loading dataset...")
dataset = load_dataset(input_dir=args.input_dir, deg_shuffle=args.deg_shuffle)
train_set, test_set = split_dataset(dataset, ratio=0.66)

args.can_size = dataset["can"].max()        # cancer type dimension
args.sga_size = dataset["sga"].max()        # SGA dimension
args.deg_size = dataset["deg"].shape[1]     # DEG output dimension
args.num_max_sga = dataset["sga"].shape[1]  # maximum number of SGAs in a tumor

print("Hyperparameters:")
print(args)

model = GIT(args)                           # initialize GIT model
model.build()                               # build GIT model

if args.train_model:                        # train from scratch
  print("Training...")
  model.train(train_set, test_set,
      batch_size=args.batch_size,
      test_batch_size=args.test_batch_size,
      max_iter=args.max_iter,
      max_fscore=args.max_fscore,
      test_inc_size=args.test_inc_size)
  model.load_model(os.path.join(args.output_dir, "trained_model.pth"))
else:                                      # or directly load trained model
  model.load_model(os.path.join(args.input_dir, "trained_model.pth"))

# evaluation
print("Evaluating...")
labels, preds, _, _, _, _, _ = model.test(test_set, test_batch_size=args.test_batch_size)
precision, recall, f1score, accuracy = evaluate(labels, preds, epsilon=model.epsilon)
print("prec=%.3f, recall=%.3f, F1=%.3f, acc=%.3f"%(precision, recall, f1score, accuracy))
# prec=0.702, recall=0.565, F1=0.626, acc=0.788

