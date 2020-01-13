""" Prepare the dataset.pkl from more widely used data format.

"""

import os
import argparse
import numpy as np
import pickle
import pandas as pd

__author__ = "Yifeng Tao"


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="directory of input files", type=str, default="mydata")
parser.add_argument("--output_dir", help="directory of output files", type=str, default="mydata")
args = parser.parse_args()


# input file 1: DEG matrix in csv format
df = pd.read_csv(os.path.join(args.input_dir, "deg.csv"), index_col=0)

tmr = list(df.index)

idx2deg = {i:d for i, d in enumerate(df.columns)}

deg = df.values


# input file 2: cancer type list in txt format
can = []
with open(os.path.join(args.input_dir,"cancer_type.txt"), "r") as f:
  for line in f:
    can.append(line.strip())

idx2can = {i:c for i, c in enumerate(list(set(can)))}
can2idx = {idx2can[i]:i for i in idx2can.keys()}

can = [can2idx[c] for c in can]

can = np.array(can)


# input file 3: SGA list in txt format
sga = []
with open(os.path.join(args.input_dir, "sga.txt"), "r") as f:
  next(f)
  for line in f:
    line = line.strip().split(", ")
    sga.append(line[1:])

idx2sga = {i:s for i,s in enumerate(list(set([l for line in sga for l in line])))}
sga2idx = {idx2sga[i]:i for i in idx2sga.keys()}

sga = [[sga2idx[l] for l in line] for line in sga]


# output file: dictionary in pickle format
dataset = {
    "can": can,
    "deg": deg,
    "idx2can": idx2can,
    "idx2deg": idx2deg,
    "idx2sga": idx2sga,
    "sga": sga,
    "tmr": tmr}

with open(os.path.join(args.output_dir, "dataset.pkl"), "wb") as f:
  pickle.dump(dataset, f, protocol=2)

