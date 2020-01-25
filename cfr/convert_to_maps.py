import pickle
import os
import argparse
import torch

if __name__ == "__main__":
  filename = "./07/avg_strategy_0.pkl"
  # filename = "./total_regrets_0.pkl"

  d = {}
  with open(filename, "rb") as f:
    d = pickle.load(f)

  print("Loaded {} items from pickle file {}".format(len(d), filename))

  folder = os.path.abspath(os.path.dirname(filename))
  with open(os.path.join(folder, "avg_strategy.txt"), "w") as f:
    for key in d:
      space_sep = " ".join([str(v) for v in list(d[key].numpy())])
      f.write("{} {}\n".format(key, space_sep))

  print("Done.")
