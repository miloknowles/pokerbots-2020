import os
import argparse
import torch

file_dir = os.path.dirname(__file__)


class Options(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Deep CFR Options")

    self.parser.add_argument("--EXPERIMENT_NAME",
                type=str,
                help="Name for this training experiment",
                default="deep_cfr_default")
    self.parser.add_argument("--NUM_CFR_ITERS",
                type=int,
                help="Number of CFR iterations to do",
                default=100)
    self.parser.add_argument("--NUM_TRAVERSALS_PER_ITER",
                type=int,
                help="Number of game tree traversals to do per CFR iter",
                default=30000)
    self.parser.add_argument("--MEM_BUFFER_MAX_SIZE",
                type=int,
                help="Maximum size of a memory buffer",
                default=1e5)
    self.parser.add_argument("--SINGLE_PROC_MEM_BUFFER_MAX_SIZE",
                type=int,
                help="Size of mem buffers for a single worker",
                default=1e5)
    self.parser.add_argument("--NUM_TRAVERSE_WORKERS",
                type=int,
                help="Should <= number of available CPU cores",
                default=2)
    self.parser.add_argument("--TRAVERSE_DEBUG_PRINT_HZ",
                type=int,
                help="Print out debug statement after this many traversals",
                default=500)
    self.parser.add_argument("--EMBED_DIM",
                type=int,
                help="Size of vector embedding for cards",
                default=64)
    self.parser.add_argument("--SGD_ITERS",
                type=int,
                help="Training gradient descent update iterations",
                default=32000)
    self.parser.add_argument("--SGD_LR",
                type=float,
                help="Learning rate",
                default=1e-3)
    self.parser.add_argument("--SGD_BATCH_SIZE",
                type=int,
                help="Batch size for training networks",
                default=20000)
    self.parser.add_argument("--TRAIN_DATASET_SIZE",
                type=int,
                help="The number of training examples to hold in memory at a time",
                default=1e6)
    self.parser.add_argument("--NUM_DATA_WORKERS",
                type=int,
                help="Number of dataset batch fetching workers",
                default=4)

  #=========================== TRAVERSAL PARAMS ==============================
  # NUM_CFR_ITERS = 100               # Exploitability seems to converge around 100 iters.
  # NUM_TRAVERSALS_PER_ITER = 30000   # 100k seems to be the best in Brown et. al.
  # MEM_BUFFER_MAX_SIZE = 1e5         # Brown. et. al. use 40 million for all 3 buffers.
  # SINGLE_PROC_MEM_BUFFER_MAX_SIZE = 1e5
  # NUM_TRAVERSE_WORKERS = 8
  # TRAVERSE_DEBUG_PRINT_HZ = 500

  # #=========================== NETWORK PARAMS ===============================
  # DEVICE = torch.device("cuda:0")
  # EMBED_DIM = 64                   # Seems like 128 gave the best performance.
  # SGD_ITERS = 32000                 # Same as Brown et. al.
  # SGD_LR = 1e-3                     # Same as Brown et. al.
  # SGD_BATCH_SIZE = 20000            # Same as Brown et. al.
  # TRAIN_DATASET_SIZE = 1e6          # TODO(milo): Try something bigger?
  # NUM_DATA_WORKERS = 0

  def setup_after_parse(self, options):
    options.MEMORY_FOLDER = os.path.join("./memory/", options.EXPERIMENT_NAME)
    options.TRAIN_LOG_FOLDER = os.path.join("./training_logs/", options.EXPERIMENT_NAME)

    # Call THIS_FMT.format(PLAYER_UID) to get the right name.
    options.ADVT_BUFFER_FMT = "advt_mem_{}"
    options.STRAT_BUFFER_FMT = "strt_mem"  

  def parse(self):
    """
    Parses from the command line.
    """
    self.options = self.parser.parse_args()
    self.setup_after_parse(self.options)

    return self.options

  def parse_default(self):
    """
    Useful for getting options without using the command line (i.e in a test suite).
    """
    self.options = self.parser.parse_args([])
    self.setup_after_parse(self.options)

    return self.options

