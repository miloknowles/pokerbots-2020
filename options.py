import os
import argparse
import torch

file_dir = os.path.dirname(__file__)


class Options(object):
  """
  TODO(milo): Make a command line interface.
  """
  EXPERIMENT_NAME = "deep_cfr_paper"

  NUM_CFR_ITERS = 100               # Exploitability seems to converge around 100 iters.
  NUM_TRAVERSALS_PER_ITER = 1e5     # 100k seems to be the best in Brown et. al.
  MEM_BUFFER_MAX_SIZE = 1e6         # Brown. et. al. use 40 million for all 3 buffers.
  EMBED_DIM = 128                   # Seems like this gave the best performance.

  SGD_ITERS = 32000                 # Same as Brown et. al.
  SGD_LR = 1e-3                     # Same as Brown et. al.
  SGD_BATCH_SIZE = 20000            # Same as Brown et. al.
  TRAIN_DATASET_SIZE = 1e6          # TODO(milo): Try something bigger?

  DEVICE = torch.device("cuda")
  NUM_DATA_WORKERS = 0

  MEMORY_FOLDER = os.path.join("./memory/", EXPERIMENT_NAME)
  TRAIN_LOG_FOLDER = os.path.join("./training_logs/", EXPERIMENT_NAME)

  ADVT_BUFFER_FMT = "advt_mem_{}"
  STRAT_BUFFER_FMT = "strat_mem_{}"

  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Deep CFR Options")
