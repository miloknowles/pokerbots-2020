import torch
from torch.utils.data import Dataset

from memory import *


class MemoryBufferDataset(Dataset):
  def __init__(self, folder, buffer_name, n):
    """
    A PyTorch dataset for loading infosets and targets from disk. Because there are too many to fit
    into memory at once, we resample a dataset of size n << N periodically by choosing n random
    items from all N on disk.
    
    folder (str) : The folder where a manifest .csv is for the memory buffer.
    buffer_name (str) : The name of this buffer (used a prefix in files).
    n (int) : The container size of this dataset (n << N).
    """
    self._folder = folder
    self._buffer_name = buffer_name
    self._n = int(n)

    self._manifest_df = get_manifest_entries(self._folder, self._buffer_name)
    self._N = int(self._manifest_df["num_entries"].sum())

    self._infosets = None
    self._weights = None
    self._items = None
    print(">> MemoryBufferDataset | folder={} | name={} | size(n)={} | total(N)={}".format(
      self._folder, self._buffer_name, self._n, self._N))

  def resample(self):
    """
    Resample the dataset by sampling n items from the N total items on disk.
    """
    self._infosets = None
    self._weights = None
    self._items = None

    idx = torch.randint(0, self._N, (self._n,)).long()
    cumul_idx = 0
    for i in self._manifest_df.index:
      num_entries = self._manifest_df["num_entries"][i]
      idx_this_file = (idx[idx.ge(cumul_idx) * idx.le(cumul_idx + num_entries - 1)] - cumul_idx).long()
      d = torch.load(self._manifest_df["filename"][i])
      if self._infosets is None:
        self._infosets = d["infosets"][idx_this_file]
      else:
        self._infosets = torch.cat([self._infosets, d["infosets"][idx_this_file]], axis=0)
      if self._weights is None:
        self._weights = d["weights"][idx_this_file]
      else:
        self._weights = torch.cat([self._weights, d["weights"][idx_this_file]], axis=0)
      if self._items is None:
        self._items = d["items"][idx_this_file]
      else:
        self._items = torch.cat([self._items, d["items"][idx_this_file]], axis=0)
      cumul_idx += num_entries

  def __len__(self):
    return self._N

  def __getitem__(self, idx):
    infoset = unpack_infoset(self._infosets[idx])

    hole_cards, board_cards = infoset.get_card_input_tensors()

    # NOTE(milo): This function unsqueezes the first dim for traversal, but the DataLoader will
    # add another batch dimension anyways.
    bets_input, position_mask = infoset.get_bet_input_tensors()
    bets_input = bets_input * position_mask

    return {
      "hole_cards": hole_cards.squeeze(0),
      "board_cards": board_cards.squeeze(0),
      "bets_input": bets_input.squeeze(0),
      "weights": self._weights[idx].unsqueeze(0),
      "target": self._items[idx]
    }
