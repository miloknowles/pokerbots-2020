from sys import getsizeof
import os

import torch
from torch.utils.data import Dataset

import pandas as pd

class InvalidBoardSizeException(Exception):
  pass


class InfoSet(object):
  def __init__(self, hole_cards, board_cards, bet_history_vec, player_position):
    """
    hole_cards (torch.Tensor): The 0-51 encoding of each  of (2) hole cards.
    board_cards (torch.Tensor) : The 0-51 encoding of each of (3-5) board cards.
    bet_history_vec (torch.Tensor) : Betting actions, represented as a fraction of the pot size.
    player_position (int) : 0 if the acting player is the SB and 1 if they are BB.

    The bet history has size (num_streets * num_actions_per_street) = 6 * 4 = 24.
    """
    self.hole_cards = hole_cards
    self.board_cards = board_cards
    self.bet_history_vec = bet_history_vec
    self.player_position = 0

  def get_card_input_tensors(self):
    """
    The network expects (tuple of torch.Tensor):
    Shape ((B x 2), (B x 3)[, (B x 1), (B x 1)]) # Hole, board [, turn, river]).
    """
    if len(self.board_cards) == 0:
      return [self.hole_cards.unsqueeze(0).long(), -1*torch.ones(1, 3).long(),
              -1*torch.ones(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 3:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards.unsqueeze(0).long(),
             -1*torch.ones(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 4:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards[:3].unsqueeze(0).long(),
              self.board_cards[3].view(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 5:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards[:3].unsqueeze(0).long(),
              self.board_cards[3].view(1, 1).long(), self.board_cards[4].view(1, 1).long()]
    else:
      raise InvalidBoardSizeException()

  def get_bet_input_tensors(self):
    """
    The network expects (torch.Tensor) with shape (B x num_betting_actions).
    """
    nbets = self.bet_history_vec.shape[0]
    position_mask = torch.zeros(nbets)
    position_mask[torch.arange(self.player_position, nbets, 2)] = 1
    position_mask[torch.arange((self.player_position + 1) % 2, nbets, 2)] = -1
    return self.bet_history_vec.unsqueeze(0), position_mask
  
  def pack(self):
    """
    Packs the infoset into a compact torch.Tensor of size:
      (1 player position, 2 hole cards, 5 board cards, num_betting_actions)
    """
    board_cards_fixed_size = torch.zeros(5)
    board_cards_fixed_size[:len(self.board_cards)] = self.board_cards
    return torch.cat([
      torch.Tensor([self.player_position]),
      self.hole_cards,
      board_cards_fixed_size,
      self.bet_history_vec])
    

def get_buffer_manifest_path(folder, buffer_name):
  """
  Get the path to a manifest file listing all of the saved memory buffers.

  Args:
    folder (str) : The folder for this experiment.
    buffer_name (str) : The name of this memory buffer (i.e advt_mem_P1)
  """
  return os.path.join(folder, "manifest_{}.csv".format(buffer_name))


def get_buffer_path(folder, buffer_name, index):
  return os.path.join(folder, "{0}_{1:05d}.pth".format(buffer_name, index))


def get_manifest_entries(folder, buffer_name):
  """
  Read in a manifest.csv of buffer entries with the following columns:

    index, filename (relative to manifest directory), num_entries

  Args:
    folder (str) : The folder for this experiment.
    buffer_name (str) : The name of this memory buffer (i.e advt_mem_P1)
  """
  manifest_path = get_buffer_manifest_path(folder, buffer_name)
  if os.path.exists(manifest_path):
    return pd.read_csv(manifest_path, header=0, sep=",", index_col="index")
  else:
    return None


class MemoryBuffer(object):
  def __init__(self, info_set_size, item_size, max_size=80000, store_weights=True,
               device=torch.device("cpu")):
    self._max_size = max_size
    self._info_set_size = info_set_size
    self._item_size = item_size
    self._device = device
    self._infosets = torch.zeros((int(max_size), info_set_size), dtype=torch.float32).to(self._device)
    self._items = torch.zeros((int(max_size), item_size), dtype=torch.float32).to(self._device)
    if store_weights:
      self._has_weights = True
      self._weights = torch.zeros(int(max_size), dtype=torch.float32).to(self._device)
    else:
      self._has_weights = False

    self._next_index = 0

  def add(self, infoset, item):
    if self.full():
      return
    self._infosets[self._next_index] = infoset.pack()
    self._items[self._next_index] = item
    self._next_index += 1

  def add_weighted(self, infoset, item, weight):
    if self.full():
      return
    self._weights[self._next_index] = weight
    self.add(infoset, item)

  def size(self):
    return self._next_index

  def full(self):
    return self._next_index >= self._infosets.shape[0]

  def size_mb(self):
    total = getsizeof(self._infosets.storage())
    total += getsizeof(self._items.storage())
    if self._has_weights:
      total += getsizeof(self._weights.storage())
    return total / 1e6

  def clear(self):
    self._infosets = torch.zeros((int(self._max_size), self._info_set_size), dtype=torch.float32).to(self._device)
    self._items = torch.zeros((int(self._max_size), self._item_size), dtype=torch.float32).to(self._device)
    self._weights = torch.zeros(int(self._max_size), dtype=torch.float32).to(self._device)
    self._next_index = 0

  def save(self, folder, buffer_name):
    """
    Save the current buffer to a .pth file. The file will contain a dictionary with:
      {
        "infosets": ...,
        "items": ...,
        "weights": ...
      }

    This function will automatically figure out the next .pth file to save in, and add an entry
    to the manifest file in folder.
    """
    manifest_df = get_manifest_entries(folder, buffer_name)
    manifest_file_path = get_buffer_manifest_path(folder, buffer_name)

    # If this is the first time saving, the manifest won't exist, so create it.
    if manifest_df is None:
      os.makedirs(os.path.abspath(folder), exist_ok=True)
      print("Manifest does not exist at {}, creating...".format(manifest_file_path))
      with open(manifest_file_path, "w") as f:
        f.write("index,filename,num_entries\n")
      next_avail_idx = 0
    else:
      next_avail_idx = manifest_df.index[-1] + 1

    buf_path = get_buffer_path(folder, buffer_name, next_avail_idx)

    # Resize to minimum size.
    self._infosets = self._infosets[:self.size(),:].clone()
    self._items = self._items[:self.size(),:].clone()
    self._weights = self._weights[:self.size()].clone()

    torch.save({
      "infosets": self._infosets,
      "items": self._items,
      "weights": self._weights
    }, buf_path)
    print(">> Saved buffer to {}".format(buf_path))

    with open(manifest_file_path, "a") as f:
      f.write("{},{},{}\n".format(next_avail_idx, buf_path, self.size()))
    print(">> Updated manifest file at {}".format(manifest_file_path))


class MemoryBufferDataset(Dataset):
  def __init__(self, folder, buffer_name, n):
    """
    A PyTorch dataset for loading infosets and targets from disk. Because there are too many to fit
    into memory at once, we resample a dataset of size n << N periodically by choosing n random
    items from all N on disk.
    """
    self._folder = folder
    self._buffer_name = buffer_name
    self._n = int(n)

    self._manifest_df = get_manifest_entries(self._folder, self._buffer_name)
    self._N = int(self._manifest_df["num_entries"].sum())

    self._infosets = None
    self._weights = None
    self._items = None
    print(">> Made MemoryBufferDataset to store {}/{} items at a time".format(self._n, self._N))

  def resample(self):
    """
    Resample the dataset by sampling n items from the N total items on disk.
    """
    self._infosets = None
    self._weights = None
    self._items = None

    idx = torch.randint(0, self._N, (self._n,))
    cumul_idx = 0
    for i in self._manifest_df.index:
      num_entries = self._manifest_df["num_entries"][i]
      idx_this_file = idx[idx.ge(cumul_idx) * idx.le(cumul_idx + num_entries - 1)] - cumul_idx
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
    return self._n

  def __getitem__(self, idx):
    return {
      "infoset": self._infosets[idx],
      "weight": self._weights[idx],
      "target": self._items[idx]
    }
