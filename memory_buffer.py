from sys import getsizeof
import os

import torch
import pandas as pd

from constants import Constants
# from infoset import InfoSet
from infoset import EvInfoSet


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
  def __init__(self, info_set_size, item_size, max_size=80000, device=torch.device("cpu"),
               autosave_params=None, save_lock=None):
    self._max_size = max_size
    self._info_set_size = info_set_size
    self._item_size = item_size
    self._autosave_params = autosave_params
    if self._autosave_params is not None:
      print(">> NOTE: Autosaving is turned ON for buffer")
      print("  >> folder={}".format(self._autosave_params[0]))
      print("  >> name={}".format(self._autosave_params[1]))

    self._device = device

    self._infosets = torch.zeros((int(max_size), info_set_size), dtype=torch.float32).to(self._device)
    self._items = torch.zeros((int(max_size), item_size), dtype=torch.float32).to(self._device)
    self._weights = torch.zeros(int(max_size), dtype=torch.float32).to(self._device)

    self._next_index = 0

    self._save_lock = save_lock

  def add(self, infoset, item, weight):
    """
    Add an infoset and corresponding item to the buffer.

    If the buffer is full, then it will either:
      (1) Save its contents and then clear before adding the item (autosave=(folder, name)).
      (2) Ignore the add and do nothing (autosave=None).
    """
    if self.full():
      if self._autosave_params is not None:
        self.save(self._autosave_params[0], self._autosave_params[1])
        self.clear()
      else:
        return
    self._infosets[self._next_index] = infoset.pack()
    self._items[self._next_index] = item
    self._weights[self._next_index] = weight
    self._next_index += 1

  def size(self):
    return self._next_index

  def full(self):
    return self._next_index >= self._infosets.shape[0]

  def size_mb(self):
    total = getsizeof(self._infosets.storage())
    total += getsizeof(self._items.storage())
    total += getsizeof(self._weights.storage())
    return total / 1e6

  def clear(self):
    self._infosets = torch.zeros((int(self._max_size), self._info_set_size), dtype=torch.float32).to(self._device)
    self._items = torch.zeros((int(self._max_size), self._item_size), dtype=torch.float32).to(self._device)
    self._weights = torch.zeros(int(self._max_size), dtype=torch.float32).to(self._device)
    self._next_index = 0

  def autosave(self):
    if self._autosave_params is not None:
      self.save(self._autosave_params[0], self._autosave_params[1])
      self.clear()
      return True
    else:
      return False

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
    if self._save_lock is not None: self._save_lock.acquire()

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
      next_avail_idx = 0 if len(manifest_df.index) == 0 else (manifest_df.index[-1] + 1)

    buf_path = get_buffer_path(folder, buffer_name, next_avail_idx)

    with open(manifest_file_path, "a") as f:
      f.write("{},{},{}\n".format(next_avail_idx, buf_path, self.size()))
    print(">> Updated manifest file at {}".format(manifest_file_path))

    if self._save_lock is not None: self._save_lock.release()

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
