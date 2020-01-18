import unittest, time
import os, shutil
from sys import getsizeof

from memory_buffer import MemoryBuffer
from utils import *
from test_utils import make_dummy_ev_infoset

import torch


class MemoryBufferTest(unittest.TestCase):
  def test_memory_buffer_size(self):
    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    max_size = int(1e6)
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size)
    print(mb._infosets.dtype)
    print(mb._items.dtype)
    print(mb._weights.dtype)
    print("Memory buffer size (max_size={}): {} mb".format(max_size, mb.size_mb()))

  def test_memory_buffer_save(self):
    # Make sure the folder doesn't exist so the manifest has to be created.
    if os.path.exists("./memory/memory_buffer_test/"):
      shutil.rmtree("./memory/memory_buffer_test/")
    info_set_size = 1 + 2 + 5 + 24
    item_size = 64
    max_size = int(1e6)
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size)
    mb.save("./memory/memory_buffer_test/", "test_buffer")

    self.assertTrue(os.path.exists("./memory/memory_buffer_test/manifest_test_buffer.csv"))
    self.assertTrue(os.path.exists("./memory/memory_buffer_test/test_buffer_00000.pth"))

    # Now save again.
    mb.save("./memory/memory_buffer_test/", "test_buffer")
    self.assertTrue(os.path.exists("./memory/memory_buffer_test/test_buffer_00001.pth"))

  def test_memory_buffer_autosave(self):
    print("\n ================= AUTOSAVE TEST ====================")
    # Make sure the folder doesn't exist so the manifest has to be created.
    if os.path.exists("./memory/memory_buffer_test/"):
      shutil.rmtree("./memory/memory_buffer_test/")
    info_set_size = 1 + 1 + 24
    item_size = 64
    max_size = int(1e3)

    # Add autosave params.
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size,
                      autosave_params=("./memory/memory_buffer_test/", "test_buffer"))

    for _ in range(max_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 1234)
    self.assertTrue(mb.full())

    # This should trigger the save and reset.
    mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 1234)


if __name__ == "__main__":
  unittest.main()
