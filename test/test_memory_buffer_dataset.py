import torch

import unittest, os, shutil
from memory_buffer import MemoryBuffer
from memory_buffer_dataset import MemoryBufferDataset
from infoset import EvInfoSet

def make_dummy_ev_infoset():
  ev = 0.43
  bet_history_vec = torch.ones(16)
  bet_history_vec[3:7] = 0
  infoset = EvInfoSet(ev, bet_history_vec, 1)
  return infoset


class MemoryBufferDatasetTest(unittest.TestCase):
  def test_resample(self):
    if os.path.exists("./memory/memory_buffer_test/"):
      shutil.rmtree("./memory/memory_buffer_test/")

    # Make a few saved memory buffers.
    info_set_size = 1 + 1 + 16
    item_size = 6
    max_size = int(1e4)
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size)

    buf1_size = 100
    for i in range(buf1_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 0)
    mb.save("./memory/memory_buffer_test/", "advt_mem_0")
    mb.clear()

    buf2_size = 200
    for i in range(buf2_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 1)
    mb.save("./memory/memory_buffer_test/", "advt_mem_0")
    mb.clear()

    buf3_size = 300
    for i in range(buf3_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 2)
    mb.save("./memory/memory_buffer_test/", "advt_mem_0")
    mb.clear()

    # Make a dataset using the saved buffers.
    # n = (buf1_size + buf2_size) // 10
    n = 1000
    dataset = MemoryBufferDataset("./memory/memory_buffer_test/", "advt_mem_0", n)
    # min_size = min(n, buf1_size + buf2_size + buf3_size)
    # print(min_size)

    for _ in range(1):
      dataset.resample()
      self.assertEqual(len(dataset), n)
      self.assertEqual(len(dataset._infosets), n)
      self.assertEqual(len(dataset._items), n)
      self.assertEqual(len(dataset._weights), n)
      # print(dataset._weights)

    # Test iteration over the dataset.
    for inputs in dataset:
      print(inputs.keys())

    print(dataset._weights)


if __name__ == "__main__":
  unittest.main()
