import torch

import unittest, os, shutil
from memory_buffer import MemoryBuffer
from memory_buffer_dataset import MemoryBufferDataset
from test_utils import make_dummy_ev_infoset


class MemoryBufferDatasetTest(unittest.TestCase):
  def test_resample(self):
    if os.path.exists("./memory/memory_buffer_test/"):
      shutil.rmtree("./memory/memory_buffer_test/")

    # Make a few saved memory buffers.
    info_set_size = 1 + 1 + 24
    item_size = 64
    max_size = int(1e4)
    mb = MemoryBuffer(info_set_size, item_size, max_size=max_size)

    buf1_size = 100
    for i in range(buf1_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 1234)
    mb.save("./memory/memory_buffer_test/", "advt_buf_P1")
    mb.clear()

    buf2_size = 200
    for i in range(buf2_size):
      mb.add(make_dummy_ev_infoset(), torch.zeros(item_size), 1234)
    mb.save("./memory/memory_buffer_test/", "advt_buf_P1")
    mb.clear()

    # Make a dataset using the saved buffers.
    n = (buf1_size + buf2_size) // 10
    dataset = MemoryBufferDataset("./memory/memory_buffer_test/", "advt_buf_P1", n)

    for _ in range(10):
      dataset.resample()
      self.assertEqual(len(dataset), n)
      self.assertEqual(len(dataset._infosets), n)
      self.assertEqual(len(dataset._items), n)
      self.assertEqual(len(dataset._weights), n)

    # Test iteration over the dataset.
    for inputs in dataset:
      print(inputs.keys())


if __name__ == "__main__":
  unittest.main()
