import unittest, shutil, os, random
from constants import Constants
from traverse import create_new_round, make_infoset
from cfr import RegretMatchedStrategy

import torch


class CFRTest(unittest.TestCase):
  def test_save_load(self):
    if os.path.exists("./memory/test_cfr"):
      shutil.rmtree("./memory/test_cfr")

    rm = RegretMatchedStrategy()
    self.assertEqual(rm.size(), 0)

    random.seed(123)
    sb_index = 0
    round_state = create_new_round(sb_index)
    infoset = make_infoset(round_state, 0, True)

    rm.add_regret(infoset, torch.ones(Constants.NUM_ACTIONS))
    self.assertEqual(rm.size(), 1)

    save_path = "./memory/test_cfr/regrets_0.pkl"
    rm.save(save_path)

    rm.load(save_path)


if __name__ == "__main__":
  unittest.main()
