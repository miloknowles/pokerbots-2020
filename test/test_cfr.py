import unittest, shutil, os, random
from constants import Constants
from traverse import create_new_round, make_infoset, make_precomputed_ev
from engine import CallAction
from cfr import RegretMatchedStrategy, traverse_cfr

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

    sigma = rm.get_strategy(infoset, torch.ones(Constants.NUM_ACTIONS))
    self.assertEqual(sigma.sum(), 1.0)
    print("Strategy (all allowed):", sigma)

    sigma = rm.get_strategy(infoset, torch.Tensor([1.0, 0, 1.0, 0, 0, 0]))
    self.assertEqual(sigma.sum(), 1.0)
    print("Strategy (0 and 2 allowed):", sigma)

    round_state = round_state.proceed(CallAction())
    infoset = make_infoset(round_state, 1, False)
    rm.add_regret(infoset, torch.Tensor([-5, -5, -1, -5, -5, -6]))
    self.assertEqual(rm.size(), 2)

    sigma = rm.get_strategy(infoset, torch.ones(Constants.NUM_ACTIONS))
    print("Strategy (2 is best):", sigma)

    save_path = "./memory/test_cfr/regrets_0.pkl"
    rm.save(save_path)
    rm.load(save_path)
    self.assertEqual(rm.size(), 2)

  def test_traverse_cfr(self):
    regrets = {
      0: RegretMatchedStrategy(),
      1: RegretMatchedStrategy()
    }

    avg_strategy = RegretMatchedStrategy()

    for k in range(1000):
      sb_index = k % 2
      round_state = create_new_round(sb_index)
      precomputed_ev = make_precomputed_ev(round_state)
      ctr = [0]
      traverse_cfr(round_state, 0, sb_index, regrets, avg_strategy, 0, precomputed_ev, rctr=ctr)

      if k % 50 == 0:
        print("Finished {} traversals".format(k))
        print("REGRETS P1:", regrets[0].size())
        print("REGRETS P2:", regrets[1].size())
        print("AVG STRATEGY:", avg_strategy.size())


if __name__ == "__main__":
  unittest.main()
