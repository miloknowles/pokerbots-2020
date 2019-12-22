import unittest, time

from trainer import Trainer
from options import Options


class TrainerTest(unittest.TestCase):
  def test_do_cfr_iteration_for_player(self):
    opt = Options()
    opt.NUM_TRAVERSALS_PER_ITER = 200
    
    trainer = Trainer(opt)

    t0 = time.time()
    trainer.do_cfr_iteration_for_player("P1", 0)
    elapsed = time.time() - t0
    print("Did {} traversals in {} sec (avg {} trav/sec)".format(
      opt.NUM_TRAVERSALS_PER_ITER, elapsed, opt.NUM_TRAVERSALS_PER_ITER / elapsed))


if __name__ == "__main__":
  unittest.main()
