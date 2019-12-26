import time

from trainer import Trainer
from options import Options


if __name__ == "__main__":
  opt = Options()
  trainer = Trainer(opt.parse())
  trainer.main()
