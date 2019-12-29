from trainer import Trainer
from options import Options


if __name__ == "__main__":
  opt = Options().parse()
  trainer = Trainer(opt)

  trainer.load_networks(6)
  trainer.eval_value_network("cfr", 6, None)
