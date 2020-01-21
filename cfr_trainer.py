import ray
from copy import deepcopy
import os, time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from constants import Constants
from utils import *
from cfr import traverse_cfr, RegretMatchedStrategy
from traverse import make_actions, make_infoset, create_new_round, make_precomputed_ev
from infoset import EvInfoSet


def traverse_worker(worker_id, traverse_plyr_idx, regrets, avg_strategy, r_lock, avg_lock, opt, t):
  """
  A worker that traverses the game tree K times. Each worker gets a copy of the regret buffers,
  and we merge all the results at the end.
  """
  num_traversals_per_worker = int(opt.NUM_TRAVERSALS_PER_ITER / opt.NUM_TRAVERSE_WORKERS)
  
  t0 = time.time()
  for k in range(num_traversals_per_worker):
    ctr = [0]

    # Generate a random initialization, alternating the SB player each time.
    sb_plyr_idx = k % 2
    round_state = create_new_round(sb_plyr_idx)

    precomputed_ev = make_precomputed_ev(round_state)
    info = traverse_cfr(round_state, traverse_plyr_idx, sb_plyr_idx, regrets, avg_strategy,
                        t, precomputed_ev, rctr=ctr, allow_updates=True)

    if (k % opt.TRAVERSE_DEBUG_PRINT_HZ) == 0:
      elapsed = time.time() - t0
      print("[WORKER #{}] done with {}/{} traversals | recursion depth={} | regrets={}/{} | avg_stategy={} | elapsed={} sec".format(
            worker_id, k, num_traversals_per_worker, ctr[0], regrets[0].size(), regrets[1].size(), avg_strategy.size(), elapsed))

  # Save all the buffers one last time.
  print("[WORKER #{}] Final autosave ...".format(worker_id))
  regrets_filename = opt.REGRETS_FMT.format(traverse_plyr_idx)
  regrets[traverse_plyr_idx].merge_and_save(regrets_filename, r_lock)

  avg_strategy_filename = opt.AVG_STRT_FMT
  avg_strategy.merge_and_save(avg_strategy_filename, avg_lock)
  print('[WORKER #{}] Done!'.format(worker_id))


class Trainer(object):
  def __init__(self, opt):
    self.opt = opt

    self.regrets = {
      0: RegretMatchedStrategy(),
      1: RegretMatchedStrategy()
    }
    self.avg_strategy = RegretMatchedStrategy()

    self.writers = {}
    self.writers["cfr"] = SummaryWriter(os.path.join(opt.TRAIN_LOG_FOLDER, "cfr"))

  def main(self):
    eval_t = 0
    for t in range(self.opt.NUM_CFR_ITERS):
      for traverse_plyr_idx in (0, 1):
        self.accumulate_regret(traverse_plyr_idx, t)

        # NOTE: Since we have several workers adding things to disk, need to reload their merged results.
        self.load(regrets_to_load=[traverse_plyr_idx], load_avg_strt=True)

        self.evaluate(eval_t)
        self.log_sizes(eval_t)
        eval_t += 1

  def load(self, regrets_to_load=[0, 1], load_avg_strt=True):
    """
    Load regrets and average strategy if files already exist.
    """
    for plyr in (0, 1):
      if plyr in regrets_to_load:
        print("NOTE: Reloading regrets for player {}".format(plyr))
        filename = self.opt.REGRETS_FMT.format(plyr)
        self.regrets[plyr].load(filename)
    
    if load_avg_strt:
      print('NOTE: Reloading average strategy')
      avg_filename = self.opt.AVG_STRT_FMT
      self.avg_strategy.load(avg_filename)
  
  def accumulate_regret(self, traverse_plyr_idx, t):
    print("\nAccumulating regrets (t={}) for player {}".format(t, traverse_plyr_idx))

    manager = mp.Manager()
    r_lock = manager.Lock()
    avg_lock = manager.Lock()

    t0 = time.time()

    mp.spawn(
      traverse_worker,
      args=(traverse_plyr_idx, self.regrets, self.avg_strategy, r_lock, avg_lock, self.opt, t),
      nprocs=self.opt.NUM_TRAVERSE_WORKERS, join=True, daemon=False)

    elapsed = time.time() - t0
    print("Time for {} traversals across {} workers: {} sec".format(
      self.opt.NUM_TRAVERSALS_PER_ITER, self.opt.NUM_TRAVERSE_WORKERS, elapsed))

  def evaluate(self, steps):
    print("\nEvaluating average strategy after {} steps".format(steps))

    t0 = time.time()
    exploits = []

    strategies = {
      0: self.avg_strategy,
      1: self.avg_strategy
    }

    for k in range(self.opt.NUM_TRAVERSALS_EVAL):
      sb_plyr_idx = k % 2
      round_state = create_new_round(sb_plyr_idx)
      precomputed_ev = make_precomputed_ev(round_state)

      # NOTE: disable updates to memories.
      info = traverse_cfr(round_state, 0, sb_plyr_idx, strategies, self.avg_strategy,
                          1234, precomputed_ev, rctr=[0], allow_updates=False)
      exploits.append(info.exploitability.sum())

    elapsed = time.time() - t0
    print("Time for {} eval traversals {} sec".format(self.opt.NUM_TRAVERSALS_EVAL, elapsed))

    mbb_per_game = 1e3 * torch.Tensor(exploits) / (2.0 * Constants.SMALL_BLIND_AMOUNT)
    mean_mbb_per_game = mbb_per_game.mean()
    stdev_mbb_per_game = mbb_per_game.std()

    writer = self.writers["cfr"]
    writer.add_scalar("avg_exploit_mbbg_mean", mean_mbb_per_game, steps)
    writer.add_scalar("avg_exploit_mbbg_stdev", stdev_mbb_per_game, steps)
    writer.close()
    print("===> [EVAL] [AVG STRATEGY] Exploitability | mean={} mbb/g | stdev={} | (steps={})".format(
        mean_mbb_per_game, stdev_mbb_per_game, steps))

  def log_sizes(self, step):
    writer = self.writers["cfr"]
    writer.add_scalar("num_infosets/regrets/0", self.regrets[0].size(), step)
    writer.add_scalar("num_infosets/regrets/1", self.regrets[1].size(), step)
    writer.add_scalar("num_infosets/avg_strategy", self.avg_strategy.size(), step)
    writer.close()
