import os, time, math

import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from constants import Constants
from utils import *
from cfr import *
from infoset import EvInfoSet


def traverse_worker(worker_id, traverse_plyr, regret_filenames, strategy_filenames, r_lock, s_lock, opt, t):
  """
  A worker that traverses the game tree K times. Each worker gets a copy of the regret buffers,
  and we merge all the results at the end.
  """
  num_traversals_per_worker = int(opt.NUM_TRAVERSALS_PER_ITER / opt.NUM_TRAVERSE_WORKERS)

  # Load everything in from disk to make sure that every worker has identical initialization.
  t0 = time.time()
  regrets = { 0: RegretMatchedStrategy(), 1: RegretMatchedStrategy() }
  strategies = { 0: RegretMatchedStrategy(), 1: RegretMatchedStrategy() }
  for i in (0, 1):
    regrets[i].load(regret_filenames[i])
    strategies[i].load(strategy_filenames[i])
  elapsed = time.time() - t0
  print("[WORKER #{}] Loaded everything from disk in {} sec".format(worker_id, elapsed))
  
  t0 = time.time()
  for k in range(num_traversals_per_worker):
    ctr = [0]

    # Generate a random initialization, alternating the SB player each time.
    sb_plyr_idx = 1 - (k % 2)
    round_state = create_new_round(sb_plyr_idx)

    precomputed_ev = make_precomputed_ev(round_state)
    info = traverse_cfr(round_state, traverse_plyr, sb_plyr_idx, regrets, strategies,
                        t, torch.ones(2), precomputed_ev, rctr=ctr, allow_updates=True,
                        do_external_sampling=True, skip_unreachable_actions=False)

    if (k % opt.TRAVERSE_DEBUG_PRINT_HZ) == 0:
      elapsed = time.time() - t0
      print("[WORKER #{}] Finished {}/{} traversals | exploit={} | explored={} | R1={} R2={} | S1={} S2={} | elapsed={} sec".format(
            worker_id, k, num_traversals_per_worker, info.exploitability.sum(), ctr[0], regrets[0].size(),
            regrets[1].size(), strategies[0].size(), strategies[1].size(), elapsed))

  # Save all the buffers one last time.
  print("[WORKER #{}] Doing final save".format(worker_id))
  for i in (0, 1):
    regrets[i].merge_and_save(regret_filenames[i], r_lock)
    strategies[i].merge_and_save(strategy_filenames[i], s_lock)
  print('[WORKER #{}] Done!'.format(worker_id))


class Trainer(object):
  def __init__(self, opt):
    self.opt = opt

    self.regrets = {
      0: RegretMatchedStrategy(),
      1: RegretMatchedStrategy()
    }

    self.strategies = {
      0: RegretMatchedStrategy(),
      1: RegretMatchedStrategy()
    }

    self.writers = {}
    self.writers["cfr"] = SummaryWriter(os.path.join(opt.TRAIN_LOG_FOLDER, "cfr"))

    r0_exists = os.path.exists(os.path.dirname(opt.REGRETS_FMT.format(0)))
    r1_exists = os.path.exists(os.path.dirname(opt.REGRETS_FMT.format(1)))
    s0_exists = os.path.exists(os.path.dirname(opt.STRATEGIES_FMT.format(0)))
    s1_exists = os.path.exists(os.path.dirname(opt.STRATEGIES_FMT.format(1)))

    if r0_exists and r1_exists and s0_exists and s1_exists:
      print("\n*** WARNING: Found existing files, resuming from where we left off")
      for i in (0, 1):
        self.regrets[i].load(opt.REGRETS_FMT.format(i))
        self.strategies[i].load(opt.STRATEGIES_FMT.format(i))
    else:
      # Save to create these initial files.
      print("\n*** NOTE: Doing initial save so that stuff exists")
      for i in (0, 1):
        self.regrets[i].save(opt.REGRETS_FMT.format(i))
        self.strategies[i].save(opt.STRATEGIES_FMT.format(i))

  def main(self):
    cfr_steps = 0
    for t in range(self.opt.NUM_CFR_ITERS):
      for traverse_plyr in (0, 1):
        # Accumulate regrets/strategy profiles for the traverse player. At the end, all of this data
        # is saved to disk and then lost from memory, so we should reload it.
        self.accumulate_regret(traverse_plyr, t)
        self.load()
        # self.load(regrets_to_load=[traverse_plyr], strategies_to_load=[traverse_plyr])
        self.evaluate(cfr_steps)
        cfr_steps += 1

  def load(self, regrets_to_load=[0, 1], strategies_to_load=[0, 1]):
    """
    Load regrets and average strategy if files already exist.
    """
    for plyr in (0, 1):
      if plyr in regrets_to_load:
        filename = self.opt.REGRETS_FMT.format(plyr)
        print("NOTE: Reloading regrets for player {} from {}".format(plyr, filename))
        self.regrets[plyr].load(filename)
      if plyr in strategies_to_load:
        filename = self.opt.STRATEGIES_FMT.format(plyr)
        print('NOTE: Reloading average strategy for player {} from {}'.format(plyr, filename))
        self.strategies[plyr].load(filename)
  
  def accumulate_regret(self, traverse_plyr, t):
    print("\nAccumulating regrets for player {} (t={})".format(traverse_plyr, t))

    manager = mp.Manager()
    r_lock = manager.Lock()
    s_lock = manager.Lock()

    regret_filenames = [self.opt.REGRETS_FMT.format(plyr) for plyr in (0, 1)]
    strategy_filenames = [self.opt.STRATEGIES_FMT.format(plyr) for plyr in (0, 1)]

    print(regret_filenames)
    print(strategy_filenames)

    t0 = time.time()
    mp.spawn(
      traverse_worker,
      args=(traverse_plyr, regret_filenames, strategy_filenames, r_lock, s_lock, self.opt, t),
      nprocs=self.opt.NUM_TRAVERSE_WORKERS, join=True, daemon=False)
    elapsed = time.time() - t0
    print("Time for {} traversals across {} workers: {} sec".format(
      self.opt.NUM_TRAVERSALS_PER_ITER, self.opt.NUM_TRAVERSE_WORKERS, elapsed))

  def evaluate(self, step):
    print("\nEvaluating average strategy after {} steps".format(step))

    t0 = time.time()
    exploits = []

    for k in range(self.opt.NUM_TRAVERSALS_EVAL):
      sb_plyr_idx = k % 2
      round_state = create_new_round(sb_plyr_idx)
      precomputed_ev = make_precomputed_ev(round_state)

      # NOTE: disable updates to memories.
      ctr = [0]
      info = traverse_cfr(round_state, 0, sb_plyr_idx, self.strategies, self.strategies,
                          1234, torch.ones(2), precomputed_ev, rctr=ctr, allow_updates=False,
                          do_external_sampling=False, skip_unreachable_actions=True)
      exploits.append(info.exploitability.sum())

      if (k % self.opt.TRAVERSE_DEBUG_PRINT_HZ) == 0:
        print("Finished {}/{} eval traversals | exploit={} | explored={}".format(
            k, self.opt.NUM_TRAVERSALS_EVAL, info.exploitability.sum(), ctr[0]))


    elapsed = time.time() - t0
    print("Time for {} eval traversals {} sec".format(self.opt.NUM_TRAVERSALS_EVAL, elapsed))

    mbb_per_game = 1e3 * torch.Tensor(exploits) / (2.0 * Constants.SMALL_BLIND_AMOUNT)
    mean_mbb_per_game = mbb_per_game.mean().item()
    stdev_mbb_per_game = mbb_per_game.std().item()
    sterr_mbb_per_game = stdev_mbb_per_game / math.sqrt(self.opt.NUM_TRAVERSALS_EVAL)

    print("===> [EVAL] [AVG STRATEGY] Exploitability | mean={} mbb/g | stdev={} | stderr={} | (step={})".format(
        mean_mbb_per_game, stdev_mbb_per_game, sterr_mbb_per_game, step))

    # Looks like everything has to be logged in one function...
    writer = self.writers["cfr"]
    writer.add_scalar("exploit/mbbg_mean", mean_mbb_per_game, int(step))
    writer.add_scalar("exploit/mbbg_stderr", sterr_mbb_per_game, int(step))
    writer.add_scalar("exploit/mbbg_stdev", stdev_mbb_per_game, int(step))

    # Log the sizes of each memory.
    for i in (0, 1):
      writer.add_scalar("num_infosets/regrets/{}".format(i), self.regrets[i].size(), step)
      writer.add_scalar("num_infosets/strategies/{}".format(i), self.strategies[i].size(), step)
    writer.close()
