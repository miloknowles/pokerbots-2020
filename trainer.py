import ray
from copy import deepcopy
import os, time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse, make_actions, make_infoset, create_new_round, make_precomputed_ev
from memory_buffer import MemoryBuffer
from infoset import EvInfoSet
from memory_buffer_dataset import MemoryBufferDataset
from network_wrapper import NetworkWrapper


def traverse_worker(worker_id, traverse_player_idx, strategies, save_lock, opt, t, eval_mode,
                    info_queue):
  """
  A worker that traverses the game tree K times, saving things to memory buffers. Each worker
  maintains its own memory buffers and saves them after finishing.

  If eval_mode is set to True, no memory buffers are created.
  """
  # assert(strategies[0]._network.device == torch.device("cpu"))
  # assert(strategies[1]._network.device == torch.device("cpu"))

  advt_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                          max_size=opt.SINGLE_PROC_MEM_BUFFER_MAX_SIZE,
                          autosave_params=(opt.MEMORY_FOLDER, opt.ADVT_BUFFER_FMT.format(traverse_player_idx)),
                          save_lock=save_lock) if eval_mode == False else None
  
  strt_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                          max_size=opt.SINGLE_PROC_MEM_BUFFER_MAX_SIZE,
                          autosave_params=(opt.MEMORY_FOLDER, opt.STRT_BUFFER_FMT),
                          save_lock=save_lock) if eval_mode == False else None

  if eval_mode:
    num_traversals_per_worker = int(opt.NUM_TRAVERSALS_EVAL / opt.NUM_TRAVERSE_WORKERS)
  else:
    num_traversals_per_worker = int(opt.NUM_TRAVERSALS_PER_ITER / opt.NUM_TRAVERSE_WORKERS)
  
  t0 = time.time()
  for k in range(num_traversals_per_worker):
    ctr = [0]

    # Generate a random initialization, alternating the SB player each time.
    sb_player_idx = k % 2
    round_state = create_new_round(sb_player_idx)

    precomputed_ev = make_precomputed_ev(round_state)
    info = traverse(round_state, make_actions, make_infoset, traverse_player_idx, sb_player_idx,
                    strategies, advt_mem, strt_mem, t, precomputed_ev, recursion_ctr=ctr)

    if (k % opt.TRAVERSE_DEBUG_PRINT_HZ) == 0 and eval_mode == False:
      elapsed = time.time() - t0
      print("[WORKER #{}] done with {}/{} traversals | recursion depth={} | advt={} strt={} | elapsed={} sec".format(
            worker_id, k, num_traversals_per_worker, ctr[0], advt_mem.size(), strt_mem.size(), elapsed))

  # Save all the buffers one last time.
  print("[WORKER #{}] Final autosave ...".format(worker_id))
  if advt_mem is not None: advt_mem.autosave()
  if strt_mem is not None: strt_mem.autosave()


class Trainer(object):
  def __init__(self, opt):
    self.opt = opt

    self.value_networks = {
      0: NetworkWrapper(Constants.BET_HISTORY_SIZE,
                        Constants.NUM_ACTIONS, ev_embed_dim=opt.EV_EMBED_DIM,
                        bet_embed_dim=opt.BET_EMBED_DIM, device=opt.TRAVERSE_DEVICE),
      1: NetworkWrapper(Constants.BET_HISTORY_SIZE,
                        Constants.NUM_ACTIONS, ev_embed_dim=opt.EV_EMBED_DIM,
                        bet_embed_dim=opt.BET_EMBED_DIM, device=opt.TRAVERSE_DEVICE)
    }
    # self.value_networks[0]._network.share_memory()
    # self.value_networks[1]._network.share_memory()
    print("[DONE] Made value networks")

    self.strategy_network = NetworkWrapper(Constants.BET_HISTORY_SIZE,
                        Constants.NUM_ACTIONS, ev_embed_dim=opt.EV_EMBED_DIM,
                        bet_embed_dim=opt.BET_EMBED_DIM, device=opt.TRAIN_DEVICE)
    print("[DONE] Made strategy network")

    self.writers = {}
    for mode in ["train", "cfr"]:
      self.writers[mode] = SummaryWriter(os.path.join(opt.TRAIN_LOG_FOLDER, mode))

  def main(self):
    eval_t = 0
    for t in range(self.opt.NUM_CFR_ITERS):
      for traverse_player_idx in (0, 1):
        print("Weights before TRAVERSE")
        print(self.value_networks[traverse_player_idx].network().ev1.bias)
        print(self.value_networks[1 - traverse_player_idx].network().ev1.bias)
        self.do_cfr_iter_for_player(traverse_player_idx, t)
        self.train_value_network(traverse_player_idx, t)
        print("Weights after TRAINING:")
        print(self.value_networks[traverse_player_idx].network().ev1.bias)
        print(self.value_networks[1 - traverse_player_idx].network().ev1.bias)
        self.eval_value_network("cfr", eval_t, None, traverse_player_idx)
        print("Weights after EVAL:")
        print(self.value_networks[traverse_player_idx].network().ev1.bias)
        print(self.value_networks[1 - traverse_player_idx].network().ev1.bias)
        eval_t += 1
    # TODO(milo): Train strategy network.
  
  def do_cfr_iter_for_player(self, traverse_player_idx, t):
    print("\nDoing CFR iteration t={} for player {}".format(t, traverse_player_idx))
    self.value_networks[0]._network = self.value_networks[0]._network.to(self.opt.TRAVERSE_DEVICE)
    self.value_networks[1]._network = self.value_networks[1]._network.to(self.opt.TRAVERSE_DEVICE)
    self.value_networks[0]._device = self.opt.TRAVERSE_DEVICE
    self.value_networks[1]._device = self.opt.TRAVERSE_DEVICE
    print()

    manager = mp.Manager()
    save_lock = manager.Lock()

    t0 = time.time()

    mp.spawn(
      traverse_worker,
      args=(traverse_player_idx, self.value_networks, save_lock, self.opt, t, False, None),
      nprocs=self.opt.NUM_TRAVERSE_WORKERS, join=True, daemon=False)

    elapsed = time.time() - t0
    print("Time for {} traversals across {} workers: {} sec".format(
      self.opt.NUM_TRAVERSALS_PER_ITER, self.opt.NUM_TRAVERSE_WORKERS, elapsed))

  def linear_cfr_loss(self, output, target, weights):
    """
    Mean-squared error loss, where each example is weighted by its CFR iteration t. We divide the
    loss by the mean weight so that the batch has an average weight of 1.

    output (torch.Tensor) : Shape (batch_size, num_actions).
    target (torch.Tensor) : Shape (batch_size, num_actions).
    weights (torch.Tensor) : Shape (batch_size).
    T (int) : The current CFR iteration that we're training for (used in normalization).
    """
    # NOTE(milo): Need to add 1 to the weights to deal with zeroth CFR iteration.
    weights_safe = (weights + 1.0)
    weighted_se = weights_safe * (target - output).pow(2) / weights_safe.mean()
    return weighted_se.mean()

  def train_strategy_network(self):
    """
    Train the strategy network from scratch using all strategy buffer entries.
    """
    print("\nTraining strategy network!")

    losses = {}

    self.strategy_network._network = self.strategy_network._network.cuda()
    self.strategy_network._device = torch.device("cuda")

    net = self.strategy_network.network()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=self.opt.SGD_LR)

    buffer_name = self.opt.STRT_BUFFER_FMT
    train_dataset = MemoryBufferDataset(self.opt.MEMORY_FOLDER, buffer_name,
                                        self.opt.TRAIN_DATASET_SIZE)

    # Due to memory limitations, we can only store a subset of the dataset in memory at a time.
    # Calculate the number of times we'll have to resample and iterate over the dataset.
    # num_resample_iters = int(float(self.opt.SGD_ITERS * self.opt.SGD_BATCH_SIZE) / len(train_dataset)) + 1
    total_items = self.opt.SGD_ITERS * self.opt.SGD_BATCH_SIZE
    num_resample_iters = int(max(1.0, float(total_items) / self.opt.TRAIN_DATASET_SIZE))
    print("Will do {} resample iters (need {} total items, dataset size is {})".format(
        num_resample_iters, total_items, len(train_dataset)))

    step = 0
    for resample_iter in range(num_resample_iters):
      print("> Doing resample iteration {}/{} ...".format(resample_iter, num_resample_iters))
      train_dataset.resample()
      train_loader = DataLoader(train_dataset,
                                batch_size=self.opt.SGD_BATCH_SIZE,
                                num_workers=self.opt.NUM_DATA_WORKERS,
                                shuffle=True)
      print("> Done. DataLoader has {} batches of size {}.".format(len(train_loader), self.opt.SGD_BATCH_SIZE))

      for batch_idx, input_dict in enumerate(train_loader):
        if (batch_idx > self.opt.SGD_ITERS):
          print("Finished batch {}, didn't need to use all batches in DataLoader".format(batch_idx))
          break
        ev_input = input_dict["ev_input"].to(self.opt.TRAIN_DEVICE)
        bets_input = input_dict["bets_input"].to(self.opt.TRAIN_DEVICE)
        sigma_target = input_dict["target"].to(self.opt.TRAIN_DEVICE)
        weights = input_dict["weights"].to(self.opt.TRAIN_DEVICE)

        optimizer.zero_grad()

        # Minimize MSE between predicted advantage and instantaneous regret samples.
        sigma_pred = net(ev_input, bets_input)
        loss = self.linear_cfr_loss(sigma_pred, sigma_target, weights)
        loss.backward()
        losses["strt_mse_loss"] = loss.cpu().item()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if (step % self.opt.TRAINING_LOG_HZ) == 0:
          self.log("train", 0, 1234, losses, step)

        if (step % self.opt.TRAINING_SAVE_HZ) == 0 and step > 0:
          self.save_models(1234, save_value_networks=[], save_strategy_network=True)

        if (step % self.opt.TRAINING_EVAL_HZ) == 0:
          self.eval_strategy_network(step)
          self.strategy_network._network = self.strategy_network._network.cuda()
          self.strategy_network._device = torch.device("cuda")

        step += 1
    
    print("Saving final strategy model")
    self.save_models(1234, save_value_networks=[], save_strategy_network=True)

    print("Evaluating final strategy model")
    self.eval_strategy_network(step)

  def train_value_network(self, traverse_player_idx, t):
    """
    Train a value network from scratch using samples from the traverse player's buffer.
    """
    print("\nTraining value network for {} from scratch (t={})".format(traverse_player_idx, t))
    losses = {}

    # NOTE: This causes the network to be reset.
    self.value_networks[traverse_player_idx] = NetworkWrapper(
        Constants.BET_HISTORY_SIZE,
        Constants.NUM_ACTIONS, ev_embed_dim=self.opt.EV_EMBED_DIM,
        bet_embed_dim=self.opt.BET_EMBED_DIM, device=self.opt.TRAIN_DEVICE)
    net = self.value_networks[traverse_player_idx].network()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=self.opt.SGD_LR)

    buffer_name = self.opt.ADVT_BUFFER_FMT.format(traverse_player_idx)
    train_dataset = MemoryBufferDataset(self.opt.MEMORY_FOLDER, buffer_name,
                                        self.opt.TRAIN_DATASET_SIZE)

    # Due to memory limitations, we can only store a subset of the dataset in memory at a time.
    # Calculate the number of times we'll have to resample and iterate over the dataset.
    # num_resample_iters = int(float(self.opt.SGD_ITERS * self.opt.SGD_BATCH_SIZE) / len(train_dataset)) + 1
    total_items = self.opt.SGD_ITERS * self.opt.SGD_BATCH_SIZE
    num_resample_iters = int(max(1.0, float(total_items) / self.opt.TRAIN_DATASET_SIZE))
    print("Will do {} resample iters (need {} total items, dataset size is {})".format(num_resample_iters, total_items, len(train_dataset)))

    step = 0
    for resample_iter in range(num_resample_iters):
      print("> Doing resample iteration {}/{} ...".format(resample_iter, num_resample_iters))
      train_dataset.resample()
      train_loader = DataLoader(train_dataset,
                                batch_size=self.opt.SGD_BATCH_SIZE,
                                num_workers=self.opt.NUM_DATA_WORKERS,
                                shuffle=True)
      print("> Done. DataLoader has {} batches of size {}.".format(len(train_loader), self.opt.SGD_BATCH_SIZE))

      for batch_idx, input_dict in enumerate(train_loader):
        if (batch_idx > self.opt.SGD_ITERS):
          print("Finished batch {}, didn't need to all batches in DataLoader".format(batch_idx))
          break
        ev_input = input_dict["ev_input"].to(self.opt.TRAIN_DEVICE)
        bets_input = input_dict["bets_input"].to(self.opt.TRAIN_DEVICE)
        advt_target = input_dict["target"].to(self.opt.TRAIN_DEVICE)

        weights = input_dict["weights"].to(self.opt.TRAIN_DEVICE)
        optimizer.zero_grad()

        # Minimize MSE between predicted advantage and instantaneous regret samples.
        output = net(ev_input, bets_input)
        loss = self.linear_cfr_loss(output, advt_target, weights)
        # loss = torch.nn.functional.mse_loss(output, advt_target)
        loss.backward()
        losses["mse_loss/{}/{}".format(t, traverse_player_idx)] = loss.cpu().item()

        # NOTE: Clip gradient norm.
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        optimizer.step()

        if (step % self.opt.TRAINING_LOG_HZ) == 0:
          self.log("train", traverse_player_idx, t, losses, step)

        # Only need to save the value network for the traversing player.
        if (step % self.opt.TRAINING_SAVE_HZ) == 0 and step > 0:
          self.save_models(t, save_value_networks=[traverse_player_idx], save_strategy_network=False)

        # if (batch_idx % self.opt.TRAINING_EVAL_HZ) == 0 and batch_idx > 0:
        #   self.eval_value_network("train", t, step)
        
        step += 1
    
    print("Saving final model for traverse player {}".format(traverse_player_idx))
    self.save_models(t, save_value_networks=[traverse_player_idx], save_strategy_network=False)

  def save_models(self, t, save_value_networks=[], save_strategy_network=False):
    """
    Save model weights to disk.

    t (int) : The CFR iteration that these models are being trained on.
    save_value_networks (list of str) : Zero or more of [0, 1].
    save_strategy_network (bool) : Whether or not to save the current strategy network.
    """
    save_folder = os.path.join(self.opt.TRAIN_LOG_FOLDER, "models", "weights_{}".format(t))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder, exist_ok=True)

    # Optionally save the value networks.
    for player_name in save_value_networks:
      save_path = os.path.join(save_folder, "value_network_{}.pth".format(player_name))
      to_save = self.value_networks[player_name].network().state_dict()
      torch.save(to_save, save_path)

    # Save the strategy network also.
    if save_strategy_network:
      save_path = os.path.join(save_folder, "strategy_network_{}.pth".format(t))
      to_save = self.strategy_network.network().state_dict()
      torch.save(to_save, save_path)
    
    print("Saved models to {}".format(save_folder))

  def eval_strategy_network(self, steps):
    print("\nEvaluating strategy network after {} steps".format(steps))
    self.strategy_network._network = self.strategy_network._network.cpu()
    self.strategy_network._device = torch.device("cpu")

    for p in self.strategy_network._network.parameters():
      assert(p.device == torch.device("cpu"))

    manager = mp.Manager()
    save_lock = manager.Lock()

    t0 = time.time()
    exploits = []

    strategies = {
      0: self.strategy_network,
      1: self.strategy_network
    }

    for k in range(self.opt.NUM_TRAVERSALS_EVAL):
      sb_player_idx = k % 2
      round_state = create_new_round(sb_player_idx)
      precomputed_ev = make_precomputed_ev(round_state)
      info = traverse(round_state, make_actions, make_infoset, 0, sb_player_idx,
                      strategies, None, None, 0, precomputed_ev)
      exploits.append(info.exploitability.sum())

    elapsed = time.time() - t0
    print("Time for {} eval traversals {} sec".format(self.opt.NUM_TRAVERSALS_EVAL, elapsed))

    mbb_per_game = 1e3 * torch.Tensor(exploits) / (2.0 * Constants.SMALL_BLIND_AMOUNT)
    mean_mbb_per_game = mbb_per_game.mean()
    stdev_mbb_per_game = mbb_per_game.std()

    writer = self.writers["train"]
    writer.add_scalar("strt_exploit_mbbg_mean", mean_mbb_per_game, steps)
    writer.add_scalar("strt_exploit_mbbg_stdev", stdev_mbb_per_game, steps)
    writer.close()
    print("===> [EVAL] [STRATEGY] Exploitability | mean={} mbb/g | stdev={} | (steps={})".format(
        mean_mbb_per_game, stdev_mbb_per_game, steps))

  def eval_value_network(self, mode, t, steps, traverse_player_idx):
    """
    Evaluate the (total) exploitability of the value networks, as in Brown et. al.
    """
    print("\nEvaluating value network for player {} (t={})".format(traverse_player_idx, t))
    self.value_networks[0]._network = self.value_networks[0]._network.to(self.opt.TRAVERSE_DEVICE)
    self.value_networks[1]._network = self.value_networks[1]._network.to(self.opt.TRAVERSE_DEVICE)
    self.value_networks[0]._device = self.opt.TRAVERSE_DEVICE
    self.value_networks[1]._device = self.opt.TRAVERSE_DEVICE

    manager = mp.Manager()
    save_lock = manager.Lock()

    t0 = time.time()
    exploits = []

    for k in range(self.opt.NUM_TRAVERSALS_EVAL):
      sb_player_idx = k % 2
      round_state = create_new_round(sb_player_idx)
      precomputed_ev = make_precomputed_ev(round_state)
      info = traverse(round_state, make_actions, make_infoset, traverse_player_idx, sb_player_idx,
                      self.value_networks, None, None, t, precomputed_ev)
      exploits.append(info.exploitability.sum())

    elapsed = time.time() - t0
    print("Time for {} eval traversals {} sec".format(self.opt.NUM_TRAVERSALS_EVAL, elapsed))

    mbb_per_game = 1e3 * torch.Tensor(exploits) / (2.0 * Constants.SMALL_BLIND_AMOUNT)
    mean_mbb_per_game = mbb_per_game.mean()
    stdev_mbb_per_game = mbb_per_game.std()

    writer = self.writers[mode]

    if mode == "train":
      writer.add_scalar("train_exploit_mbbg_mean/{}".format(t), mean_mbb_per_game, steps)
      writer.add_scalar("train_exploit_mbbg_stdev/{}".format(t), stdev_mbb_per_game, steps)
    
    # In eval mode, we log the mbb/g exploitability after each CFR iteration.
    else:
      writer.add_scalar("cfr_exploit_mbbg_mean", mean_mbb_per_game, t)
      writer.add_scalar("cfr_exploit_mbbg_stdev", stdev_mbb_per_game, t)
    
    writer.close()
    print("===> [EVAL] Exploitability | mean={} mbb/g | stdev={} | (cfr_iter={})".format(
        mean_mbb_per_game, stdev_mbb_per_game, t))

  def log(self, mode, traverse_player_idx, t, losses, steps):
    """
    Write an event to the tensorboard events file.
    """
    loss_name_advt = "mse_loss/{}/{}".format(t, traverse_player_idx)
    loss_name_strt = "strt_mse_loss"
    if loss_name_advt in losses:
      loss = losses[loss_name_advt]
      print("==> TRAINING [ADVT] | steps={} | loss={} | cfr_iter={}".format(steps, loss, t))
    elif loss_name_strt in losses:
      loss = losses[loss_name_strt]
      print("==> TRAINING [STRT] | steps={} | loss={} | cfr_iter={}".format(steps, loss, t))

    writer = self.writers[mode]
    for l, v in losses.items():
      writer.add_scalar("{}".format(l), v, steps)

    # For some reason need this for logging to work.
    writer.close()

  def load_networks(self, load_weights_path, t):
    if not os.path.exists(load_weights_path):
      print("WARNING: Load weights path {} does not exist".format(load_weights_path))

    for player_name in [0, 1]:
      load_path = os.path.join(load_weights_path, "value_network_{}.pth".format(player_name))
      if os.path.exists(load_path):
        model_dict = self.value_networks[player_name].network().state_dict()
        model_dict.update(torch.load(load_path))
        print("==> Loaded value network weights for {} from {}".format(player_name, load_path))

    # strt_load_path = os.path.join(load_weights_path, "strategy_network_{}.pth".format(t))
    # if os.path.exists(strt_load_path):
    #   model_dict = self.strategy_network.network().state_dict()
    #   model_dict.update(torch.load(strt_load_path))
    #   print("==> Loaded strategy network weights from", strt_load_path)
