import ray
from copy import deepcopy
import os, time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory_buffer import MemoryBuffer
from infoset import InfoSet
from memory_buffer_dataset import MemoryBufferDataset
from network_wrapper import NetworkWrapper


def make_infoset_helper(hole_suit_rank, round_state):
  """
  Make an infoset representation for the player about to act.
  """
  # NOTE(milo): Acting position is 0 if this player is the SB (first to act) and 1 if BB.
  small_blind_player_idx = (round_state["big_blind_pos"] + 1) % 2
  acting_player_idx = int(round_state["next_player"])

  # This is 0 if the current acting player is the SB and 1 if BB.
  acting_player_blind = 0 if small_blind_player_idx == acting_player_idx else 1

  # NOTE(milo): PyPokerEngine encodes cards with rank-suit i.e CJ.
  board_suit_rank = round_state["community_card"]

  bet_history_vec = torch.zeros(Constants.NUM_BETTING_ACTIONS)
  h = round_state["action_histories"]
  
  # Always start out with SB + BB in the pot.
  pot_total = (3 * Constants.SMALL_BLIND_AMOUNT)
  for street_num, street in enumerate(["preflop", "flop", "turn", "river"]):
    if street in h:
      for i, action in enumerate(h[street]):
        # Skip actions if they exceed the number of betting actions we consider.
        if (street_num * Constants.STREET_OFFSET + i) >= len(bet_history_vec):
          continue
        # Percentage of CURRENT pot.
        bet_history_vec[street_num * Constants.STREET_OFFSET + i] = action["amount"] / pot_total
        pot_total += action["amount"]

  infoset = InfoSet(
    encode_cards_suit_rank(hole_suit_rank),
    encode_cards_suit_rank(board_suit_rank),
    bet_history_vec,
    acting_player_idx)

  return infoset


def make_infoset(game_state, evt):
  acting_player_idx = int(evt["round_state"]["next_player"])
  players = game_state["table"].seats.players
  hole_suit_rank = [
    str(players[acting_player_idx].hole_card[0]),
    str(players[acting_player_idx].hole_card[1])]

  return make_infoset_helper(hole_suit_rank, evt["round_state"])


def generate_actions(valid_actions, pot_amount):
  """
  Using the valid_actions from the game engine, mask out and scale the entire set of actions.
  """
  actions_mask = torch.zeros(len(Constants.ALL_ACTIONS))
  actions_scaled = deepcopy(Constants.ALL_ACTIONS)

  for item in valid_actions:
    if item["action"] == "fold":
      actions_mask[Constants.ACTION_FOLD] = 1

    elif item["action"] == "call":
      actions_mask[Constants.ACTION_CALL] = 1
      actions_scaled[Constants.ACTION_CALL][1] = item["amount"]

    elif item["action"] == "raise":
      min_raise, max_raise = item["amount"]["min"], item["amount"]["max"]

      # actions_mask[Constants.ACTION_MINRAISE] = 1
      actions_mask[Constants.ACTION_MAXRAISE] = 1
      # actions_scaled[Constants.ACTION_MINRAISE][1] = min_raise
      actions_scaled[Constants.ACTION_MAXRAISE][1] = max_raise

      if pot_amount <= max_raise:
        actions_mask[Constants.ACTION_POTRAISE] = 1
        actions_scaled[Constants.ACTION_POTRAISE][1] = pot_amount

      # if 2 * pot_amount <= max_raise:
      #   actions_mask[Constants.ACTION_TWOPOTRAISE] = 1
      #   actions_scaled[Constants.ACTION_TWOPOTRAISE][1] = 2 * pot_amount
      
      # if 3 * pot_amount <= max_raise:
      #   actions_mask[Constants.ACTION_THREEPOTRAISE] = 1
      #   actions_scaled[Constants.ACTION_THREEPOTRAISE][1] = 3 * pot_amount

  return actions_scaled, actions_mask


def traverse_worker(worker_id, traverse_player, strategies, save_lock, opt, t, eval_mode,
                    info_queue):
  """
  A worker that traverses the game tree K times, saving things to memory buffers. Each worker
  maintains its own memory buffers and saves them after finishing.

  If eval_mode is set to True, no memory buffers are created.
  """
  advt_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                          max_size=opt.SINGLE_PROC_MEM_BUFFER_MAX_SIZE,
                          autosave_params=(opt.MEMORY_FOLDER, opt.ADVT_BUFFER_FMT.format(traverse_player)),
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
    emulator = Emulator()
    emulator.set_game_rule(
      player_num=k % 2,
      max_round=5,
      small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
      ante_amount=Constants.ANTE_AMOUNT)

    players_info = {}
    players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
    players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}

    initial_state = emulator.generate_initial_game_state(players_info)
    game_state, events = emulator.start_new_round(initial_state)

    info = traverse(game_state, [events[-1]], emulator, generate_actions, make_infoset, traverse_player,
             strategies, advt_mem, strt_mem, t, recursion_ctr=ctr, do_external_sampling=not eval_mode)

    if info_queue is not None:
      info_queue.put(info, True, 0.1)

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
      Constants.PLAYER1_UID: NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                            Constants.NUM_ACTIONS, opt.EMBED_DIM, device=opt.DEVICE),
      Constants.PLAYER2_UID: NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                            Constants.NUM_ACTIONS, opt.EMBED_DIM, device=opt.DEVICE)
    }
    self.value_networks[Constants.PLAYER1_UID]._network.share_memory()
    self.value_networks[Constants.PLAYER2_UID]._network.share_memory()
    print("[DONE] Made value networks")

    # TODO(milo): Does this need to be different than the value networks?
    self.strategy_network = NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                           Constants.NUM_ACTIONS, opt.EMBED_DIM)
    print("[DONE] Made strategy network")

    self.writers = {}
    for mode in ["train", "cfr"]:
      self.writers[mode] = SummaryWriter(os.path.join(opt.TRAIN_LOG_FOLDER, mode))

  def main(self):
    eval_t = 0
    for t in range(self.opt.NUM_CFR_ITERS):
      for traverse_player in [Constants.PLAYER1_UID, Constants.PLAYER2_UID]:
        self.do_cfr_iter_for_player(traverse_player, t)
        self.train_value_network(traverse_player, t)
        self.eval_value_network("cfr", eval_t, None)
        eval_t += 1
    # TODO(milo): Train strategy network.
  
  def do_cfr_iter_for_player(self, traverse_player, t):
    self.value_networks[Constants.PLAYER1_UID]._network.share_memory()
    self.value_networks[Constants.PLAYER2_UID]._network.share_memory()

    manager = mp.Manager()
    save_lock = manager.Lock()
    progress_queue = manager.Queue()

    t0 = time.time()

    mp.spawn(
      traverse_worker,
      args=(traverse_player, self.value_networks, save_lock, self.opt, t, False, None),
      nprocs=self.opt.NUM_TRAVERSE_WORKERS, join=True, daemon=False)

    elapsed = time.time() - t0
    print("Time for {} traversals across {} workers: {} sec".format(
      self.opt.NUM_TRAVERSALS_PER_ITER, self.opt.NUM_TRAVERSE_WORKERS, elapsed))

  def linear_cfr_loss(self, output, target, weights, T):
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

  def train_value_network(self, traverse_player, t):
    """
    Train a value network from scratch using samples from the traverse player's buffer.
    """
    losses = {}

    # This causes the network to be reset.
    model_wrap = self.value_networks[traverse_player]
    model_wrap = NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                Constants.NUM_ACTIONS, self.opt.EMBED_DIM, device=self.opt.DEVICE)

    net = model_wrap.network()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=self.opt.SGD_LR)

    buffer_name = self.opt.ADVT_BUFFER_FMT.format(traverse_player)
    train_dataset = MemoryBufferDataset(self.opt.MEMORY_FOLDER, buffer_name,
                                        self.opt.TRAIN_DATASET_SIZE)
    train_loader = DataLoader(train_dataset,
                              batch_size=self.opt.SGD_BATCH_SIZE,
                              num_workers=self.opt.NUM_DATA_WORKERS)

    # Due to memory limitations, we can only store a subset of the dataset in memory at a time.
    # Calculate the number of times we'll have to resample and iterate over the dataset.
    num_resample_iters = int((self.opt.SGD_ITERS * self.opt.SGD_BATCH_SIZE) / len(train_dataset))

    step = 0
    for resample_iter in range(num_resample_iters):
      print(">> Doing resample iteration {}/{} ...".format(resample_iter, num_resample_iters))
      train_dataset.resample()
      print(">> Done. DataLoader has {} batches of size {}.".format(len(train_loader), self.opt.SGD_BATCH_SIZE))

      for batch_idx, input_dict in enumerate(train_loader):
        hole_cards_input = input_dict["hole_cards"].to(self.opt.DEVICE)
        board_cards_input = input_dict["board_cards"].to(self.opt.DEVICE)

        bets_input = input_dict["bets_input"].to(self.opt.DEVICE)
        advt_target = input_dict["target"].to(self.opt.DEVICE)

        weights = input_dict["weights"].to(self.opt.DEVICE)

        optimizer.zero_grad()

        # Minimize MSE between predicted advantage and instantaneous regret samples.
        output = net(hole_cards_input, board_cards_input, bets_input)
        loss = self.linear_cfr_loss(output, advt_target, weights, t)
        # loss = torch.nn.functional.mse_loss(output, advt_target)
        loss.backward()
        losses["mse_loss/{}/{}".format(t, traverse_player)] = loss.cpu().item()

        optimizer.step()

        if (batch_idx % self.opt.TRAINING_LOG_HZ) == 0:
          self.log("train", traverse_player, t, losses, step)

        # Only need to save the value network for the traversing player.
        if (batch_idx % self.opt.TRAINING_VALUE_NET_SAVE_HZ) == 0:
          self.save_models(t, save_value_networks=[traverse_player], save_strategy_network=False)

        if (batch_idx % self.opt.TRAINING_VALUE_NET_EVAL_HZ) == 0 and batch_idx > 0:
          self.eval_value_network("train", t, step)
        
        step += 1

  def save_models(self, t, save_value_networks=[], save_strategy_network=False):
    """
    Save model weights to disk.

    t (int) : The CFR iteration that these models are being trained on.
    save_value_networks (list of str) : Zero or more of [PLAYER1_UID, PLAYER2_UID].
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

  def eval_value_network(self, mode, t, steps):
    """
    Evaluate the (total) exploitability of the value networks, as in Brown et. al.
    """
    manager = mp.Manager()
    save_lock = manager.Lock()
    info_queue = manager.Queue()

    t0 = time.time()

    # Use worker with eval_mode = True.
    mp.spawn(
      traverse_worker,
      args=(Constants.PLAYER1_UID, self.value_networks, save_lock, self.opt, t, True, info_queue),
      nprocs=self.opt.NUM_TRAVERSE_WORKERS, join=True, daemon=False)

    elapsed = time.time() - t0
    print("Time for {} traversals across {} workers: {} sec".format(
      self.opt.NUM_TRAVERSALS_EVAL, self.opt.NUM_TRAVERSE_WORKERS, elapsed))

    total_exploits = []

    while not info_queue.empty():
      info = info_queue.get_nowait()
      total_exploits.append(info.exploitability.sum())

    mbb_per_game = 1e3 * torch.Tensor(total_exploits) / (2.0 * Constants.SMALL_BLIND_AMOUNT)
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

  def log(self, mode, traverse_player, t, losses, steps):
    """
    Write an event to the tensorboard events file.
    """
    loss = losses["mse_loss/{}/{}".format(t, traverse_player)]
    print("==> TRAINING | steps={} | loss={} | cfr_iter={}".format(steps, loss, t))

    writer = self.writers[mode]
    for l, v in losses.items():
      writer.add_scalar("{}".format(l), v, steps)

    # For some reason need this for logging to work.
    writer.close()

  def load_networks(self, load_weights_path, t):
    if not os.path.exists(load_weights_path):
      print("WARNING: Load weights path {} does not exist".format(load_weights_path))

    for player_name in [Constants.PLAYER1_UID, Constants.PLAYER2_UID]:
      load_path = os.path.join(load_weights_path, "value_network_{}.pth".format(player_name))
      if os.path.exists(load_path):
        model_dict = self.value_networks[player_name].network().state_dict()
        model_dict.update(torch.load(load_path))
        print("==> Loaded value network weights for {} from {}".format(player_name, load_path))

    strt_load_path = os.path.join(load_weights_path, "strategy_network_{}.pth".format(t))
    if os.path.exists(strt_load_path):
      model_dict = self.strategy_network.network().state_dict()
      model_dict.update(torch.load(strt_load_path))
      print("==> Loaded strategy network weights from", strt_load_path)
