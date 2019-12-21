from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer, MemoryBufferDataset
from network_wrapper import NetworkWrapper


def make_infoset(game_state, evt):
  """
  Make an infoset representation for the player about to act.
  """
  # NOTE(milo): Acting position is 0 if this player is the SB (first to act) and 1 if BB.
  small_blind_player_idx = (evt["round_state"]["big_blind_pos"] + 1) % 2
  acting_player_idx = int(evt["round_state"]["next_player"])

  # This is 0 if the current acting player is the SB and 1 if BB.
  acting_player_blind = 0 if small_blind_player_idx == acting_player_idx else 1

  # NOTE(milo): PyPokerEngine encodes cards with rank-suit i.e CJ.
  board_suit_rank = evt["round_state"]["community_card"]

  players = game_state["table"].seats.players
  hole_suit_rank = [
    str(players[acting_player_idx].hole_card[0]),
    str(players[acting_player_idx].hole_card[1])]

  bet_history_vec = torch.zeros(Constants.NUM_BETTING_ACTIONS)
  h = evt["round_state"]["action_histories"]
  
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

      actions_mask[Constants.ACTION_MINRAISE] = 1
      actions_mask[Constants.ACTION_MAXRAISE] = 1
      actions_scaled[Constants.ACTION_MINRAISE][1] = min_raise
      actions_scaled[Constants.ACTION_MAXRAISE][1] = max_raise

      if pot_amount <= max_raise:
        actions_mask[Constants.ACTION_POTRAISE] = 1
        actions_scaled[Constants.ACTION_POTRAISE][1] = pot_amount

      if 2 * pot_amount <= max_raise:
        actions_mask[Constants.ACTION_TWOPOTRAISE] = 1
        actions_scaled[Constants.ACTION_TWOPOTRAISE][1] = 2 * pot_amount
      
      if 3 * pot_amount <= max_raise:
        actions_mask[Constants.ACTION_THREEPOTRAISE] = 1
        actions_scaled[Constants.ACTION_THREEPOTRAISE][1] = 3 * pot_amount

  return actions_scaled, actions_mask


class DeepCFRParams(object):
  EXPERIMENT_NAME = "deep_cfr_paper"

  NUM_CFR_ITERS = 100               # Exploitability seems to converge around 100 iters.
  NUM_TRAVERSALS_PER_ITER = 1e5     # 100k seems to be the best in Brown et. al.
  MEM_BUFFER_MAX_SIZE = 1e6         # Brown. et. al. use 40 million for all 3 buffers.
  EMBED_DIM = 128                   # Seems like this gave the best performance.

  SGD_ITERS = 32000                 # Same as Brown et. al.
  SGD_LR = 1e-3                     # Same as Brown et. al.
  SGD_BATCH_SIZE = 20000            # Same as Brown et. al.
  TRAIN_DATASET_SIZE = 1e6          # TODO(milo): Try something bigger?

  DEVICE = torch.device("cuda")
  NUM_DATA_WORKERS = 0

  MEMORY_FOLDER = os.path.join("./memory/", EXPERIMENT_NAME)
  TRAIN_LOG_FOLDER = os.path.join("./training_logs/", EXPERIMENT_NAME)

  ADVT_BUFFER_FMT = "advt_mem_{}"
  STRAT_BUFFER_FMT = "strat_mem_{}"


class Trainer(object):
  def __init__(self, params):
    self.params = params

    p1_avt_mem_name = params.ADVT_BUFFER_FMT.format(Constants.PLAYER1_UID)
    p2_avt_mem_name = params.ADVT_BUFFER_FMT.format(Constants.PLAYER2_UID)
    self.advantage_mems = {
      Constants.PLAYER1_UID: MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
      max_size=params.MEM_BUFFER_MAX_SIZE, autosave_params=(params.MEMORY_FOLDER, p1_avt_mem_name)),
      Constants.PLAYER2_UID: MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
      max_size=params.MEM_BUFFER_MAX_SIZE, autosave_params=(params.MEMORY_FOLDER, p2_avt_mem_name))
    }
    print("[DONE] Made ADVANTAGE memory")

    self.value_networks = {
      Constants.PLAYER1_UID: NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                            Constants.NUM_ACTIONS, params.EMBED_DIM),
      Constants.PLAYER2_UID: NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                            Constants.NUM_ACTIONS, params.EMBED_DIM)
    }
    print("[DONE] Made value networks")

    self.strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                                     max_size=params.MEM_BUFFER_MAX_SIZE)
    print("[DONE] Made strategy memory")

    # TODO(milo): Does this need to be different than the value networks?
    self.strategy_network = NetworkWrapper(Constants.NUM_STREETS, Constants.NUM_BETTING_ACTIONS,
                                           Constants.NUM_ACTIONS, params.EMBED_DIM)
    print("[DONE] Made strategy network")

    self.writers = {}
    for mode in ["train", "val"]:
      self.writers[mode] = SummaryWriter(os.path.join(params.TRAIN_LOG_FOLDER, mode))

  def main(self):
    for t in range(self.params.NUM_CFR_ITERS):
      for traverse_player in [Constants.PLAYER1_UID, Constants.PLAYER2_UID]:
        self.do_cfr_iteration_for_player(traverse_player, t)
        self.train_value_network(traverse_player, t)
    
    # Finally, train the strategy network.
    # TODO(milo)
  
  def do_cfr_iteration_for_player(self, t):
    emulator = Emulator()

    for k in range(self.params.NUM_TRAVERSALS_PER_ITER):
      emulator.set_game_rule(
        player_num=(k % 2),
        max_round=self.params.NUM_TRAVERSALS_PER_ITER,
        small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
        ante_amount=0)

      # Make sure each player has a full starting stack at the beginning of the round.
      players_info = {}
      players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
      players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}
    
      initial_state = emulator.generate_initial_game_state(players_info)
      game_state, events = emulator.start_new_round(initial_state)

      # Collect training samples by traversing the game tree with external sampling.
      traverse(game_state, events, emulator, generate_actions, make_infoset,
              traverse_player, self.value_networks[Constants.PLAYER1_UID],
              self.value_networks[Constants.PLAYER2_UID], self.advantage_mems[traverse_player],
              self.strategy_mem, t)

  def train_value_network(self, traverse_player, t):
    """
    Train the advantage network from scratch using samples from the traverse player's buffer.
    """
    losses = {}

    model_wrap = self.value_networks[traverse_player]
    model_wrap.reset_network()

    net = model_wrap.network()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=self.params.SGD_LR)

    buffer_name = self.params.ADVT_BUFFER_FMT.format(traverse_player)
    train_dataset = MemoryBufferDataset(self.params.MEMORY_FOLDER, buffer_name,
                                        self.params.TRAIN_DATASET_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=self.params.SGD_BATCH_SIZE,
                              num_workers=self.params.NUM_DATA_WORKERS)

    # Due to memory limitations, we can only store a subset of the dataset in memory at a time.
    # Calculate the number of times we'll have to resample and iterate over the dataset.
    num_resample_iters = (self.params.SGD_ITERS * self.params.SGD_BATCH_SIZE) / self.params.TRAIN_DATASET_SIZE

    for resample_iter in range(num_resample_iters):
      print(">> Doing resample iteration {}/{}".format(resample_iter, num_resample_iters))
      train_dataset.resample()

      for batch_idx, (inputs, regrets) in enumerate(train_loader):
        inputs, regrets = inputs.to(self.params.DEVICE), regrets.to(self.params.DEVICE)

        optimizer.zero_grad()

        # Get predicted advantage from network.
        output = net(inputs)

        # Minimize MSE between predicted advantage and instantaneous regret samples.
        loss = torch.nn.functional.mse_loss(inputs, regrets)
        loss.backward()
        losses["mse_loss/{}/{}".format(t, traverse_player)] = loss

        optimizer.step()

        if batch_idx % 1000 == 0:
          self.log("train", losses)
          # self.val()

  def save_models(self, t):
    """
    Save model weights to disk.
    """
    save_folder = os.path.join(self.params.TRAIN_LOG_FOLDER, "models", "weights_{}".format(t))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)

    for player_name, wrap in self.value_networks.items():
      save_path = os.path.join(save_folder, "value_network_{}.pth".format(player_name))
      to_save = wrap.network().state_dict()
      torch.save(to_save, save_path)

    # Save the strategy network also.
    save_path = os.path.join(save_folder, "strategy_nentwork.pth")
    to_save = self.strategy_network.network().state_dict()
    torch.save(to_save, save_path)

  def val(self):
    raise NotImplementedError()

  def log(self, mode, losses):
    """
    Write an event to the tensorboard events file.
    """
    writer = self.writers[mode]
    for l, v in losses.items():
      writer.add_scalar("{}".format(l), v, self.step)
