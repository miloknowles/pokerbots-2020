from copy import deepcopy

import torch

from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer
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
  NUM_CFR_ITERS = 100               # Exploitability seems to converge around 100 iters.
  NUM_TRAVERSALS_PER_ITER = 1e5     # 100k seems to be the best in Brown et. al.
  MEM_BUFFER_MAX_SIZE = 1e6         # Brown. et. al. use 40 million for all 3 buffers.
  EMBED_DIM = 128                   # Seems like this gave the best performance.

  SGD_ITERS = 32000                 # Same as Brown et. al.
  SGD_LR = 1e-3                     # Same as Brown et. al.
  SGD_BATCH_SIZE = 20000            # Same as Brown et. al.

  DEVICE = torch.device("cuda")


def run_deep_cfr(params):
  advantage_mems = {
    Constants.PLAYER1_UID: MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True),
    Constants.PLAYER2_UID: MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  }

  value_networks = {
    Constants.PLAYER1_UID: NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, params.EMBED_DIM),
    Constants.PLAYER2_UID: NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, params.EMBED_DIM)
  }

  strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)

  # Set up the game emulator.
  emulator = Emulator()
  emulator.set_game_rule(
    player_num=1,
    max_round=params.NUM_TRAVERSALS_PER_ITER,
    small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
    ante_amount=0)

  for t in range(params.NUM_CFR_ITERS):
    for traverse_player in [Constants.PLAYER1_UID, Constants.PLAYER2_UID]:
      for k in range(params.NUM_TRAVERSALS_PER_ITER):
        # Make sure each player has a full starting stack at the beginning of the round.
        players_info = {}
        players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
        players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}
      
        initial_state = emulator.generate_initial_game_state(players_info)
        game_state, events = emulator.start_new_round(initial_state)

        # Collect training samples by traversing the game tree with external sampling.
        traverse(game_state, events, emulator, generate_actions, make_infoset,
                 traverse_player, value_networks[Constants.PLAYER1_UID],
                 value_networks[Constants.PLAYER2_UID], advantage_mems[traverse_player],
                 strategy_mem, t)
  
    # Train the advantage network from scratch using samples from the traverse player's buffer.
    model_wrap = value_networks[traverse_player]
    model_wrap.reset_network()

    net = model_wrap._network
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=params.SGD_LR)

    for batch_idx, (inputs, regrets) in enumerate(train_loader):
      inputs, regrets = inputs.to(params.DEVICE), regrets.to(params.DEVICE)

      optimizer.zero_grad()
      output = net(inputs)

      loss = torch.nn.functional.mse_loss(inputs, regrets)
      loss.backward()

      optimizer.step()


if __name__ == "__main__":
  emulator = Emulator()
  emulator.set_game_rule(
    player_num=1,
    max_round=10,
    small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
    ante_amount=0)

  players_info = {}
  players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
  players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}

  initial_state = emulator.generate_initial_game_state(players_info)
  game_state, events = emulator.start_new_round(initial_state)

  p1_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)
  p2_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)

  advantage_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)

  evs = []
  t = 0
  for _ in range(100):
    ev = traverse(game_state, events, emulator, generate_actions, make_infoset,
                  Constants.PLAYER1_UID, p1_strategy, p2_strategy, advantage_mem, strategy_mem, t)
    evs.append(ev)

  avg_ev = np.array(evs).mean()
  print("Average EV={}".format(avg_ev))
