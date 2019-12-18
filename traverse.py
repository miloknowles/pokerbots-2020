import torch
import numpy as np

from pypokerengine.api.emulator import Emulator

PLAYER1_UID = "P1"
PLAYER2_UID = "P2"
INITIAL_STACK = 100


def check_terminal_node(events):
  for e in events:
    if e["type"] == "event_round_finish":
      winning_player = e["winners"][0]["uuid"]
      winning_stack = e["winners"][0]["stack"]
      return True, winning_player, winning_stack
  return False, None, None


def check_player_node(events):
  e = events[-1]
  if e["type"] == "event_ask_player":
    return True, e["uuid"], e
  else:
    return False


def sample_uniform_action(valid_actions):
  item = valid_actions[np.random.randint(len(valid_actions))]
  amount = item["amount"]

  if type(amount) == dict:
    random_amount = np.random.randint(amount["min"], high=amount["max"]+1)
    return item["action"], random_amount
  else:
    return item["action"], item["amount"]


def get_available_actions(valid_actions):
  all_actions = []
  for item in valid_actions:
    # NOTE(milo): For now just considering a min and max bet.
    if type(item["amount"]) == dict:
      all_actions.append((item["action"], item["amount"]["min"]))
      all_actions.append((item["action"], item["amount"]["max"]))
    else:
      all_actions.append((item["action"], item["amount"]))
  return all_actions


def traverse(game_state, events, emulator, traverse_player, p1_strategy, p2_strategy, adv_mem, strategy_mem, t):
  """
  Recursively traverse the game tree with external sampling.
  """
  # CASE 1: State is terminal, return payoff for player.
  is_terminal_node, winning_player, winning_stack = check_terminal_node(events)
  if is_terminal_node:
    payoff = (winning_stack - INITIAL_STACK)
    return payoff if traverse_player == winning_player else (-1 * payoff)

  # CASE 2: State is a chance node, simulate the event. NOTE(milo): The game engine takes care of
  # this external sampling for us!

  # CASE 3: Traversal player action node.
  is_player_node, uuid, evt = check_player_node(events)
  if is_player_node and uuid == PLAYER1_UID:
    actions = get_available_actions(evt["valid_actions"])
    strategy = np.ones(len(actions)) / len(actions) # Uniform strategy over samples for now.
    values = np.zeros(len(actions))

    # TODO(milo): Compute strategy from regret matching.
    for i, a in enumerate(actions):
      updated_state, new_events = emulator.apply_action(game_state, a[0], a[1])
      values[i] = traverse(updated_state, new_events, emulator, traverse_player, p1_strategy, p2_strategy, adv_mem, strategy_mem, t)

    # TODO(milo): For each action, compute advantages.
    # TODO(milo): Insert infoset and action advantages into advantage memory.
    assert np.allclose(strategy.sum(), 1.0)
    return (values * strategy).sum()

  # CASE 4: Other player action node.
  elif is_player_node and uuid == PLAYER2_UID:
    # TODO(milo): Compute and sample from a strategy using regret matching.
    # TODO(milo): Insert the infoset and action probabilities into the strategy memory.
    action, amount = sample_uniform_action(evt["valid_actions"])
    updated_state, new_events = emulator.apply_action(game_state, action, amount)
    return traverse(updated_state, new_events, emulator, traverse_player, p1_strategy, p2_strategy, adv_mem, strategy_mem, t)

  else:
    raise NotImplementedError()


if __name__ == "__main__":
  emulator = Emulator()
  emulator.set_game_rule(player_num=1, max_round=10, small_blind_amount=5, ante_amount=1)

  players_info = {}
  players_info[PLAYER1_UID] = {"name": PLAYER1_UID, "stack": 100}
  players_info[PLAYER2_UID] = {"name": PLAYER2_UID, "stack": 100}

  initial_state = emulator.generate_initial_game_state(players_info)
  game_state, events = emulator.start_new_round(initial_state)

  evs = []
  for _ in range(100):
    ev = traverse(game_state, events, emulator, "P1", None, None, None, None, 0)
    evs.append(ev)

  avg_ev = np.array(evs).mean()
  print("Average EV={}".format(avg_ev))
