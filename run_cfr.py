from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer
from regret_matching_strategy import RegretMatchingStrategy


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

  p1_strategy = RegretMatchingStrategy()
  p2_strategy = RegretMatchingStrategy()

  evs = []
  for _ in range(1):
    ev = traverse(game_state, events, emulator, Constants.PLAYER1_UID, p1_strategy, p2_strategy, None, None, 0)
    evs.append(ev)

  avg_ev = np.array(evs).mean()
  print("Average EV={}".format(avg_ev))
