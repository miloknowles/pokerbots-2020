import torch.multiprocessing as mp
import torch

import time, random
import pickle

from pypokerengine.api.emulator import Emulator

from traverse import traverse_until_turn
from trainer import generate_actions
from constants import Constants


def traverse_until_turn_worker(traverse_player):
  result = None
  while result is None:
    k = (random.random() > 0.5)

    emulator = Emulator()
    emulator.set_game_rule(
      player_num=k % 2,
      max_round=10,
      small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
      ante_amount=0)

    players_info = {}
    players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
    players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}

    initial_state = emulator.generate_initial_game_state(players_info)
    game_state, events = emulator.start_new_round(initial_state)

    ctr = [0]
    result = traverse_until_turn(game_state, events, emulator, generate_actions, traverse_player, recursion_ctr=ctr)
  return result


if __name__ == "__main__":
  num_proc = 8
  pool = mp.Pool(processes=num_proc)

  num_jobs = 1000

  t0 = time.time()

  random_turn_situations = pool.starmap_async(
      traverse_until_turn_worker, [(Constants.PLAYER1_UID,) for _ in range(num_jobs)])
  out = random_turn_situations.get()
  print(len(out))

  print(out[0])

  with open("tmp.pkl", "wb") as f:
    pickle.dump(out, f)

  elapsed = time.time() - t0
  print("Randomly sampled {} turn scenarios in {} sec".format(num_jobs, elapsed))

  with open("tmp.pkl", "rb") as f:
    out = pickle.load(f)
    print(out)
