import ray

from copy import deepcopy
import os, time

import torch
import torch.multiprocessing as mp

from pypokerengine.api.emulator import Emulator

from constants import Constants
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer, MemoryBufferDataset
from network_wrapper import NetworkWrapper
from ray_network_wrapper import RayNetworkWrapper
from trainer import generate_actions, make_infoset


NUM_TRAVERSALS_TOTAL = 4000
NUM_PROCESSES = 8
NUM_TRAVERSALS_EACH = int(NUM_TRAVERSALS_TOTAL / NUM_PROCESSES)


@ray.remote
def traverse_multiple(worker_id, traverse_player, strategies, advantage_mem, strategy_mem, t):
  for k in range(NUM_TRAVERSALS_EACH):
    ctr = [0]

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

    traverse(game_state, [events[-1]], emulator, generate_actions, make_infoset, traverse_player,
             strategies, advantage_mem, strategy_mem, t, recursion_ctr=ctr, remote=True)

    if (k % 100) == 0:
      print("Finished {}/{} traversals".format(k, NUM_TRAVERSALS_EACH))
      print("Traversal used {} recursive calls (worker={})".format(ctr[0], worker_id))


if __name__ == '__main__':
  ray.init()

  p1_strategy = RayNetworkWrapper.remote(4, Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 128,
                                torch.device("cuda:0" if torch.cuda.device_count() >= 2 else "cuda"))
  p2_strategy = RayNetworkWrapper.remote(4, Constants.BET_HISTORY_SIZE, Constants.NUM_ACTIONS, 128,
                                torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda"))
  # p1_strategy._network.share_memory()
  # p2_strategy._network.share_memory()

  strategies = {
    Constants.PLAYER1_UID: p1_strategy,
    Constants.PLAYER2_UID: p2_strategy
  }

  # advantage_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  # strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  advantage_mem = None
  strategy_mem = None

  t0 = time.time()

  results = ray.get([traverse_multiple.remote(1234, Constants.PLAYER1_UID, strategies, advantage_mem, strategy_mem, 0) for _ in range(NUM_PROCESSES)])
  # mp.spawn(
  #   traverse_multiple,
  #   args=(Constants.PLAYER1_UID, strategies, advantage_mem, strategy_mem, 0),
  #   nprocs=NUM_PROCESSES, join=True, daemon=False) #, start_method='spawn')

  elapsed = time.time() - t0
  print("Time for {} traversals across {} threads: {} sec".format(NUM_TRAVERSALS_TOTAL, NUM_PROCESSES, elapsed))
