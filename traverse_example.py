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
from trainer import generate_actions, make_infoset


NUM_TRAVERSALS_TOTAL = 400
NUM_PROCESSES = 2
NUM_TRAVERSALS_EACH = int(NUM_TRAVERSALS_TOTAL / NUM_PROCESSES)

def traverse_multiple(worker_id, game_state, events, emulator, action_generator, infoset_generator,
                      traverse_player, strategies, advantage_mem, strategy_mem, t):
  for _ in range(NUM_TRAVERSALS_EACH):
    ctr = [0]

    players_info = {}
    players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
    players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}
    
    initial_state = emulator.generate_initial_game_state(players_info)
    game_state, events = emulator.start_new_round(initial_state)

    traverse(game_state, [events[-1]], emulator, action_generator, infoset_generator, traverse_player,
             strategies, advantage_mem, strategy_mem, t, ctr)
    # print("Traversal used {} recursive calls (worker={})".format(ctr[0], worker_id))


if __name__ == '__main__':
  emulator = Emulator()
  emulator.set_game_rule(
    player_num=1,
    max_round=1000,
    small_blind_amount=Constants.SMALL_BLIND_AMOUNT,
    ante_amount=0)

  players_info = {}
  players_info[Constants.PLAYER1_UID] = {"name": Constants.PLAYER1_UID, "stack": Constants.INITIAL_STACK}
  players_info[Constants.PLAYER2_UID] = {"name": Constants.PLAYER2_UID, "stack": Constants.INITIAL_STACK}

  initial_state = emulator.generate_initial_game_state(players_info)
  game_state, events = emulator.start_new_round(initial_state)

  p1_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)
  p2_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, 128)
  p1_strategy._network.share_memory()
  p2_strategy._network.share_memory()

  strategies = {
    Constants.PLAYER1_UID: p1_strategy,
    Constants.PLAYER2_UID: p2_strategy
  }

  # advantage_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  # strategy_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS, max_size=1e6, store_weights=True)
  advantage_mem = None
  strategy_mem = None

  t0 = time.time()

  mp.spawn(
    traverse_multiple,
    args=(game_state, events, emulator, generate_actions, make_infoset, Constants.PLAYER1_UID,
    strategies, advantage_mem, strategy_mem, 0),
    nprocs=NUM_PROCESSES, join=True, daemon=False) #, start_method='spawn')

  elapsed = time.time() - t0
  print("Time for {} traversals across {} threads: {} sec".format(NUM_TRAVERSALS_TOTAL, NUM_PROCESSES, elapsed))
