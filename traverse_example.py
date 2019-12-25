import ray

from copy import deepcopy
import os, time

import torch
import torch.multiprocessing as mp
from pypokerengine.api.emulator import Emulator

from constants import Constants
from options import Options
from utils import *
from traverse import traverse
from memory import InfoSet, MemoryBuffer, MemoryBufferDataset
from network_wrapper import NetworkWrapper
from trainer import generate_actions, make_infoset


NUM_TRAVERSALS_TOTAL = 10000
NUM_PROCESSES = 8
NUM_TRAVERSALS_EACH = int(NUM_TRAVERSALS_TOTAL / NUM_PROCESSES)


def traverse_multiple(worker_id, traverse_player, strategies, t, save_lock):
  advt_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                          max_size=1e4,
                          autosave_params=("./memory/traverse_example/", "p1_advt_mem"),
                          save_lock=save_lock)
  
  strt_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
                          max_size=1e4,
                          autosave_params=("./memory/traverse_example/", "strategy_mem"),
                          save_lock=save_lock)

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
             strategies, advt_mem, strt_mem, t, recursion_ctr=ctr)

    if (k % 100) == 0:
      print("Finished {}/{} traversals".format(k, NUM_TRAVERSALS_EACH))
      print("Traversal used {} recursive calls (worker={})".format(ctr[0], worker_id))
      # print("Memory sizes: value={} strategy={}".format(advantage_mem.size(), strategy_mem.size()))


def memory_manager(mem, in_queue):
  try:
    ctr = 0
    while True:
      if not in_queue.empty():
        tup = in_queue.get(True, 0.01)
        mem.add(tup[0], tup[1], tup[2])
        ctr += 1
        if (ctr % 100) == 0:
          print("Managed memory size = {}".format(mem.size()))
          print("Queue size:", in_queue.qsize())
          ctr = 0
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':
  opt = Options()

  p1_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, opt.EMBED_DIM,
                                torch.device("cuda:0" if torch.cuda.device_count() >= 2 else "cuda"))
  p2_strategy = NetworkWrapper(4, Constants.NUM_BETTING_ACTIONS, Constants.NUM_ACTIONS, opt.EMBED_DIM,
                                torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda"))

  p1_strategy._network.share_memory()
  p2_strategy._network.share_memory()

  strategies = {
    Constants.PLAYER1_UID: p1_strategy,
    Constants.PLAYER2_UID: p2_strategy
  }

  manager = mp.Manager()
  advantage_mem_queue = manager.Queue()
  strategy_mem_queue = manager.Queue()

  save_lock = manager.Lock()

  # advantage_mem = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
  #                              max_size=opt.MEM_BUFFER_MAX_SIZE,
  #                              autosave_params=("./memory/traverse_example/", "p1_advt_mem"),
  #                              save_lock=save_lock)
  # strategy_mem1 = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
  #                             max_size=opt.MEM_BUFFER_MAX_SIZE,
  #                             autosave_params=("./memory/traverse_example/", "strategy_mem"),
  #                             save_lock=save_lock)
  
  # strategy_mem2 = MemoryBuffer(Constants.INFO_SET_SIZE, Constants.NUM_ACTIONS,
  #                             max_size=opt.MEM_BUFFER_MAX_SIZE,
  #                             autosave_params=("./memory/traverse_example/", "strategy_mem"),
  #                             save_lock=save_lock)

  t0 = time.time()

  # advt_mem_worker1 = mp.Process(target=memory_manager, args=(advantage_mem, advantage_mem_queue))
  # advt_mem_worker1.start()

  # strat_mem_worker1 = mp.Process(target=memory_manager, args=(strategy_mem1, strategy_mem_queue))
  # strat_mem_worker1.start()

  # strat_mem_worker2 = mp.Process(target=memory_manager, args=(strategy_mem2, strategy_mem_queue))
  # strat_mem_worker2.start()

  # traverse_multiple(1234, Constants.PLAYER1_UID, strategies, advantage_mem_queue, strategy_mem_queue, 0)
    # with multiprocessing.Pool(nprocess) as pool:
        # pool.map(cycle, offsets)
  mp.spawn(
    traverse_multiple,
    args=(Constants.PLAYER1_UID, strategies, 0, save_lock),
    nprocs=NUM_PROCESSES, join=True, daemon=False)

  # advt_mem_worker1.terminate()
  # strat_mem_worker1.terminate()
  # strat_mem_worker2.terminate()

  elapsed = time.time() - t0
  print("Time for {} traversals across {} threads: {} sec".format(NUM_TRAVERSALS_TOTAL, NUM_PROCESSES, elapsed))
