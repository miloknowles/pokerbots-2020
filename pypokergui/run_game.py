import torch
from pypokerengine.api.game import setup_config, start_poker

import sys
sys.path.append("/home/milo/pokerbots-2020/")

from deep_cfr_player import DeepCFRPlayer


if __name__ == "__main__":
  P1 = DeepCFRPlayer("/home/milo/pokerbots-2020/training_logs/analysis/weights_6/value_network_P1.pth")
  P2 = DeepCFRPlayer("/home/milo/pokerbots-2020/training_logs/analysis/weights_0/value_network_P1.pth")

  num_trials = 10

  total_winnings = torch.zeros(num_trials, 2)

  for trial in range(num_trials):
    k = 1000
    for i in range(k):
      config = setup_config(max_round=1, initial_stack=100, small_blind_amount=1)

      config.register_player(name="P1", algorithm=P1)
      config.register_player(name="P2", algorithm=P2)
      game_result = start_poker(config, verbose=0)

      for p in game_result["players"]:
        if p["name"] == "P1":
          total_winnings[trial, 0] += (p["stack"] - game_result["rule"]["initial_stack"])
        else:
          total_winnings[trial, 1] += (p["stack"] - game_result["rule"]["initial_stack"])

  total_winnings_mean = torch.mean(total_winnings, axis=0)
  total_winnings_stdev = torch.std(total_winnings, axis=0)

  win_rates = 1e3 * total_winnings_mean / (2 * k)

  print("\n======================== RESULTS ==========================")
  print("Final stacks | P1_mean={} P1_stdev={} | P2_mean={} P2_stdev={}".format(
    total_winnings_mean[0], total_winnings_stdev[1], total_winnings_mean[1], total_winnings_stdev[1]))
  print("Win rates: P1={} mbb/g | P2={} mbb/g".format(win_rates[0], win_rates[1]))
