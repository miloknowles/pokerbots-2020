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
