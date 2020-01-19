import torch

from pypokerengine.players import BasePokerPlayer

from constants import Constants
from network_wrapper import NetworkWrapper
from trainer import make_infoset, make_infoset_helper, generate_actions


class DeepCFRPlayer(BasePokerPlayer):
  def __init__(self, load_weights_path):
    self.wrap = NetworkWrapper(Constants.NUM_STREETS, Constants.BET_HISTORY_SIZE,
                               Constants.NUM_ACTIONS, 64, device=torch.device("cuda:0"))

    if load_weights_path is not None:
      model_dict = self.wrap.network().state_dict()
      model_dict.update(torch.load(load_weights_path))
      print("==> Loaded value network weights from {}".format(load_weights_path))

  def declare_action(self, valid_actions, hole_card, round_state):
    """
    valid_actions (list of dict) : i.e [{'action': 'fold', 'amount': 0}, {'action': 'call', 'amount': 2}]
    hole_card (list of str) : i.e ['H2', 'C8']
    round_state (dict) : i.e {'action_histories': {'preflop': [{'add_amount': 1, 'action': 'SMALLBLIND', 'amount': 1, 'uuid': 'lgwgyegccmfobokqugvwud'}, {'add_amount': 1, 'action': 'BIGBLIND', 'amount': 2, 'uuid': 'jkuneswljvhvmhmcopbpmj'}]}, 'community_card': [], 'round_count': 1, 'big_blind_pos': 0, 'seats': [{'name': 'P1', 'stack': 98, 'uuid': 'jkuneswljvhvmhmcopbpmj', 'state': 'participating'}, {'name': 'P2', 'stack': 99, 'uuid': 'lgwgyegccmfobokqugvwud', 'state': 'participating'}], 'street': 'preflop', 'pot': {'main': {'amount': 3}, 'side': []}, 'next_player': 1, 'small_blind_pos': 1, 'dealer_btn': 0, 'small_blind_amount': 1}
    """
    pot_size = round_state["pot"]["main"]["amount"]
    actions, mask = generate_actions(valid_actions, pot_size)

    infoset = make_infoset_helper(hole_card, round_state)
    action_probs = self.wrap.get_action_probabilities(infoset, mask)

    # If no predicted actions are valid, choose one of the valid actions at random.
    action_probs *= mask
    if (action_probs.sum() <= 0):
      print("WARNING: Had to choose a random valid action")
      action_probs = mask / mask.sum()

    action, amount = actions[torch.multinomial(action_probs, 1).item()]

    return action, amount

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
