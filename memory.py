import torch
import numpy as np


class InfoSet(object):
  def __init__(self, hole_cards, board_cards, bet_history_vec):
    """
    hole_cards (np.array): The 0-51 encoding of each hole card.
    board_cards (np.array) : The 0-51 encoding of each of 3-5 board cards.
    bet_history_vec (np.array) : Betting actions, represented as a fraction of the pot size.

    The bet history has size (num_streets * num_actions_per_street) = 6 * 4 = 24.
    """
    self.hole_cards = hole_cards
    self.board_cards = board_cards
    self.bet_history_vec = bet_history_vec
  
  def pack(self):
    """
    Packs the infoset into a compact numpy array of size:
      (2 hole cards, 5 board cards, num_betting_actions)
    """
    board_cards_fixed_size = np.zeros(5)
    board_cards_fixed_size[:len(self.board_cards)] = self.board_cards
    return np.concatenate([self.hole_cards, board_cards_fixed_size, self.bet_history_vec]).astype(np.float32)
    

class MemoryBuffer(object):
  def __init__(self, info_set_size, item_size, max_size=80000, store_weights=False):
    self._infosets = np.zeros((max_size, info_set_size), dtype=np.float32)
    self._items = np.zeros((max_size, item_size), dtype=np.float32)
    if store_weights:
      self._has_weights = True
      self._weights = np.zeros(max_size, dtype=np.float32)
    else:
      self._has_weights = False

    self._next_index = 0

  def add(self, infoset, item):
    if self.full():
      return
    self._infosets[self._next_index] = infoset
    self._items[self._next_index] = item
    self._next_index += 1

  def add_weighted(self, infoset, item, weight):
    if self.full():
      return
    self._weights[self._next_index] = weight
    self.add(infoset, item)

  def full(self):
    return self._next_index >= self._infosets.shape[0]
