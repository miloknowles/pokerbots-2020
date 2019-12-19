from sys import getsizeof

import torch


class InfoSet(object):
  def __init__(self, hole_cards, board_cards, bet_history_vec, player_position):
    """
    hole_cards (torch.Tensor): The 0-51 encoding of each  of (2) hole cards.
    board_cards (torch.Tensor) : The 0-51 encoding of each of (3-5) board cards.
    bet_history_vec (torch.Tensor) : Betting actions, represented as a fraction of the pot size.
    player_position (int) : 0 if the acting player is the SB and 1 if they are BB.

    The bet history has size (num_streets * num_actions_per_street) = 6 * 4 = 24.
    """
    self.hole_cards = hole_cards
    self.board_cards = board_cards
    self.bet_history_vec = bet_history_vec
    self.player_position = 0
  
  def pack(self):
    """
    Packs the infoset into a compact torch.Tensor of size:
      (1 player position, 2 hole cards, 5 board cards, num_betting_actions)
    """
    board_cards_fixed_size = torch.zeros(5)
    board_cards_fixed_size[:len(self.board_cards)] = self.board_cards
    return torch.cat([
      torch.Tensor([self.player_position]),
      self.hole_cards,
      board_cards_fixed_size,
      self.bet_history_vec])
    

class MemoryBuffer(object):
  def __init__(self, info_set_size, item_size, max_size=80000, store_weights=False,
               device=torch.device("cpu")):
    self._device = device
    self._infosets = torch.zeros((int(max_size), info_set_size), dtype=torch.float32).to(self._device)
    self._items = torch.zeros((int(max_size), item_size), dtype=torch.float32).to(self._device)
    if store_weights:
      self._has_weights = True
      self._weights = torch.zeros(int(max_size), dtype=torch.float32).to(self._device)
    else:
      self._has_weights = False

    self._next_index = 0

  def add(self, infoset, item):
    if self.full():
      return
    self._infosets[self._next_index] = infoset.pack()
    self._items[self._next_index] = item
    self._next_index += 1

  def add_weighted(self, infoset, item, weight):
    if self.full():
      return
    self._weights[self._next_index] = weight
    self.add(infoset, item)

  def full(self):
    return self._next_index >= self._infosets.shape[0]

  def size_mb(self):
    total = getsizeof(self._infosets.storage())
    total += getsizeof(self._items.storage())
    if self._has_weights:
      total += getsizeof(self._weights.storage())
    return total / 1e6
