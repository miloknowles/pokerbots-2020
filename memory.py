from sys import getsizeof

import torch


class InvalidBoardSizeException(Exception):
  pass


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

  def get_card_input_tensors(self):
    """
    The network expects (tuple of torch.Tensor):
    Shape ((B x 2), (B x 3)[, (B x 1), (B x 1)]) # Hole, board [, turn, river]).
    """
    if len(self.board_cards) == 0:
      return [self.hole_cards.unsqueeze(0).long(), -1*torch.ones(1, 3).long(),
              -1*torch.ones(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 3:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards.unsqueeze(0).long(),
             -1*torch.ones(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 4:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards[:3].unsqueeze(0).long(),
              self.board_cards[3].view(1, 1).long(), -1*torch.ones(1, 1).long()]
    elif len(self.board_cards) == 5:
      return [self.hole_cards.unsqueeze(0).long(), self.board_cards[:3].unsqueeze(0).long(),
              self.board_cards[3].view(1, 1).long(), self.board_cards[4].view(1, 1).long()]
    else:
      raise InvalidBoardSizeException()

  def get_bet_input_tensors(self):
    """
    The network expects (torch.Tensor) with shape (B x num_betting_actions).
    """
    nbets = self.bet_history_vec.shape[0]
    position_mask = torch.zeros(nbets)
    position_mask[torch.arange(self.player_position, nbets, 2)] = 1
    position_mask[torch.arange((self.player_position + 1) % 2, nbets, 2)] = -1
    return self.bet_history_vec.unsqueeze(0), position_mask
  
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
