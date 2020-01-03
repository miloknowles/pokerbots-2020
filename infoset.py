import torch

from constants import Constants


class InvalidBoardSizeException(Exception):
  pass


class IncompatibleInfosetException(Exception):
  pass


class InfoSet(object):
  def __init__(self, hole_cards, board_cards, bet_history_vec, player_position):
    """
    hole_cards (torch.Tensor): The 0-51 encoding of each  of (2) hole cards.
    board_cards (torch.Tensor) : The 0-51 encoding of each of (0, 3, or 5) board cards.
    bet_history_vec (torch.Tensor) : Betting actions, represented as a fraction of the pot size.
    player_position (int) : 0 if the acting player is the SB and 1 if they are BB.

    The bet history has size (num_streets * num_actions_per_street) = 6 * 4 = 24.
    """
    if len(board_cards) not in [0, 3, 4, 5]:
      raise InvalidBoardSizeException()

    self.hole_cards = hole_cards
    self.board_cards = board_cards
    self.bet_history_vec = bet_history_vec
    self.player_position = player_position

  def get_card_input_tensors(self):
    """
    Returns hole cards (1 x 2) and board cards (1 x 5).
    """
    if len(self.board_cards) not in [0, 3, 4, 5]:
      raise InvalidBoardSizeException()

    hole_cards = self.hole_cards.unsqueeze(0).long()
    board_cards = -1 * torch.ones(1, 5).long()
    board_cards[:,:len(self.board_cards)] = self.board_cards.unsqueeze(0).long()

    return hole_cards, board_cards

  def get_bet_input_tensors(self):
    """
    The network expects (torch.Tensor) with shape (B x num_betting_actions).
    """
    nbets = self.bet_history_vec.shape[0]
    position_mask = torch.zeros(nbets)
    position_mask[torch.arange(self.player_position, nbets, 2).long()] = 1
    position_mask[torch.arange((self.player_position + 1) % 2, nbets, 2).long()] = -1
    return self.bet_history_vec.unsqueeze(0), position_mask
  
  def pack(self):
    """
    Packs the infoset into a compact torch.Tensor of size:
      (1 player position, 2 hole cards, 5 board cards, num_betting_actions)
    """
    board_cards_fixed_size = -1 * torch.ones(5)
    board_cards_fixed_size[:len(self.board_cards)] = self.board_cards
    return torch.cat([
      torch.Tensor([self.player_position]),
      self.hole_cards,
      board_cards_fixed_size,
      self.bet_history_vec])


def unpack_infoset(tensor):
  """
  Unpack a compactified infoset tensor into an Infoset object.
  """
  player_position = tensor[0]
  hole_cards = tensor[1:3]
  num_board_cards = torch.sum(tensor[3:8] >= 0) # Cards with -1 indicate NA.
  board_cards = tensor[3:3+num_board_cards]
  bet_history_vec = tensor[1+2+5:]

  if len(bet_history_vec) != Constants.NUM_BETTING_ACTIONS:
    raise IncompatibleInfosetException()

  return InfoSet(hole_cards, board_cards, bet_history_vec, player_position)
