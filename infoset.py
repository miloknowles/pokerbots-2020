import torch

from constants import Constants


class InvalidBoardSizeException(Exception):
  pass


class IncompatibleInfosetException(Exception):
  pass


class EvInfoSet(object):
  def __init__(self, ev, bet_history_vec, player_position):
    """
    ev (float) : EV of the current hand and board.
    bet_history_vec (torch.Tensor) : Betting actions, represented as a fraction of the pot size.
    player_position (int) : 0 if the acting player is the SB and 1 if they are BB.
    """
    self.ev = ev
    self.bet_history_vec = bet_history_vec
    self.player_position = player_position
  
  def get_ev_input_tensors(self):
    """
    Returns: (torch.Tensor) with shape (batch_size x 1).
    """
    return torch.Tensor([self.ev]).unsqueeze(0)
  
  def get_bet_input_tensors(self):
    """
    Returns: (torch.Tensor) with shape (batch_size x BET_HISTORY_SIZE).
    """
    nbets = self.bet_history_vec.shape[0]
    position_mask = torch.zeros(nbets)
    position_mask[torch.arange(self.player_position, nbets, 2).long()] = 1
    position_mask[torch.arange((self.player_position + 1) % 2, nbets, 2).long()] = -1
    return self.bet_history_vec.unsqueeze(0), position_mask
  
  def pack(self):
    """
    Packs the infoset into a compact torch.Tensor of size:
      (1 player position, 1 ev, BET_HISTORY_SIZE)
    """
    return torch.cat([
      torch.Tensor([self.player_position]),
      torch.Tensor([self.ev]),
      self.bet_history_vec])


def unpack_ev_infoset(tensor):
  """
  Unpack a compactified infoset tensor into an Infoset object.
  """
  player_position = tensor[0]
  player_ev = tensor[1]
  bet_history_vec = tensor[2:]

  if len(bet_history_vec) != Constants.BET_HISTORY_SIZE:
    raise IncompatibleInfosetException()

  return EvInfoSet(player_ev, bet_history_vec, player_position)
