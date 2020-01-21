import torch

from constants import Constants


class InvalidBoardSizeException(Exception):
  pass


class IncompatibleInfosetException(Exception):
  pass


class EvInfoSet(object):
  def __init__(self, ev, bet_history_vec, player_position, street):
    """
    ev (float) : EV of the current hand and board.
    bet_history_vec (torch.Tensor) : Betting actions, represented as a fraction of the pot size.
    player_position (int) : 0 if the acting player is the SB and 1 if they are BB.
    street (int) : 0, 1, 2, or 3.
    """
    self.ev = ev
    self.bet_history_vec = bet_history_vec
    self.player_position = player_position
    self.street = street
  
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


def bucket_small(infoset):
  """
  Apply a tiny abstraction to an infoset.

  - SB or BB (0 or 1)
  - Hand strengths are bucketed into { 0-40%, 40-60%, 60-80%, 80-100% }
  - Current street has 5 betting actions of { 0, C, P/2, P, 2P }
  - Previous streets are summarized by:
    - 0 or 1: whether the player raised
    - 0 or 1: whether the opponent raised

  [ SB/BB,
    CURRENT_HS,
    P_RAISED_P, P_RAISED_F, P_RAISED_T, P_RAISED_R,
    O_RAISED_P, O_RAISED_F, O_RAISED_T, O_RAISED_R,
    A0, A1, A2, A3
    CURRENT_STREET
  ]
  """
  # Entries with -1 are considered not filled in.
  h = ['x' for _ in range(1 + 1 + 1 + 4 + 4 + 4)]
  h[0] = 'SB' if infoset.player_position == 0 else 'BB'
  h[1] = {0: 'P', 1: 'F', 2: 'T', 3: 'R'}[infoset.street]
  
  if infoset.ev < 0.4:
    h[2] = 'H0'
  elif infoset.ev < 0.6:
    h[2] = 'H1'
  elif infoset.ev < 0.8:
    h[2] = 'H2'
  else:
    h[2] = 'H3'

  assert(len(infoset.bet_history_vec) == Constants.BET_HISTORY_SIZE)
  
  pips = [0, 0]
  plyr_raised_offset = 3
  opp_raised_offset = 7
  street_actions_offset = 11
  
  cumul = torch.cumsum(infoset.bet_history_vec, dim=0)

  for i in range(0, Constants.BET_HISTORY_SIZE):
    if (i % Constants.BET_ACTIONS_PER_STREET) == 0:
      pips = [0, 0]

    street = i // Constants.BET_ACTIONS_PER_STREET
    is_player = (street == 0 and (i % 2) == infoset.player_position) or \
                (street > 0 and (i % 2) != infoset.player_position)
    
    # Detect a raise.
    amt_after_action = pips[i % 2] + infoset.bet_history_vec[i]
    action_is_fold = amt_after_action < pips[1 - (i % 2)]
    if action_is_fold:
      break
    action_is_check = amt_after_action == pips[1 - (i % 2)] and infoset.bet_history_vec[i] == 0 
    action_is_call = amt_after_action == pips[1 - (i % 2)] and infoset.bet_history_vec[i] > 0
    action_is_raise = amt_after_action > pips[1 - (i % 2)]
    if (action_is_raise) and i >= 2:
      if is_player:
        h[plyr_raised_offset + street] = 'R'
      else:
        h[opp_raised_offset + street] = 'R'

      # Raise is defined as a percentage of the called pot.
      if street == infoset.street:
        call_amt = abs(pips[0] - pips[1])
        raise_amt = (infoset.bet_history_vec[i] - call_amt) / (cumul[i-1] + call_amt)
        action_offset = (i - 2) if street == 0 else i % Constants.BET_ACTIONS_PER_STREET

        if action_is_check:
          h[street_actions_offset + action_offset] = 'CK'
        elif action_is_call:
          h[street_actions_offset + action_offset] = 'CL'
        else:
          assert(raise_amt > 0)
          if raise_amt <= 0.5:
            h[street_actions_offset + action_offset] = 'HP'
          elif raise_amt <= 1.0:
            h[street_actions_offset + action_offset] = '1P'
          else:
            h[street_actions_offset + action_offset] = '2P'

    pips[i % 2] += infoset.bet_history_vec[i]

  return h


def bucket_small_join(b):
  return '.'.join(b[:3]) + '|' + '.'.join(b[3:7]) + '|' + '.'.join(b[7:11]) + '|' + '.'.join(b[11:])
