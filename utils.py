from collections import OrderedDict

import torch
import numpy as np


RANK_TO_VALUE = OrderedDict({
  "2": 0,
  "3": 1,
  "4": 2,
  "5": 3,
  "6": 4,
  "7": 5,
  "8": 6,
  "9": 7,
  "t": 8,
  "j": 9,
  "q": 10,
  "k": 11,
  "a": 12
})


SUIT_TO_VALUE = OrderedDict({
  "c": 0,
  "d": 1,
  "h": 2,
  "s": 3
})


def encode_cards(card_iterable, rank_index=0):
  """
  Encode card strings into their rank-major numerical order.
  i.e 2c 2d 2h 2s ... Ac Ad Ah As
  """
  out = torch.zeros(len(card_iterable))
  for i, c in enumerate(card_iterable):
    lower = c.lower()
    rank, suit = lower[rank_index], lower[(rank_index + 1) % 2]
    out[i] = 4 * RANK_TO_VALUE[rank] + SUIT_TO_VALUE[suit]
  return out


def encode_cards_rank_suit(card_iterable):
  return encode_cards(card_iterable, rank_index=0)


def encode_cards_suit_rank(card_iterable):
  return encode_cards(card_iterable, rank_index=1)


def apply_mask_and_normalize(probs, mask):
  p = probs * mask

  # If all nonzero probability actions are masked out, choose a random one.
  if p.sum() == 0:
    p[np.random.randint(0, high=len(p))] = 1.0
  p = p / torch.sum(p)

  return p


def apply_mask_and_uniform(probs, mask):
  p = probs * mask

  if p.sum() == 0:
    p = mask
  p = p / torch.sum(p)

  return p


def sample_uniform_action(valid_actions):
  item = valid_actions[np.random.randint(len(valid_actions))]
  amount = item["amount"]

  if type(amount) == dict:
    random_amount = np.random.randint(amount["min"], high=amount["max"]+1)
    return item["action"], random_amount
  else:
    return item["action"], item["amount"]
