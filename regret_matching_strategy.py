import numpy as np
from constants import Constants


class RegretMatchingStrategy(object):
  def __init__(self):
    pass

  def get_action_probabilities(self, infoset):
    return np.ones(len(Constants.ALL_ACTIONS)) / len(Constants.ALL_ACTIONS)

