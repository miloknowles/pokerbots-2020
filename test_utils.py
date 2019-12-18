import unittest
from utils import *


class UtilsTest(unittest.TestCase):
  def test_encode_cards_rank_suit(self):
    ranks = [2, 3, 4, 5, 6, 7, 8, 9, "T", "J", "Q", "K", "A"]
    suits = ["c", "d", "h", "s"]

    cards = []
    for rank in ranks:
      for suit in suits:
        cards.append(str(rank) + suit)

    encoded = encode_cards_rank_suit(cards)
    self.assertTrue((encoded == np.arange(52)).all())


if __name__ == "__main__":
  unittest.main()
