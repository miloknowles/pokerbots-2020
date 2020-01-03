import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedding(nn.Module):
  def __init__(self, embed_dim):
    """
    embed_dim (int) : The dimension of the embedding for cards (64 in the paper).
    """
    super(CardEmbedding, self).__init__()
    self.rank = nn.Embedding(13, embed_dim)
    self.suit = nn.Embedding(4, embed_dim)
    self.card = nn.Embedding(52, embed_dim)
  
  def forward(self, x):
    """
    Returns the sum of embeddings for cards in input.

    input (torch.Tensor) : Shape (batch_size, num_cards).
    
    Returns:
      (torch.Tensor) : Shape (batch_size, embed_dim).
    """
    B, num_cards = x.shape

    valid = x.ge(0).float() # -1 means "no card".
    x = x.clamp(min=0)

    # Cards are ordered from 0 to 51, in suit-major order.
    # For example: Ac Ad Ah As 2c 2d 2h 2s ...
    # This means that the rank increases every 4 indices, and the suit is given
    # by the index mod 4.
    embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
    embs = embs * valid.unsqueeze(-1) # Mask out the "no card" embeddings.

    # Sum across the cards in the hole/board set input.
    return embs.view(B, num_cards, -1).sum(1)


class DeepCFRModel(nn.Module):
  def __init__(self, ncardtypes, nbets, nactions, embed_dim=256):
    """
    ncardtypes (int) : The number of streets that have been played (i.e 1 for preflop, 2 flop, 3 turn, 4 river).
    nbets (int) : The number of betting actions in a game (num_streets * num_bets_per_street * num_players).
    nactions (int) : The number of output actions that the network can take.
    embed_dim (int) : Dimension of the card embeddings (i.e 64).
    """
    super(DeepCFRModel, self).__init__()

    # Create an embedding layer for each set of cards we would see during a game.
    # i.e hole cards, flop, turn, river.
    self.card_embeddings = nn.ModuleList(
      [CardEmbedding(embed_dim) for _ in range(ncardtypes)])
    
    self.card1 = nn.Linear(embed_dim * ncardtypes, embed_dim)
    self.card2 = nn.Linear(embed_dim, embed_dim)
    self.card3 = nn.Linear(embed_dim, embed_dim)

    # Has size nbets * 2 because we pass in the bet amounts AND a mask of the same size.
    self.bet1 = nn.Linear(nbets * 2, embed_dim)
    self.bet2 = nn.Linear(embed_dim, embed_dim)

    self.comb1 = nn.Linear(2 * embed_dim, embed_dim)
    self.comb2 = nn.Linear(embed_dim, embed_dim)
    self.comb3 = nn.Linear(embed_dim, embed_dim)

    self.normalize = nn.BatchNorm1d(embed_dim)
    self.action_head = nn.Linear(embed_dim, nactions)

    # NOTE(milo): Initialize so that all actions have equal output.
    # NOTE(milo): I think this could be problematic for training, so leaving it out. Better to just
    # sample actions uniformly on the first training iteration.
    # torch.nn.init.zeros_(self.action_head.weight)
    # torch.nn.init.ones_(self.action_head.bias)

  def forward(self, hole_cards, board_cards, bets):
    """
    hole_cards (torch.Tensor) : Shape (B x 2)
    board_cards (torch.Tensor) : Shape (B x 5) where NA cards have value -1.
    bets (torch.Tensor) : Shape (batch_size, nbets).

    Returns:
      (torch.Tensor) : Shape (batch_size, nactions).
    """
    # STEP 1: Embed the hole, flop, and optionally turn and river cards.
    card_embs = [self.card_embeddings[0](hole_cards)]

    flop = board_cards[:, :3]
    turn = board_cards[:, 3].unsqueeze(-1).contiguous()
    river = board_cards[:, 4].unsqueeze(-1).contiguous()

    card_embs.append(self.card_embeddings[1](flop))
    card_embs.append(self.card_embeddings[2](turn))
    card_embs.append(self.card_embeddings[3](river))

    card_embs = torch.cat(card_embs, dim=1)

    x = F.relu(self.card1(card_embs))
    x = F.relu(self.card2(x))
    x = F.relu(self.card3(x))

    # STEP 2: Process betting history.
    bet_size = bets.clamp(-1e6, 1e6)

    # NOTE(milo): Changed this to a not-equal, since we encode opponent bets with negative numbers.
    bet_occurred = bets.ne(0)
    bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)

    y = F.relu(self.bet1(bet_feats))
    y = F.relu(self.bet2(y) + y)

    # STEP 3: Combine features and predict action advantage.
    z = torch.cat([x, y], dim=1)
    z = F.relu(self.comb1(z))
    z = F.relu(self.comb2(z) + z)
    z = F.relu(self.comb3(z) + z)

    # Normalized to have zero mean and stdev 1.
    z = (z - z.mean()) / torch.std(z)

    return self.action_head(z)
