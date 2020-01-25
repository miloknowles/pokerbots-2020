#include "cfr.hpp"

#include <chrono>

namespace pb {


std::pair<ActionVec, ActionMask> MakeActions(RoundState* round_state, int active) {
  const int legal_actions = round_state->legal_actions();

  const int my_pip = round_state->pips[active];
  const int opp_pip = round_state->pips[1-active];

  const int min_raise = round_state->raise_bounds()[0];
  const int max_raise = round_state->raise_bounds()[1];

  const int my_stack = round_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = round_state->stacks[1-active];  // the number of chips your opponent has remaining

  const int pot_size = 2 * 200 - my_stack - opp_stack;

  const int bet_actions_so_far = round_state->bet_history.back().size();
  const int bet_actions_this_street = (round_state->street > 0) ? kMaxActionsPerStreet : (kMaxActionsPerStreet + 2);
  const bool force_fold_call = bet_actions_so_far >= (bet_actions_this_street - 1);

  ActionMask actions_mask;
  std::fill(actions_mask.begin(), actions_mask.end(), 0);

  // NOTE: These are HALF POT multiples.
  ActionVec actions_unscaled = {
    FoldAction(),
    CallAction(),
    CheckAction(),
    RaiseAction(1),
    RaiseAction(2),
    RaiseAction(4)
  };

  for (int i = 0; i < actions_unscaled.size(); ++i) {
    const Action& a = actions_unscaled.at(i);

    const bool action_is_allowed = a.action_type & legal_actions;
    if (action_is_allowed && !(a.action_type == RAISE_ACTION_TYPE && force_fold_call)) {
      actions_mask.at(i) = 1;
    }

    if (a.action_type == RAISE_ACTION_TYPE) {
      const int pot_size_after_call = pot_size + std::abs(my_pip - opp_pip);
      const float amt_to_add = static_cast<float>(pot_size_after_call) * static_cast<float>(a.amount) / 2.0f;
      const int amt_to_raise = std::max(my_pip, opp_pip) + static_cast<int>(amt_to_add);
      const int amt = std::min(max_raise, std::max(min_raise, amt_to_raise));
      actions_unscaled.at(i) = RaiseAction(amt);
    }
  }

  assert(actions_unscaled.size() == actions_mask.size());
  return std::make_pair(actions_unscaled, actions_mask);
}


EvInfoSet MakeInfoSet(const RoundState* round_state, int active_plyr_idx, bool player_is_sb,
                      PrecomputedEv precomputed_ev) {
  FixedHistory fh;
  std::fill(fh.begin(), fh.end(), 0);

  FlexHistory history = round_state->bet_history;
  for (int street = 0; street < history.size(); ++street) {
    const std::vector<int>& actions_this_street = history.at(street);
    const int offset = street * kMaxActionsPerStreet + (street > 0 ? 2 : 0);

    for (int i = 0; i < actions_this_street.size(); ++i) {
      const int max_this_street = (street > 0) ? kMaxActionsPerStreet : (kMaxActionsPerStreet + 2);
      const int wrap = std::min(i, max_this_street - 2 + i % 2);
      fh.at(offset + wrap) += actions_this_street.at(i);
    }
  }
  const int street_0123 = GetStreet0123(round_state->street);
  const float ev = precomputed_ev.at(active_plyr_idx).at(street_0123);

  return EvInfoSet(ev, fh, player_is_sb ? 0 : 1, street_0123);
}


PrecomputedEv MakePrecomputedEv(const RoundState& round_state) {
  PrecomputedEv out; // 2x5

  const std::string h1 = round_state.hands[0][0] + round_state.hands[0][1];
  const std::string h2 = round_state.hands[1][0] + round_state.hands[1][1];

  for (int s = 0; s < 4; ++s) {
    int iters = 1;
    if (s == 1) {
      iters = 10000;
    } else if (s == 2) {
      iters = 10000;
    } else if (s == 3) {
      iters = 1326;
    }

    const std::string board = round_state.deck[0] + round_state.deck[1] + round_state.deck[2] + round_state.deck[3] + round_state.deck[4];

    const float ev1 = PbotsCalcEquity(h1 + ":xx", board, "", iters);
    const float ev2 = PbotsCalcEquity(h2 + ":xx", board, "", iters);
    out[0][s] = ev1;
    out[1][s] = ev2;
  }
}


RoundState CreateNewRound(int sb_plyr_idx) {
  std::vector<int> deck;
  for (int i = 0; i < 52; ++i) { deck.emplace_back(i); }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(deck.begin(), deck.end(), std::default_random_engine(seed));

  std::array<std::array<std::string, 2>, 2> hands;
  std::array<std::string, 5> board;

  hands[0][0] = ConvertCodeToCard(deck[0]);
  hands[0][1] = ConvertCodeToCard(deck[1]);

  hands[1][0] = ConvertCodeToCard(deck[2]);
  hands[1][1] = ConvertCodeToCard(deck[3]);

  board[0] = ConvertCodeToCard(deck[4]);
  board[1] = ConvertCodeToCard(deck[5]);
  board[2] = ConvertCodeToCard(deck[6]);
  board[3] = ConvertCodeToCard(deck[7]);
  board[4] = ConvertCodeToCard(deck[8]);

  std::array<int, 2> pips = { 1, 2 };
  std::array<int, 2> stacks = { 199, 198 };

  if (sb_plyr_idx == 1) {
    pips = { pips[1], pips[0] };
    stacks = { stacks[1], stacks[0] };
  }

  return RoundState(sb_plyr_idx, 0, pips, stacks, hands, board, nullptr, {{1, 2}}, sb_plyr_idx);
}


ActionRegrets ApplyMaskAndUniform(const ActionRegrets& p, const ActionMask& mask) {
  double denom = 0;
  int valid = 0;
  ActionRegrets out;
  for (int i = 0; i < p.size(); ++i) {
    const double masked = p[i] * static_cast<double>(mask[i]);
    denom += masked;
    out[i] = masked;
    valid += mask[i];
  }

  if (denom <= 1e-3) {
    for (int i = 0; i < p.size(); ++i) {
      out[i] = static_cast<double>(mask[i]) / static_cast<double>(valid);
    }
  } else {
    for (int i = 0; i < p.size(); ++i) {
      out[i] /= denom;
    }
  }

  return out;
}


static std::array<double, 2> ComputeEv(const ActionValues& values, const std::array<double, 6>& probs) {
  std::array<double, 2> ev = { 0, 0 };

  for (int i = 0; i < values.size(); ++i) {
    ev[0] += values[i][0] * probs[i];
    ev[1] += values[i][1] * probs[i];
  }

  return ev;
}

static std::array<double, 6> Multiply(const std::array<double, 6>& v1, const std::array<double, 6>& v2) {
  std::array<double, 6> out = v1;
  for (int i = 0; i < 6; ++i) {
    out[i] *= v2[i];
  }
  return out;
}

static std::array<double, 6> Multiply(const std::array<double, 6>& v1, const double v2) {
  std::array<double, 6> out = v1;
  for (int i = 0; i < 6; ++i) {
    out[i] *= v2;
  }
  return out;
}

static void FillZeros(ActionValues& values) {
  for (int i = 0; i < values.size(); ++i) {
    std::fill(values[i].begin(), values[i].end(), 0);
  }
}


NodeInfo TraverseCfr(State* state,
                     int traverse_plyr,
                     int sb_plyr_idx,
                     std::array<RegretMatchedStrategy, 2>& regrets,
                     std::array<RegretMatchedStrategy, 2>& strategies,
                     int t,
                     const std::array<double, 2>& reach_probabilities,
                     const PrecomputedEv& precomputed_ev,
                     int* rctr,
                     bool allow_updates,
                     bool do_external_sampling,
                     bool skip_unreachable_actions) {

  NodeInfo node_info;

  //========================== TERMINAL STATE =============================
  if (typeid(state).name() == "TerminalState") {
    TerminalState* terminal_state = dynamic_cast<TerminalState*>(state);
    node_info.strategy_ev = { static_cast<float>(terminal_state->deltas[0]),
                              static_cast<float>(terminal_state->deltas[1]) };
    node_info.best_response_ev = node_info.strategy_ev;
    return node_info;
  }

  RoundState* round_state = dynamic_cast<RoundState*>(state);

  const int active_plyr_idx = round_state->button % 2;
  const int inactive_plyr_idx = 1 - active_plyr_idx;

  const auto& actions_and_mask = MakeActions(round_state, active_plyr_idx);
  const ActionVec actions = actions_and_mask.first;
  const ActionMask mask = actions_and_mask.second;

  const EvInfoSet& infoset = MakeInfoSet(
      round_state, active_plyr_idx, active_plyr_idx == sb_plyr_idx, precomputed_ev); 

  std::array<double, 6> action_probs = regrets[active_plyr_idx].GetStrategy(infoset);
  action_probs = ApplyMaskAndUniform(action_probs, mask);

  ActionValues action_values;
  ActionValues br_values;
  FillZeros(action_values);
  FillZeros(br_values);

  //========================= PLAYER ACTION =============================
  for (int i = 0; i < actions.size(); ++i) {
    if (mask[i] <= 0 || (skip_unreachable_actions && action_probs[i] <= 0)) {
      continue;
    }

    assert(mask[i] > 0);

    State* next_round_state = round_state->proceed(actions[i]);
    std::array<double, 2> next_reach_prob = reach_probabilities;
    next_reach_prob[active_plyr_idx] *= action_probs[i];

    const NodeInfo child_node_info = TraverseCfr(
      next_round_state, traverse_plyr, sb_plyr_idx, regrets, strategies, t,
      next_reach_prob, precomputed_ev, rctr, allow_updates,
      do_external_sampling, skip_unreachable_actions);
    
    action_values[i] = child_node_info.strategy_ev;
    br_values[i] = child_node_info.best_response_ev;
  }

  node_info.strategy_ev = ComputeEv(action_values, action_probs);
  ActionRegrets immediate_regrets = { 0, 0, 0, 0, 0, 0 };
  for (int i = 0; i < immediate_regrets.size(); ++i) {
    if (mask[i]) {
      immediate_regrets[i] = action_values[i][active_plyr_idx] - node_info.strategy_ev[active_plyr_idx];
    }
  }

  double br_active = std::numeric_limits<double>::min();
  for (int i = 0; i < br_values.size(); ++i) {
    if (mask[i]) { br_active = std::fmax(br_active, br_values[i][active_plyr_idx]); }
  }
  node_info.best_response_ev[active_plyr_idx] = br_active;
  node_info.best_response_ev[inactive_plyr_idx] = ComputeEv(br_values, action_probs)[inactive_plyr_idx];
  node_info.exploitability = { node_info.best_response_ev[0] - node_info.strategy_ev[0],
                               node_info.best_response_ev[1] - node_info.strategy_ev[1] };

  if (allow_updates && active_plyr_idx == traverse_plyr) {
    const double counterfactual = reach_probabilities[inactive_plyr_idx];
    strategies[active_plyr_idx].AddRegret(infoset, Multiply(action_probs, counterfactual));
    regrets[active_plyr_idx].AddRegret(infoset, Multiply(immediate_regrets, counterfactual));
  }

  return node_info;
}

}
