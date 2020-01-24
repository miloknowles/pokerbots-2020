#include <iostream>
#include "history_tracker.hpp"

namespace pb {

static inline void PrintVector(const std::vector<int>& vec) {
  for (const int t : vec) {
    std::cout << t << " ";
  }
  std::cout << "\n";
}


void HistoryTracker::Update(int my_contrib, int opp_contrib, int street) {
  const bool did_start_new_street = (prev_street_ != street);

  if (did_start_new_street) {
    // If we don't have a previous street, don't need to do call detection below.
    if (street > 0) {
      std::vector<int>& prev_street_adds = history_.back();
      // PrintVector(prev_street_adds);

      // The amount that was put in the pot by each player during the last street.
      const int prev_street_pip = std::min(my_contrib, opp_contrib) - prev_street_contrib_;
      std::array<int, 2> pips = { 0, 0 };
      for (int i = 0; i < prev_street_adds.size(); ++i) {
        pips[i % 2] += prev_street_adds.at(i);
      }

      const int call_amt_0 = prev_street_pip - pips[0];
      const int call_amt_1 = prev_street_pip - pips[1];
      
      // If both players owe money in the last round, it must have been because I raised and they called.
      if (call_amt_0 > 0 && call_amt_1 > 0) {
        const int next_idx = prev_street_adds.size();
        if (next_idx % 2 == 0) {
          prev_street_adds.emplace_back(call_amt_0);
          prev_street_adds.emplace_back(call_amt_1);
        } else {
          prev_street_adds.emplace_back(call_amt_1);
          prev_street_adds.emplace_back(call_amt_0);
        }
      } else if (call_amt_0 > 0) {
        prev_street_adds.emplace_back(call_amt_0);
      } else if (call_amt_1 > 0) {
        prev_street_adds.emplace_back(call_amt_1);
      }

      // Handle a double check here.
      if (prev_street_adds.size() < 2) {
        while (prev_street_adds.size() < 2) { prev_street_adds.emplace_back(0); }
      }

      // The next action idx should be the first of the new street.
      next_action_idx_ = kMaxActionsPerStreet * GetStreet0123(street);

      contributions_[0] = std::min(my_contrib, opp_contrib);
      contributions_[1] = std::min(my_contrib, opp_contrib);
      prev_street_contrib_ = std::min(my_contrib, opp_contrib);

      // IMPORTANT: Make sure this happens AFTER call detection.
      history_.emplace_back(std::vector<int>());
    }

    prev_street_ = street;
  }

  const bool we_go_first_this_street = ((street == 0) && !is_big_blind_) || (street > 0 && is_big_blind_);

  // If the opponent goes first and this is the first update we're doing for this street,
  // then ONLY an opponent action has been performed.
  if (!we_go_first_this_street && did_start_new_street) {
    UpdateOpponent(opp_contrib,  street);
  } else if (we_go_first_this_street && did_start_new_street) {
    return;
  } else {
    UpdatePlayer(my_contrib, street);
    UpdateOpponent(opp_contrib, street);
  }
}


void HistoryTracker::UpdatePlayer(int my_contrib, int street) {
  const int add_amt = my_contrib - contributions_[0];
  if (add_amt > 0 || history_.back().size() < 2) {
    history_.back().emplace_back(add_amt);
    contributions_[0] = my_contrib;
  }
}


void HistoryTracker::UpdateOpponent(int opp_contrib, int street) {
  const int add_amt = opp_contrib - contributions_[1];
  if (add_amt > 0 || history_.back().size() < 2) {
    history_.back().emplace_back(opp_contrib - contributions_[1]);
    contributions_[1] = opp_contrib;
  }
}


std::vector<std::string> BucketInfoSetSmall(const EvInfoSet& infoset) {
  std::vector<std::string> h(1 + 1 + 1 + 4 + 4 + 4);
  std::fill(h.begin(), h.end(), "x");

  h[0] = infoset.player_position == 0 ? "SB" : "BB";
  if (infoset.street == 0) {
    h[1] = "P";
  } else if (infoset.street == 1) {
    h[1] = "F";
  } else if (infoset.street == 2) {
    h[1] = "T"; 
  } else {
    h[1] = "R";
  }

  if (infoset.ev < 0.4) {
    h[2] = "H0";
  } else if (infoset.ev < 0.6) {
    h[2] = "H1";
  } else if (infoset.ev < 0.8) {
    h[2] = "H2";
  } else {
    h[2] = "H3";
  }

  assert(infoset.bet_history_vec.size() == (2 + 4*kMaxActionsPerStreet));
  // PrintVector(std::vector<int>(infoset.bet_history_vec.begin(), infoset.bet_history_vec.end()));

  std::array<int, 2> pips = { 0, 0 };
  const int plyr_raised_offset = 3;
  const int opp_raised_offset = 7;
  const int street_actions_offset = 11;

  std::vector<int> cumul = { infoset.bet_history_vec.at(0) };
  for (int i = 1; i < infoset.bet_history_vec.size(); ++i) {
    cumul.emplace_back(cumul.at(i-1) + infoset.bet_history_vec.at(i));
  }

  for (int i = 0; i < (2 + 4*kMaxActionsPerStreet); ++i) {
    const bool is_new_street = ((i == 0) || ((i - 2) % kMaxActionsPerStreet) == 0) && i > 2;

    if (is_new_street) {
      pips = { 0, 0 };
    }

    const int street = i > 2 ? (i - 2) / kMaxActionsPerStreet : 0;
    if (street > infoset.street) {
      break;
    }

    const bool is_player = (street == 0 && ((i % 2) == infoset.player_position)) ||
                           (street > 0 && ((i % 2) != infoset.player_position));
    
    const int amt_after_action = pips[i % 2] + infoset.bet_history_vec.at(i);
    const bool action_is_fold = (amt_after_action < pips[1 - (i % 2)]) && (infoset.bet_history_vec[i] == 0);
    const bool action_is_wrapped_raise = (amt_after_action < pips[1 - (i % 2)]) && (infoset.bet_history_vec[i] > 0);

    if (action_is_fold) {
      break;
    }

    const bool action_is_check = (amt_after_action == pips[1 - (i % 2)]) && (infoset.bet_history_vec[i] == 0); 
    const bool action_is_call = (amt_after_action == pips[1 - (i % 2)]) && (infoset.bet_history_vec[i] > 0);
    const bool action_is_raise = (amt_after_action > pips[1 - (i % 2)]);

    if (action_is_raise && (i >= 2)) {
      if (is_player) {
        h[plyr_raised_offset + street] = "R";
      } else {
        h[opp_raised_offset + street] = "R";
      }
    }

    if (street == infoset.street && (i >= 2)) {
      const int call_amt = std::abs(pips[0] - pips[1]);
      const float raise_amt = static_cast<float>(infoset.bet_history_vec[i] - call_amt) / static_cast<float>(cumul[i-1] + call_amt);
      const int action_offset = (street == 0) ? (i - 2) : ((i - 2) % kMaxActionsPerStreet);

      if (action_is_check) {
        const bool bet_occurs_after = (i < (infoset.bet_history_vec.size() - 1)) && (infoset.bet_history_vec[i+1] > 0);
        if (action_offset == 0 && !(is_player || bet_occurs_after)) {
          h[street_actions_offset + action_offset] = "CK";
        } else {
          break;
        }
      } else if (action_is_call) {
        h[street_actions_offset + action_offset] = "CL";
      } else if (action_is_wrapped_raise) {
        h[street_actions_offset + action_offset] = "?P";
      } else {
        assert(raise_amt > 0);
        if (raise_amt <= 0.75) {
          h[street_actions_offset + action_offset] = "HP";
        } else if (raise_amt <= 1.5) {
          h[street_actions_offset + action_offset] = "1P";
        } else {
          h[street_actions_offset + action_offset] = "2P";
        }
      }
    }

    pips[i % 2] += infoset.bet_history_vec.at(i);
  }

  return h;
}

}
