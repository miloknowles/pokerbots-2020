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

}
