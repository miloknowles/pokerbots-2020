#include <iostream>
#include "history_tracker.hpp"

namespace pb {

void HistoryTracker::Update(int my_contrib, int opp_contrib, int street) {
  const bool did_start_new_street = (prev_street_ != street);

  if (did_start_new_street) {
    prev_street_ = street;

    // If we don't have a previous street, don't need to do call detection below.
    if (street > 0) {
      // Make sure the previous street has equal pot contributions.
      const int this_street_off = kMaxActionsPerStreet * GetStreet0123(street);
      const int prev_street_off = this_street_off - kMaxActionsPerStreet;

      // The amount that was put in the pot by each player during the last street.
      const int prev_street_pip = std::min(my_contrib, opp_contrib) - prev_street_contrib_;
      std::array<int, 2> pips = { 0, 0 };

      // We're looking for the call that ended the previous round of betting.
      for (int i = prev_street_off; i < this_street_off; ++i) {
        const int add_amt = history_.at(i);
        const int remaining_amt = (prev_street_pip - pips[i % 2]);
        if (add_amt == 0 && remaining_amt > 0 && i > prev_street_off) {
          history_.at(i) = remaining_amt;
        }
        pips.at(i % 2) += history_.at(i);
        if (pips[0] == prev_street_pip && pips[1] == prev_street_pip) {
          break;
        }
      }

      // The next action idx should be the first of the new street.
      next_action_idx_ = kMaxActionsPerStreet * GetStreet0123(street);

      contributions_[0] = std::min(my_contrib, opp_contrib);
      contributions_[1] = std::min(my_contrib, opp_contrib);
      prev_street_contrib_ = std::min(my_contrib, opp_contrib);
    }
  }
  const bool we_go_first_this_street = ((street == 0) && !is_big_blind_) || (street > 0 && is_big_blind_);
  // const bool first_of_street = ((next_action_idx_ % kMaxActionsPerStreet) == 0) ||
  //                              (street == 0 && next_action_idx_ == 2);
  bool first_of_street = did_start_new_street;
  
  // printf("idx=%d | ME=%d OPP=%d\n", next_action_idx_, my_contrib, opp_contrib);

  // If the opponent goes first and this is the first update we're doing for this street,
  // then ONLY an opponent action has been performed.
  if (!we_go_first_this_street && first_of_street) {
    UpdateOpponent(opp_contrib,  street);
  } else {
    // If we we go first and its the first action, then no updates yet.
    if (!first_of_street) {
      UpdatePlayer(my_contrib, street);
      UpdateOpponent(opp_contrib, street);
    }
  }
}

void HistoryTracker::UpdatePlayer(int my_contrib, int street) {
  history_.at(next_action_idx_) = (my_contrib - contributions_[0]);
  contributions_[0] = my_contrib;

  // If we surpass kMaxActionsPerStreet, keeping adding contributions to the LAST betting action.
  ++next_action_idx_;
  const int parity = next_action_idx_ % 2;
  const int max_idx_this_street = kMaxActionsPerStreet * (GetStreet0123(street) + 1);
  next_action_idx_ = std::min(next_action_idx_, max_idx_this_street - 2 + parity);
}

void HistoryTracker::UpdateOpponent(int opp_contrib, int street) {
  const int add_amt = (opp_contrib - contributions_[1]);
  history_.at(next_action_idx_) = add_amt;
  contributions_[1] = opp_contrib;
  
  // If we surpass kMaxActionsPerStreet, keeping adding contributions to the LAST betting action.
  ++next_action_idx_;
  const int parity = next_action_idx_ % 2;
  const int max_idx_this_street = kMaxActionsPerStreet * (GetStreet0123(street) + 1);
  next_action_idx_ = std::min(next_action_idx_, max_idx_this_street - 2 + parity);
}

std::pair<BettingInfo, BettingInfo> HistoryTracker::GetBettingInfo(int street) const {
  BettingInfo ply;
  BettingInfo opp;
  const int player_parity = (street == 0) ? is_big_blind_ : !is_big_blind_;
  const int start_idx = kMaxActionsPerStreet * GetStreet0123(street);
  const int end_idx = start_idx + kMaxActionsPerStreet;

  std::array<int, 2> pips = { 0, 0 };
  for (int i = start_idx; i < end_idx; ++i) {
    const int add_amt = history_.at(i);
    if ((i % 2) == player_parity) {
      if (i >= 2) {
        // Either a bet or a raise.
        if ((pips[0] + add_amt) > pips[1]) {
          if (pips[1] == pips[0]) {
            ply.num_bets += 1;
          } else {
            ply.num_raises += 1;
          }
        // Must be a call.
        } else if (add_amt > 0) {
          ply.num_calls += 1;
        }
      }
      pips[0] += add_amt;
    } else {
      if (i >= 2) {
        // Either a bet or a raise.
        if ((pips[1] + add_amt) > pips[0]) {
          if (pips[0] == pips[1]) {
            opp.num_bets += 1;
          } else {
            opp.num_raises += 1;
          }
        // Must be a call.
        } else if (add_amt > 0) {
          opp.num_calls += 1;
        } 
      }
      pips[1] += add_amt;
    }
  }

  return std::make_pair(ply, opp);
}

}
