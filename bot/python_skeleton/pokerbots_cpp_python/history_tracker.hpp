#pragma once

#include <array>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <string>
#include <utility>

#include "infoset.hpp"

namespace pb {

class HistoryTracker {
 public:
  HistoryTracker(bool is_big_blind) : is_big_blind_(is_big_blind) {
    contributions_[0] = is_big_blind ? 2 : 1;
    contributions_[1] = is_big_blind ? 1 : 2;

    // History starts out with an empty preflop bet vector.
    assert(history_.size() == 0);
    const std::vector<int> blinds = { 1, 2 };
    history_.emplace_back(blinds);
  }

  void Update(int my_contrib, int opp_contrib, int street);
  void UpdatePlayer(int my_contrib, int street);
  void UpdateOpponent(int opp_contrib, int street);

  void Print() const {
    for (const std::vector<int>& st : history_) {
      for (const int add_amt : st) {
        std::cout << add_amt << " ";
      }
      std::cout << "| ";
    }
    std::cout << std::endl;
  }

  FlexHistory History() const { return history_; }

 private:
  int prev_street_ = -1;
  bool is_big_blind_;

  // 0 is us, 1 is opp.
  std::array<int, 2> contributions_{};
  int prev_street_contrib_ = 0;

  // Encodes the actions taken so far (bets as % of pot).
  int next_action_idx_ = 2;
  FlexHistory history_{};
};

inline EvInfoSet MakeInfoSet(const HistoryTracker& ht, int player_idx, bool player_is_sb, float ev, int current_street) {
  FixedHistory fh;
  std::fill(fh.begin(), fh.end(), 0);

  FlexHistory history = ht.History();
  for (int street = 0; street < history.size(); ++street) {
    const std::vector<int>& actions_this_street = history.at(street);
    const int offset = street * kMaxActionsPerStreet + (street > 0 ? 2 : 0);

    for (int i = 0; i < actions_this_street.size(); ++i) {
      const int max_this_street = (street > 0) ? kMaxActionsPerStreet : (kMaxActionsPerStreet + 2);
      const int wrap = std::min(i, max_this_street - 2 + i % 2);
      fh.at(offset + wrap) += actions_this_street.at(i);
    }
  }

  return EvInfoSet(ev, fh, player_is_sb ? 0 : 1, GetStreet0123(current_street));
}

}
