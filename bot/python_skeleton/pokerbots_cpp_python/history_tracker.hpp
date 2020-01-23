#pragma once

#include <array>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <utility>

namespace pb {

static constexpr int kMaxActionsPerStreet = 4;

typedef std::vector<std::vector<int>> FlexHistory;
typedef std::array<int, 2 + 4*kMaxActionsPerStreet> FixedHistory;

inline int GetStreet0123(const int street_sz) {
  return street_sz == 0 ? 0 : (street_sz - 2);
}

struct BettingInfo {
  BettingInfo() = default;
  int num_bets = 0;
  int num_calls = 0;
  int num_raises = 0;
};

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

  // // Total number of actions taken by each player to increase the pot size during a street.
  // std::pair<int, int> TotalBets(int street) const {
  //   const auto& info = GetBettingInfo(street);
  //   const int ply_bets = info.first.num_raises + info.first.num_bets;
  //   const int opp_bets = info.second.num_raises + info.second.num_bets;
  //   return std::make_pair(ply_bets, opp_bets);
  // }

  // // Was the first action of a street a CHECK?
  // bool FirstActionWasCheck(int street) const {
  //   const int offset = kMaxActionsPerStreet * GetStreet0123(street);
  //   return history_.at(offset) == 0;
  // }

  // std::pair<BettingInfo, BettingInfo> GetBettingInfo(int street) const;

  void Print() const {
    for (const std::vector<int>& st : history_) {
      for (const int add_amt : st) {
        std::cout << add_amt << " ";
      }
      std::cout << "| ";
    }
    std::cout << std::endl;
  }

  // std::vector<int> Vector() const { return std::vector<int>(history_.begin(), history_.end()); }
  FlexHistory History() const { return history_; }

 private:
  int prev_street_ = -1;
  bool is_big_blind_;

  // 0 is us, 1 is opp.
  std::array<int, 2> contributions_{};
  int prev_street_contrib_ = 0;

  // Encodes the actions taken so far (bets as % of pot).
  int next_action_idx_ = 2;
  // std::array<int, 4*kMaxActionsPerStreet> history_{};
  FlexHistory history_{};
};


struct EvInfoSet {
  explicit EvInfoSet(float ev, const FixedHistory& h, int player_position, int street) :
      ev(ev), bet_history_vec(h), player_position(player_position), street(street) {
  }

  void Print() const {
    printf("EvInfoSet | ev=%f | player_position=%d | street=%d\n", ev, player_position, street);
    for (const int add_amt : bet_history_vec) {
      std::cout << add_amt << " ";
    }
    std::cout << std::endl;
  }

  float ev;
  int player_position;          // 0 if player is SB, 1 if BB.
  FixedHistory bet_history_vec;
  int street;                   // 0, 1, 2, 3.
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
