#pragma once

#include <array>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <utility>

namespace pb {

static constexpr int kMaxActionsPerStreet = 8;

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
    history_[0] = 1;
    history_[1] = 2;
  }

  void Update(int my_contrib, int opp_contrib, int street);

  void UpdatePlayer(int my_contrib, int street);
  void UpdateOpponent(int opp_contrib, int street);

  // Total number of actions taken by each player to increase the pot size during a street.
  std::pair<int, int> TotalBets(int street) const {
    const auto& info = GetBettingInfo(street);
    const int ply_bets = info.first.num_raises + info.first.num_bets;
    const int opp_bets = info.second.num_raises + info.second.num_bets;
    return std::make_pair(ply_bets, opp_bets);
  }

  // Was the first action of a street a CHECK?
  bool FirstActionWasCheck(int street) const {
    const int offset = kMaxActionsPerStreet * GetStreet0123(street);
    return history_.at(offset) == 0;
  }

  std::pair<BettingInfo, BettingInfo> GetBettingInfo(int street) const;

  void Print() const {
    const int actions_per = history_.size() / 4;
    for (int st = 0; st < 4; ++st) {
      for (int i = 0; i < actions_per; ++i) {
        std::cout << history_.at(st*actions_per + i) << " ";
      }
      std::cout << "| ";
    }
    std::cout << std::endl;
  }

  std::vector<int> Vector() const { return std::vector<int>(history_.begin(), history_.end()); }

 private:
  int prev_street_ = -1;
  bool is_big_blind_;

  // 0 is us, 1 is opp.
  std::array<int, 2> contributions_;
  int prev_street_contrib_ = 0;

  // Encodes the actions taken so far (bets as % of pot).
  int next_action_idx_ = 2;
  std::array<int, 4*kMaxActionsPerStreet> history_{};
};

}
