#include <array>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <utility>

namespace pb {

static constexpr int kMaxActionsPerStreet = 8;

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

  // TODO: fix
  int NumBettingRounds() const { return 2; }

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
