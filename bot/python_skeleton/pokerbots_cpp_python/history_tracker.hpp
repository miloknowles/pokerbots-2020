#pragma once

#include <array>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <string>
#include <utility>

namespace pb {

static constexpr int kMaxActionsPerStreet = 4;

typedef std::vector<std::vector<int>> FlexHistory;
typedef std::array<int, 2 + 4*kMaxActionsPerStreet> FixedHistory;


inline int GetStreet0123(const int street_sz) {
  return street_sz == 0 ? 0 : (street_sz - 2);
}


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


/*
Apply a tiny abstraction to an infoset.
- SB or BB (0 or 1)
- Hand strengths are bucketed into { 0-40%, 40-60%, 60-80%, 80-100% }
- Current street has 5 betting actions of { 0, C, P/2, P, 2P }
- Previous streets are summarized by:
  - 0 or 1: whether the player raised
  - 0 or 1: whether the opponent raised

[ SB/BB,
  CURRENT_HS,
  P_RAISED_P, P_RAISED_F, P_RAISED_T, P_RAISED_R,
  O_RAISED_P, O_RAISED_F, O_RAISED_T, O_RAISED_R,
  A0, A1, A2, A3
  CURRENT_STREET
]
*/
std::vector<std::string> BucketInfoSetSmall(const EvInfoSet& infoset);


inline std::string BucketSmallJoin(const std::vector<std::string>& b) {
  const std::string meta = b[0] + "." + b[1] + "." + b[2];
  const std::string plyr = b[3] + "." + b[4] + "." + b[5] + "." + b[6];
  const std::string opp = b[7] + "." + b[8] + "." + b[9] + "." + b[10];
  const std::string street = b[11] + "." + b[12] + "." + b[13] + "." + b[14];
  return meta + "|" + plyr + "|" + opp + "|" + street;
}

}
