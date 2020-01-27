#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <cassert>

#include "engine_modified.hpp"

namespace pb {
namespace cfr {

static constexpr int kMaxActionsPerStreet = 4;

typedef std::vector<std::vector<int>> FlexHistory;
typedef std::array<int, 2 + 4*kMaxActionsPerStreet> FixedHistory;

typedef std::array<std::array<float, 4>, 2> PrecomputedEv;
typedef std::array<Action, 6> ActionVec;
typedef std::array<int, 6> ActionMask;
typedef std::array<double, 6> ActionRegrets;
typedef std::array<std::array<double, 2>, 6> ActionValues;


inline void PrintRegrets(const ActionRegrets& r) {
  for (int i = 0; i < r.size(); ++i) {
    std::cout << r[i] << " ";
  }
  std::cout << std::endl;
}


inline void PrintFlexHistory(const FlexHistory& fh) {
  for (const std::vector<int>& v : fh) {
    for (const int vi : v) {
      std::cout << vi << " ";
    }
    std::cout << "| ";
  }
  std::cout << std::endl;
}


inline int GetStreet0123(const int street_sz) {
  return street_sz == 0 ? 0 : (street_sz - 2);
}

struct EvInfoSet {
  explicit EvInfoSet(float ev, const FixedHistory& h, int player_position, int street) :
      ev(ev), bet_history_vec(h), player_position(player_position), street(street) {
  }

  EvInfoSet() = delete;

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
  assert(b.size() == 15);
  const std::string meta = b[0] + "." + b[1] + "." + b[2];
  const std::string plyr = b[3] + "." + b[4] + "." + b[5] + "." + b[6];
  const std::string opp = b[7] + "." + b[8] + "." + b[9] + "." + b[10];
  const std::string street = b[11] + "." + b[12] + "." + b[13] + "." + b[14];
  const std::string out = meta + "|" + plyr + "|" + opp + "|" + street;
  return out;
}

}
}