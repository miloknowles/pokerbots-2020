#include <cassert>

#include "infoset.hpp"

namespace pb {
namespace cfr {

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

      // printf("on our street, i=%d, check=%d, call=%d\n", i, action_is_check, action_is_call);

      if (action_is_check) {
        const bool bet_occurs_after = (i < (infoset.bet_history_vec.size() - 1)) && (infoset.bet_history_vec[i+1] > 0);
        // printf("offset=%d is_player=%d bet_occurs_after=%d\n", action_offset, is_player, bet_occurs_after);
        if (action_offset == 0 && (!is_player || bet_occurs_after)) {
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
}
