#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "cfr_player.hpp"

namespace pb {


static bool CanCheckFoldRemainder(const int delta, const int round_num) {
  return delta > (1.5f * (1000.0f - static_cast<float>(round_num)) + 1);
}


static int MakeRelativeBet(const float frac, const int pot_size, const int min_raise, const int max_raise) {
  const int amt = static_cast<int>(frac * static_cast<float>(pot_size));
  const int clamped = std::min(max_raise, std::max(min_raise, amt));
  return clamped;
}


std::pair<ActionVec, ActionMask> MakeActions(RoundState* round_state, int active, const HistoryTracker& tracker) {
  const int legal_actions = round_state->legal_actions();

  const int my_pip = round_state->pips[active];
  const int opp_pip = round_state->pips[1-active];

  const int min_raise = round_state->raise_bounds()[0];
  const int max_raise = round_state->raise_bounds()[1];

  const int my_stack = round_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = round_state->stacks[1-active];  // the number of chips your opponent has remaining

  const int pot_size = 2 * 200 - my_stack - opp_stack;

  const int bet_actions_this_street = tracker.History().back().size();
  const bool force_fold_call = bet_actions_this_street >= (kMaxActionsPerStreet - 1);

  ActionMask actions_mask;
  std::fill(actions_mask.begin(), actions_mask.end(), 0);

  // NOTE: These are HALF POT multiples.
  ActionVec actions_unscaled = {
    FoldAction(),
    CallAction(),
    CheckAction(),
    RaiseAction(1),
    RaiseAction(2),
    RaiseAction(4)
  };

  for (int i = 0; i < actions_unscaled.size(); ++i) {
    const Action& a = actions_unscaled.at(i);

    const bool action_is_allowed = a.action_type & legal_actions;
    if (action_is_allowed && !(a.action_type == RAISE_ACTION_TYPE && force_fold_call)) {
      actions_mask.at(i) = 1;
    }

    if (a.action_type == RAISE_ACTION_TYPE) {
      const int pot_size_after_call = pot_size + std::abs(my_pip - opp_pip);
      const float amt_to_add = static_cast<float>(pot_size_after_call) * static_cast<float>(a.amount) / 2.0f;
      const int amt_to_raise = std::max(my_pip, opp_pip) + static_cast<int>(amt_to_add);
      const int amt = std::min(max_raise, std::max(min_raise, amt_to_raise));
      actions_unscaled.at(i) = RaiseAction(amt);
    }
  }

  assert(actions_unscaled.size() == actions_mask.size());
  return std::make_pair(actions_unscaled, actions_mask);
}


/**
 * Called when a new game starts. Called exactly once.
 */
CfrPlayer::CfrPlayer() {
  std::string line;
  std::ifstream infile("./avg_strategy.txt");

  while (std::getline(infile, line)) {
    std::istringstream iss(line);

    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));

    const std::string key = strs.at(0);
    ActionRegrets regrets_this_key;
    assert(strs.size() == (1 + regrets_this_key.size()));
    for (int i = 0; i < regrets_this_key.size(); ++i) {
      regrets_this_key.at(i) = std::stod(strs.at(i + 1));
    }
    regrets_.emplace(key, regrets_this_key);
  }

  std::cout << "Read in regrets for " << regrets_.size() << " bucketed infosets" << std::endl;
}

/**
 * Called when a new round starts. Called NUM_ROUNDS times.
 *
 * @param game_state Pointer to the GameState object.
 * @param round_state Pointer to the RoundState object.
 * @param active Your player's index.
 */
void CfrPlayer::handle_new_round(GameState* game_state, RoundState* round_state, int active) {
  int my_bankroll = game_state->bankroll;  // the total number of chips you've gained or lost from the beginning of the game to the start of this round
  float game_clock = game_state->game_clock;  // the total number of seconds your bot has left to play this game
  int round_num = game_state->round_num;  // the round number from 1 to NUM_ROUNDS
  
  bool big_blind = static_cast<bool>(active);
  is_small_blind_ = !big_blind;
  check_fold_mode_ = CanCheckFoldRemainder(my_bankroll, round_num);

  printf("\n================== NEW ROUND: %d ==================\n", round_num);
  std::cout << "*** TIME REMAINING: " << game_clock << std::endl;
  printf("*** Big blind: %d\n", big_blind);

  street_ev_.clear();
  current_street_ = -1;
  history_ = HistoryTracker(big_blind);
}

/**
 * Called when a round ends. Called NUM_ROUNDS times.
 *
 * @param game_state Pointer to the GameState object.
 * @param terminal_state Pointer to the TerminalState object.
 * @param active Your player's index.
 */
void CfrPlayer::handle_round_over(GameState* game_state, TerminalState* terminal_state, int active) {
  const int my_delta = terminal_state->deltas[active];  // your bankroll change from this round
  RoundState* previous_state = (RoundState*) terminal_state->previous_state;  // RoundState before payoffs
  const int street = previous_state->street;  // 0, 3, 4, or 5 representing when this round ended
  const int round_num = game_state->round_num;
  const int my_stack = previous_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = previous_state->stacks[1-active];  // the number of chips your opponent has remaining
  const int my_contribution = STARTING_STACK - my_stack;  // the number of chips you have contributed to the pot
  const int opp_contribution = STARTING_STACK - opp_stack;  // the number of chips your opponent has contributed to the pot
  history_.Update(my_contribution, opp_contribution, street);

  const std::array<std::string, 2> my_cards = previous_state->hands[active];  // your cards
  const std::array<std::string, 2> opp_cards = previous_state->hands[1-active];  // opponent's cards or "" if not revealed
  const std::array<std::string, 5> board_cards = previous_state->deck;

  const std::string win_hand = my_delta >= 0 ? my_cards[0] + my_cards[1] : opp_cards[0] + opp_cards[1];
  const std::string lose_hand = my_delta >= 0 ? opp_cards[0] + opp_cards[1] : my_cards[0] + my_cards[1];
  std::string board = "";
  for (const std::string s : board_cards) {
    board += s;
  }

  // Print out the profiling results on the last round.
  if (round_num == 1000) {
    pf_.Profile();
  }

  const bool did_see_showdown = lose_hand.size() == 4 && win_hand.size() == 4;
  if (did_see_showdown) {
    ++num_showdowns_seen_;

    const bool did_converge = pf_.Unique() == 1;

    if (pf_.Nonzero() <= 0 || did_converge) {
      if (pf_.Nonzero() <= 0) {
        std::cout << "FAILURE: PermutationFilter particles all died" << std::endl;
      } else {
        std::cout << "SUCCESS: Permutation filter converged" << std::endl;
      }
      return;
    }

    const ShowdownResult result(win_hand, lose_hand, board);
    pf_.Update(result);
    std::cout << "[ROUNDOVER] Updated with showdown result" << std::endl;
    std::cout << "[ROUNDOVER] Particle filter nonzero = " << pf_.Nonzero() << std::endl;
  }

  std::cout << "\n[ROUNDOVER] Final history:" << std::endl;
  history_.Print();
  std::cout << std::endl;
}

/**
 * Where the magic happens - your code should implement this function.
 * Called any time the engine needs an action from your bot.
 *
 * @param game_state Pointer to the GameState object.
 * @param round_state Pointer to the RoundState object.
 * @param active Your player's index.
 * @return Your action.
 */
Action CfrPlayer::get_action(GameState* game_state, RoundState* round_state, int active) {
  const int round_num = game_state->round_num;
  const int legal_actions = round_state->legal_actions();  // mask representing the actions you are allowed to take
  const int street = round_state->street;  // 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
  const std::array<std::string, 2> my_cards = round_state->hands[active];  // your cards
  const std::array<std::string, 5> board_cards = round_state->deck;  // the board cards
  const int my_pip = round_state->pips[active];  // the number of chips you have contributed to the pot this round of betting
  const int opp_pip = round_state->pips[1-active];  // the number of chips your opponent has contributed to the pot this round of betting
  const int my_stack = round_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = round_state->stacks[1-active];  // the number of chips your opponent has remaining
  const int continue_cost = opp_pip - my_pip;  // the number of chips needed to stay in the pot
  const int my_contribution = STARTING_STACK - my_stack;  // the number of chips you have contributed to the pot
  const int opp_contribution = STARTING_STACK - opp_stack;  // the number of chips your opponent has contributed to the pot

  if (street != current_street_) {
    current_street_ = street;
    if (current_street_ == 0) {
      std::cout << "*** PREFLOP ***" << std::endl;
    } else if (current_street_ == 3) {
      std::cout << "*** FLOP ***" << std::endl;
    } else if (current_street_ == 4) {
      std::cout << "*** TURN ***" << std::endl;
    } else {
      std::cout << "*** RIVER ***" << std::endl;
    }
  }
  history_.Update(my_contribution, opp_contribution, street);

  if (check_fold_mode_) {
    std::cout << "*** [WIN] CHECK-FOLD MODE ACTIVATED ***" << std::endl;
    const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
    return check_is_allowed ? CheckAction() : FoldAction();
  }

  // Check fold if no particles left.
  if (pf_.Nonzero() <= 0) {
    std::cout << "[GETACTION] [FAILURE] Particle filter empty, check-folding" << std::endl;
    return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
  }

  const bool did_converge = (num_showdowns_seen_ > num_showdowns_converge_) && pf_.Unique() < 10;
  printf("[GETACTION] Did converge? %d (unique=%d)\n", did_converge, pf_.Unique());

  // If EV hasn't been computed for this street, do it here.
  if (street_ev_.count(street) == 0) {
    std::string board_str;
    for (int i = 0; i < street; ++i) {
      board_str += board_cards[i];
    }
    // If converged, only sample ONE permutation.
    const int nsamples = did_converge ? 1 : compute_ev_samples_;
    const float ev_this_street = pf_.ComputeEvRandom(
        my_cards[0] + my_cards[1], board_str, "", nsamples, compute_ev_iters_);
    street_ev_[street] = ev_this_street;
  }
  const float EV = street_ev_.at(street);

  const int pot_size = my_contribution + opp_contribution;
  printf("[GETACTION] round=%d | street=%d | ev=%f | pot_size=%d | continue_cost=%d |\n",
      round_num, street, EV, pot_size, continue_cost);
  
  const EvInfoSet infoset = MakeInfoSet(history_, 0, is_small_blind_, EV, street);
  const std::vector<std::string> bucket = BucketInfoSetSmall(infoset);
  const std::string key = BucketSmallJoin(bucket);

  const auto& actions_and_mask = MakeActions(round_state, active, history_);
  return RegretMatching(key, actions_and_mask.first, actions_and_mask.second);
}


Action CfrPlayer::RegretMatching(const std::string& key, const ActionVec& actions, const ActionMask& mask) {
  if (regrets_.count(key) == 0) {
    std::cout << "WARNING: Could not find key in regrets: " << key << std::endl;
    ActionRegrets uniform;
    std::fill(uniform.begin(), uniform.end(), 1.0f);
    regrets_.emplace(key, uniform);
  }

  std::cout << "Getting action for: " << key << std::endl;

  const ActionRegrets& regrets = regrets_.at(key);
  ActionRegrets masked_regrets;

  for (int i = 0; i < regrets.size(); ++i) {
    masked_regrets.at(i) = static_cast<double>(mask.at(i)) * std::fmax(0.0f, regrets.at(i));
  }

  std::discrete_distribution<int> distribution(masked_regrets.begin(), masked_regrets.end());
  const int sampled_idx = distribution(gen_);

  return actions.at(sampled_idx);
}

}
