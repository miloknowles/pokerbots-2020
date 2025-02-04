#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "cfr_player.hpp"

namespace pb {


void PrintVector(const std::vector<double>& v) {
  for (const double vi : v) {
    std::cout << vi << " ";
  }
  std::cout << std::endl;
}


static bool CanCheckFoldRemainder(const int delta, const int round_num) {
  return delta > (1.5f * (1000.0f - static_cast<float>(round_num)) + 1);
}


static int MakeRelativeBet(const float frac, const int pot_size, const int min_raise, const int max_raise) {
  const int amt = static_cast<int>(frac * static_cast<float>(pot_size));
  const int clamped = std::min(max_raise, std::max(min_raise, amt));
  return clamped;
}

static Action AddNoiseToBet(const Action& action, int min_raise, int max_raise) {
  const int mult = (rand() % 2 == 0) ? -1 : 1;
  const int noise = mult * (rand() % 3);
  printf("Adding noise: %d\n", noise);
  return RaiseAction(std::min(max_raise, std::max(min_raise, action.amount + noise)));
}


std::pair<cfr::ActionVec, cfr::ActionMask> MakeActions(RoundState* round_state, int active, const HistoryTracker& tracker) {
  const int legal_actions = round_state->legal_actions();

  const int my_pip = round_state->pips[active];
  const int opp_pip = round_state->pips[1-active];

  const int min_raise = round_state->raise_bounds()[0];
  const int max_raise = round_state->raise_bounds()[1];

  const int my_stack = round_state->stacks[active];  // the number of chips you have remaining
  const int opp_stack = round_state->stacks[1-active];  // the number of chips your opponent has remaining

  const int pot_size = 2 * 200 - my_stack - opp_stack;

  const int bet_actions_so_far = tracker.History().back().size();
  const int bet_actions_this_street = (round_state->street > 0) ? cfr::kMaxActionsPerStreet : (cfr::kMaxActionsPerStreet + 2);
  const bool force_fold_call = bet_actions_so_far >= (bet_actions_this_street - 1);

  cfr::ActionMask actions_mask;
  std::fill(actions_mask.begin(), actions_mask.end(), 0);

  // NOTE: These are HALF POT multiples.
  cfr::ActionVec actions_unscaled = {
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
  strategy_.Load("./avg_strategy.txt");
  std::cout << "Read in regrets for " << strategy_.Size() << " bucketed infosets" << std::endl;
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
  // sampled_perms_.clear();
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

void CfrPlayer::MaybePrintNewStreet(int street) {
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
}

static cfr::ActionRegrets ApplyMaskAndUniformSquash(const cfr::ActionRegrets& p, const cfr::ActionMask& mask) {
  double denom = 0;
  int valid = 0;
  cfr::ActionRegrets out;

  for (int i = 0; i < p.size(); ++i) {
    const double masked = p[i] * static_cast<double>(mask[i]);
    denom += masked;
    out[i] = masked;
    valid += mask[i];
  }

  // If the sum of regrets <= 0, return uniform dist over valid actions.
  if (denom <= 1e-3) {
    for (int i = 0; i < p.size(); ++i) {
      out[i] = static_cast<double>(mask[i]) / static_cast<double>(valid);
    }
    return out;

  // Otherwise normalize by sum.
  } else {
    for (int i = 0; i < p.size(); ++i) {
      out[i] /= denom;
    }

    // Squash anything below 0.05.
    for (int i = 0; i < p.size(); ++i) {
      out[i] = out[i] < 0.05 ? 0 : out[i];
    }

    return out;
  }
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

  MaybePrintNewStreet(street);
  history_.Update(my_contribution, opp_contribution, street);

  // Check fold if we can afford to do so without losing.
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
  printf("[GETACTION] CONVERGED=%d (unique=%d)\n", did_converge, pf_.Unique());

  // If EV hasn't been computed for this street, do it here.
  if (street_ev_.count(street) == 0) {
    // Sample new permutations for this street.
    // sampled_perms_ = pf_.SampleValid(did_converge ? 1 : compute_ev_samples_);

    std::string board_str;
    for (int i = 0; i < street; ++i) {
      board_str += board_cards[i];
    }

    // If converged, only sample ONE permutation.
    int nsamples = did_converge ? 1 : std::min(street_nsamples_.at(street), pf_.Unique());
    const std::string hand = my_cards[0] + my_cards[1];
    const float ev_this_street = pf_.ComputeEvRandom(hand, board_str, "", nsamples, compute_ev_iters_.at(street));
    street_ev_[street] = ev_this_street;
  }
  const float EV = street_ev_.at(street);

  const int pot_size = my_contribution + opp_contribution;
  printf("[GETACTION] round=%d | street=%d | ev=%f | pot_size=%d | continue_cost=%d |\n",
      round_num, street, EV, pot_size, continue_cost);
  
  // Do bucketing and regret matching to get an action.
  const cfr::EvInfoSet infoset = MakeInfoSet(history_, 0, is_small_blind_, EV, street);
  const std::string key = bucket_function_(infoset);
  std::cout << key << std::endl;

  // If CFR never encountered this situation, revert to backup logic.
  if (!strategy_.HasBucket(key)) {
    std::cout << "WARNING: Couldn't find CFR bucket: " << key << std::endl;
    const int min_raise = round_state->raise_bounds()[0];
    const int max_raise = round_state->raise_bounds()[1];
    const bool is_bb = static_cast<bool>(active);
    if (street == 0) {
      return HandleActionPreflop(EV, round_num, street, pot_size, continue_cost, legal_actions,
                                min_raise, max_raise, my_contribution, opp_contribution, is_bb);
    } else if (street == 3) {
      return HandleActionFlop(EV, round_num, street, pot_size, continue_cost, legal_actions,
                              min_raise, max_raise, my_contribution, opp_contribution, is_bb);
    } else {
      return HandleActionTurn(EV, round_num, street, pot_size, continue_cost, legal_actions,
                              min_raise, max_raise, my_contribution, opp_contribution, is_bb);
    }

  // Otherwise, do regret matching to choose an action.
  } else {
    const auto& actions_and_mask = MakeActions(round_state, active, history_);

    const cfr::ActionRegrets& action_probs = strategy_.GetStrategy(infoset);
    const cfr::ActionRegrets& action_probs_masked = cfr::ApplyMaskAndUniform(action_probs, actions_and_mask.second);

    // const cfr::ActionRegrets action_probs_masked = ApplyMaskAndUniformSquash(action_probs, actions_and_mask.second);

    std::discrete_distribution<int> distribution(action_probs_masked.begin(), action_probs_masked.end());
    const int sampled_idx = distribution(gen_);

    std::cout << key << std::endl;
    std::cout << "[CFR] Action probabilities=" << std::endl;
    PrintVector(std::vector<double>(action_probs_masked.begin(), action_probs_masked.end()));

    const cfr::ActionVec& actions = actions_and_mask.first;
    const Action chosen_action = actions.at(sampled_idx);
    printf("[CFR] Sampled action %d\n", sampled_idx);

    // if (chosen_action.action_type == RAISE_ACTION_TYPE) {
    //   std::cout << "adding noise" << std::endl;
    //   const int min_raise = round_state->raise_bounds()[0];
    //   const int max_raise = round_state->raise_bounds()[1];
    //   return AddNoiseToBet(chosen_action, min_raise, max_raise);
    // }
    if (street >= 3) {

    }

    // If it's after the flop and an action would put us all in, do the less extreme action.
    if (street >= 3) {
      if (chosen_action.action_type == CALL_ACTION_TYPE) {
        const bool is_all_in = (my_contribution + continue_cost) == 200;
        if (EV <= 0.80 && is_all_in) {
          std::cout << "[TOO RISKY] Call would put me all in, EV <= 0.80 so FOLDING" << std::endl;
          return FoldAction();
        }
      } else if (chosen_action.action_type == RAISE_ACTION_TYPE) {
        const bool is_all_in = (chosen_action.amount + my_contribution) == 200;
        if (EV <= 0.80 && is_all_in) {
          std::cout << "[TOO RISKY] Raise would put me all in, EV <= 0.80 so CHECK/CALLING" << std::endl;
          const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
          return check_is_allowed ? CheckAction() : CallAction();
        }
      }
    }

    return chosen_action;
  }
}


//================================= BACKUP STRATEGY =======================================

Action CfrPlayer::HandleActionPreflop(float EV, int round_num, int street, int pot_size,
                                   int continue_cost, int legal_actions, int min_raise,
                                   int max_raise, int my_contribution, int opp_contribution,
                                   bool is_big_blind) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool must_pay_to_continue = continue_cost > 0;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  const bool is_our_first_action = (is_big_blind && my_contribution == BIG_BLIND) ||
                                   (!is_big_blind && my_contribution == SMALL_BLIND);

  if (is_our_first_action) {
    // CASE 1: First action and we are SMALLBLIND.
    if (!is_big_blind) {
      std::cout << "PREFLOP_1: We are SB, first action" << std::endl;
      assert(continue_cost > 0);
      if (EV >= 0.80 && raise_is_allowed) {
        std::cout << "1A: great EV, 8BB raise" << std::endl;
        return RaiseAction(8 * BIG_BLIND);
      } else if (EV >= 0.70 && raise_is_allowed) {
        std::cout << "1B: pretty good EV, 6BB raise" << std::endl;
        return RaiseAction(6 * BIG_BLIND);
      } else if (EV >= 0.50 && raise_is_allowed) {
        std::cout << "1C: okay EV, 10BB raise to try to push them out" << std::endl;
        return RaiseAction(6 * BIG_BLIND);
      } else if (EV >= 50) {
        return CallAction();
      } else {
        std::cout << "EV not good enough to call BB, folding" << std::endl;
        return FoldAction();
      }
    // CASE 2: First action and we are BIG.
    } else {
      std::cout << "PREFLOP_2: first action and we are BB" << std::endl;
      const bool other_player_did_raise = (continue_cost > 0);

      // If other player DID raise, consider reraising.
      if (other_player_did_raise) {
        std::cout << "the opponent raised on their first action" << std::endl;
        const int pot_after_call = 2 * opp_contribution;
        const float equity = EV * static_cast<float>(pot_after_call);
        if (EV > 0.80) {
          std::cout << "2A: really high chance of winning, doing 6BB re-raise" << std::endl;
          return RaiseAction(6 * BIG_BLIND);
        } else if (EV > 0.70) {
          std::cout << "2B: pretty good EV, doing 4BB re-raise" << std::endl;
          return RaiseAction(4 * BIG_BLIND);
        } else if (equity > 0.60) {
          std::cout << "2C: OK ev, calling" << std::endl;
          return CallAction();
        } else {
          std::cout << "2D: other player raised but equity not high enough, folding" << std::endl;
          return FoldAction();
        }
      
      // Other player called but didn't raise.
      } else {
        std::cout << "We are BB, opponent called their first action." << std::endl;
        // Try to make the other player fold.
        if (EV >= 0.65 && raise_is_allowed) {
          std::cout << "2D: okay EV, trying to push out other player before flop, raising 8BB" << std::endl;
          return RaiseAction(6 * BIG_BLIND);
        } else {
          std::cout << "2G: EV not high enough to raise, checking" << std::endl;
          return CheckAction();
        }
      }
    }
  
  // Not our first action, limit the number of re-raises we'll do.
  } else {
    const bool other_player_did_raise = (continue_cost > 0);
  
    // CASE 3: Not our first action, must pay to continue.
    if (other_player_did_raise) {
      std::cout << "PREFLOP_3: not our first action, other player raised" << std::endl;
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);
      
      // Raise again only if pot odds are really good.
      if (equity > 8.0f * continue_cost) {
        std::cout << "3A: great pot odds (8:1), 6BB raise" << std::endl;
        return raise_is_allowed ? RaiseAction(6 * BIG_BLIND) : CallAction();
    
      // Otherwise, stay in it if it's barely worth it.
      } else if (equity > 5.0f * continue_cost) {
        std::cout << "3B: good pot odds (5:1), raising 2BB" << std::endl;
        return raise_is_allowed ? RaiseAction(4 * BIG_BLIND) : CallAction();
      } else if (equity > 1.5f * continue_cost) {
        return CallAction();
      } else {
        std::cout << "3C: equity not good enough to continue, folding" << std::endl;
        return FoldAction();
      }
    
    // CASE 4: Not our first action, don't need to pay to continue.
    } else {
      std::cout << "PREFLOP_4: not our first action, call NOT required." << std::endl;
      // Try to make the other player fold.
      if (EV >= 0.80 && raise_is_allowed) {
        std::cout << "4A: great EV, doing 8BB bet" << std::endl;
        return RaiseAction(8 * BIG_BLIND);
      } else if (EV > 0.70 && raise_is_allowed) {
        std::cout << "4B: good EV, doing 4BB bet" << std::endl;
        return RaiseAction(4 * BIG_BLIND);
      } else {
        std::cout << "4C: EV not good enough to bet, just checking" << std::endl;
        return CheckAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

Action CfrPlayer::HandleActionFlop(float EV, int round_num, int street, int pot_size,
                                int continue_cost, int legal_actions, int min_raise,
                                int max_raise, int my_contribution, int opp_contribution,
                                bool is_big_blind) {
  std::cout << "Getting a FLOP action" << std::endl;
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  if (!is_big_blind) {
    // CASE 1: If the opponent checked the flop, they probably don't have anything too good.
    if (check_is_allowed) {
      // EV isn't very good, consider trying to fold out the opponent.
      if (EV < 0.5) {
        // i.e if EV is 30%, bluff it 30% of the time.
        const bool do_bluff_raise = real_(gen_) < EV;
        if (do_bluff_raise) {
          std::cout << "1A: opponent checked on the FLOP, randomly chose to BLUFF" << std::endl;
          const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
          return raise_is_allowed ? RaiseAction(amt) : CheckAction();
        } else {
          std::cout << "1B: opponent checked on the FLOP, randomly chose to CHECK" << std::endl;
          return CheckAction();
        }
      } else if (EV > 0.9) {
        const int amt = MakeRelativeBet(3.0f, pot_size, min_raise, max_raise);
        std::cout << "1C: really good EV, 3pot bet" << std::endl;
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else if (EV > 0.7) {
        std::cout << "1D: pretty good EV, pot bet" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else {
        std::cout << "1E: middle EV, check" << std::endl;
        return CheckAction();
      }
    
    // CASE 2: The opponent bet on the flop, they probably have something good?
    } else {
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);

      if (equity > 4*continue_cost) {
        std::cout << "2A: great pot odds, pot raise" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CallAction();
      } else if (equity > 1.3*continue_cost) {
        std::cout << "2B: pretty good pot odds, just call" << std::endl;
        return CallAction();
      } else {
        std::cout << "2C: not good pot odds, fold" << std::endl;
        return FoldAction();
      }
    }
  } else {
    // CASE 3: We are BB and this is the FIRST action.
    if (check_is_allowed) {
      if (EV > 0.8) {
        std::cout << "3A: great EV, 2pot raise" << std::endl;
        const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else if (EV > 0.7) {
        std::cout << "3B: good EV, pot raise" << std::endl;
        const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CheckAction();
      } else {
        return CheckAction();
      }
    
    // CASE 4: We are BB and this is not the FIRST action - opponent must have bet or raised.
    } else {
      const int pot_after_call = 2 * opp_contribution;
      const float equity = EV * static_cast<float>(pot_after_call);

      if (equity > 4*continue_cost) {
        std::cout << "4A: great pot odds (4:1), pot raise" << std::endl;
        const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
        return raise_is_allowed ? RaiseAction(amt) : CallAction();
      } else if (equity > 2*continue_cost) {
        std::cout << "4B: pretty good (2:1) pot odds, just call" << std::endl;
        return CallAction();
      } else {
        std::cout << "4C: not good pot odds, fold" << std::endl;
        return FoldAction();
      }
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

Action CfrPlayer::HandleActionTurn(float EV, int round_num, int street, int pot_size,
                                int continue_cost, int legal_actions, int min_raise,
                                int max_raise, int my_contribution, int opp_contribution,
                                bool is_big_blind) {
  const bool check_is_allowed = CHECK_ACTION_TYPE & legal_actions;
  const bool raise_is_allowed = RAISE_ACTION_TYPE & legal_actions;

  // Don't need to pay to continue.
  if (check_is_allowed) {
    // Just check if neutral odds.
    if (EV < 0.75 || !raise_is_allowed) {
      std::cout << "1A: (CheckAllowed) EV < 0.7 || !raise_is_allowed ==> CheckAction" << std::endl;
      return CheckAction();
  
    // If EV is pretty good (70-85%), do a pot raise.
    } else if (EV <= 0.85) {
      const int raise_amt = MakeRelativeBet(0.5, pot_size, min_raise, max_raise);
      std::cout << "1B: (CheckAllowed) EV <= 0.8 ==> 1/2PotRaise" << std::endl;
      return RaiseAction(raise_amt);

    // If EV above 0.8, do pot raise.
    } else {
      const int raise_amt = MakeRelativeBet(1.0, pot_size, min_raise, max_raise);
      std::cout << "1C: (CheckAllowed) EV above 0.8 ==> PotRaise" << std::endl;
      return RaiseAction(raise_amt);
    }

  // Must pay to continue.
  } else {
    const int pot_after_call = 2 * opp_contribution;
    const float equity = EV * static_cast<float>(pot_after_call);

    if (EV > 0.95) {
      std::cout << "2A: really high EV, 2pot raise" << std::endl;
      const int amt = MakeRelativeBet(2.0f, pot_size, min_raise, max_raise);
      return raise_is_allowed ? RaiseAction(amt) : CallAction();
    } else if (EV > 0.80) {
      std::cout << "2B: pretty high EV, pot raise" << std::endl;
      const int amt = MakeRelativeBet(1.0f, pot_size, min_raise, max_raise);
      return raise_is_allowed ? RaiseAction(amt) : CallAction();
    } else if (equity > 3*continue_cost) {
      std::cout << "2C: good enough pot odds (3:1), calling" << std::endl;
      return CallAction();
    } else {
      std::cout << "pot odds not good enough, folding" << std::endl;
      return FoldAction();
    }
  }

  std::cout << "WARNING: betting logic didn't handle a case, doing check-fold" << std::endl;
  return (CHECK_ACTION_TYPE & legal_actions) ? CheckAction() : FoldAction();
}

}
