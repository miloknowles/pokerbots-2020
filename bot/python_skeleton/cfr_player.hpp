#pragma once

#include <unordered_map>
#include <random>
#include <utility>

#include "./cpp_skeleton/actions.hpp"
#include "./cpp_skeleton/states.hpp"
#include "./cpp_skeleton/bot.hpp"

#include "./pokerbots_cpp_python/permutation_filter.hpp"
#include "./history_tracker.hpp"

namespace pb {


typedef std::array<Action, 6> ActionVec;
typedef std::array<int, 6> ActionMask;
typedef std::array<double, 6> ActionRegrets;


std::pair<ActionVec, ActionMask> MakeActions(RoundState* round_state, int active, const HistoryTracker& tracker);


class CfrPlayer : public Bot {
  private:
    PermutationFilter pf_{25000};
    int compute_ev_samples_ = 3;
    int compute_ev_iters_ = 2000;

    int num_showdowns_seen_ = 0;
    int num_showdowns_converge_ = 50;

    bool check_fold_mode_ = false;
    bool is_small_blind_ = false;

    int current_street_ = -1;
    HistoryTracker history_{false};

    // Keep track of some info for betting.
    std::unordered_map<int, float> street_ev_{};

    std::random_device rd_{};
    std::mt19937 gen_{rd_()};
    std::uniform_real_distribution<> real_{0, 1};

    std::unordered_map<std::string, ActionRegrets> regrets_;

  public:
    /**
     * Called when a new game starts. Called exactly once.
     */
    CfrPlayer();

    /**
     * Called when a new round starts. Called NUM_ROUNDS times.
     *
     * @param game_state Pointer to the GameState object.
     * @param round_state Pointer to the RoundState object.
     * @param active Your player's index.
     */
    void handle_new_round(GameState* game_state, RoundState* round_state, int active);

    /**
     * Called when a round ends. Called NUM_ROUNDS times.
     *
     * @param game_state Pointer to the GameState object.
     * @param terminal_state Pointer to the TerminalState object.
     * @param active Your player's index.
     */
    void handle_round_over(GameState* game_state, TerminalState* terminal_state, int active);

    /**
     * Where the magic happens - your code should implement this function.
     * Called any time the engine needs an action from your bot.
     *
     * @param game_state Pointer to the GameState object.
     * @param round_state Pointer to the RoundState object.
     * @param active Your player's index.
     * @return Your action.
     */
    Action get_action(GameState* game_state, RoundState* round_state, int active);

    Action RegretMatching(const std::string& key, const ActionVec& actions, const ActionMask& mask);

    void MaybePrintNewStreet(int street);

    // Backup action handlers (when CFR fails).
    Action HandleActionPreflop(float ev, int round_num, int street, int pot_size, int continue_cost,
                               int legal_actions, int min_raise, int max_raise, int my_contribution,
                               int opp_contribution, bool is_big_blind);

    Action HandleActionFlop(float ev, int round_num, int street, int pot_size, int continue_cost,
                               int legal_actions, int min_raise, int max_raise, int my_contribution,
                               int opp_contribution, bool is_big_blind);
    
    Action HandleActionTurn(float ev, int round_num, int street, int pot_size, int continue_cost,
                               int legal_actions, int min_raise, int max_raise, int my_contribution,
                               int opp_contribution, bool is_big_blind);
};

}
