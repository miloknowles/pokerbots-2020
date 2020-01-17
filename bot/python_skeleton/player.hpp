#pragma once

#include <unordered_map>

#include "./cpp_skeleton/actions.hpp"
#include "./cpp_skeleton/states.hpp"
#include "./cpp_skeleton/bot.hpp"

#include "./pokerbots_cpp_python/permutation_filter.hpp"

namespace pb {

// 4 betting actions per player.
static constexpr int kMaxActionsPerStreet = 8;

class Player : public Bot {
  private:
    PermutationFilter pf_{25000};
    int compute_ev_samples_ = 5;
    int compute_ev_iters_ = 2000;

    int num_showdowns_seen_ = 0;
    int num_showdowns_converge_ = 50;

    // Keep track of some info for betting.
    int prev_street_ = -1;

    // 0 is us, 1 is opp.
    std::array<int, 2> contributions_;
    int prev_street_contrib_;
    std::unordered_map<int, float> street_ev_{};

    // Encodes the actions taken so far (bets as % of pot).
    int next_action_idx_;
    std::array<int, 4*kMaxActionsPerStreet> history_{};
    // std::unordered_map<int, int> street_num_raises_{};

  public:
    /**
     * Called when a new game starts. Called exactly once.
     */
    Player();

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

    Action HandleActionConverged(float ev, int round_num, int street, int pot_size, int continue_cost,
                                 int legal_actions, int min_raise, int max_raise, int my_contribution,
                                 int opp_contribution);

    Action HandleActionNotConverged(float ev, int round_num, int street, int pot_size, int continue_cost,
                                 int legal_actions, int min_raise, int max_raise, int my_contribution,
                                 int opp_contribution);
    
    void UpdateHistory(int my_contrib, int opp_contrib, int street);
};

}
