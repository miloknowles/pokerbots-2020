#pragma once

#include <unordered_map>

#include "./cpp_skeleton/actions.hpp"
#include "./cpp_skeleton/states.hpp"
#include "./cpp_skeleton/bot.hpp"

#include "./pokerbots_cpp_python/permutation_filter.hpp"

namespace pb {

class Player : public Bot {
  private:
    PermutationFilter pf_;
    int compute_ev_samples_ = 1;
    int compute_ev_iters_ = 100;

    int num_showdowns_seen_ = 0;
    int num_showdowns_converge_ = 120;

    std::unordered_map<int, float> street_ev_{};
    std::unordered_map<int, int> street_num_raises_{};

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
};

}
