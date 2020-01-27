#include "./cpp_skeleton/runner.hpp"
// #include "player.hpp"
#include "cfr_player.hpp"

int main(int argc, char* argv[]) {
  // pb::Player player;
  pb::CfrPlayer player;
  vector<string> args = parse_args(argc, argv);
  run_bot(&player, args);
  return 0;
}
