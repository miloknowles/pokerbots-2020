#include "cfr_trainer.hpp"
#include "engine_modified.hpp"


int main(int argc, char const *argv[]) {
  pb::cfr::Options opt;
  // opt.NUM_TRAVERSALS_PER_ITER = 10;
  // opt.NUM_TRAVERSALS_EVAL = 10;
  // opt.BUCKET_FUNCTION = pb::cfr::Bucket_10_16;
  opt.EXPERIMENT_NAME = "MC_CFR_KMEANS";
  opt.EXPERIMENT_PATH = "/home/milo/kmeans/pokerbots-2020/cfr/" + opt.EXPERIMENT_NAME + "/";
  pb::cfr::CfrTrainer trainer(opt);
  trainer.Run();
  std::cout << "Finished running CFR" << std::endl;
  return 0;
}