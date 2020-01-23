python ../train_cfr.py --EXPERIMENT_NAME mc_cfr_03 \
  --NUM_TRAVERSE_WORKERS 28 \       # 28 works well on 32 core machine.
  --NUM_TRAVERSALS_PER_ITER 560 \   # Multiple of num workers.
  --NUM_TRAVERSALS_EVAL 10 \        # Fewer needed with no external sampling.
  --TRAVERSE_DEBUG_PRINT_HZ 5 \
  --NUM_CFR_ITERS 500 \
