#!/bin/bash
# Run remaining A/B cycles sequentially, clean attribution (base + one lever each).
cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks

echo "########## CYCLE 2: schedule-conservativeness 0.5 ##########"
bash sglang_ab.sh sched05 "--speculative-attention-mode decode --schedule-conservativeness 0.5" >> ab_all.log 2>&1 || echo "cycle2 failed" >> ab_all.log

echo "########## CYCLE 3: sampling-backend pytorch ##########"
bash sglang_ab.sh sbsam "--speculative-attention-mode decode --sampling-backend pytorch" >> ab_all.log 2>&1 || echo "cycle3 failed" >> ab_all.log

echo "########## CYCLE 4: enable-linear-replayssm-spec ##########"
bash sglang_ab.sh replayssm "--speculative-attention-mode decode --enable-linear-replayssm-spec" >> ab_all.log 2>&1 || echo "cycle4 failed" >> ab_all.log

echo "########## CYCLE 5: enable-overlap-schedule ##########"
bash sglang_ab.sh overlap "--speculative-attention-mode decode --enable-overlap-schedule" >> ab_all.log 2>&1 || echo "cycle5 failed" >> ab_all.log

echo "ALL-CYCLES-DONE" >> ab_all.log
