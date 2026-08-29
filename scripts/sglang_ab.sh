#!/bin/bash
# One A/B cycle: stop server, drop caches (both nodes), optionally set EXTRA_ARGS,
# boot, wait for :8888, benchmark (aggregate + single-stream).
# Usage: sglang-ab.sh <label> [new_EXTRA_ARGS]   (empty EXTRA_ARGS = don't touch)
set -e
cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks
LABEL=$1
NEWARGS=$2

./stop.sh >/dev/null 2>&1 || true
sleep 3
IMG=lmsysorg/sglang:qwen38flashnext
docker run --rm --privileged --pid=host $IMG sh -c "sync && echo 3 > /proc/sys/vm/drop_caches" >/dev/null 2>&1
ssh -o BatchMode=yes spark2 "docker run --rm --privileged --pid=host $IMG sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'" >/dev/null 2>&1

if [ -n "$NEWARGS" ]; then
  sed -i "s|^EXTRA_ARGS=.*|EXTRA_ARGS=${NEWARGS}|" .env
fi

setsid nohup ./start.sh serve > boot_ab_${LABEL}.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
  if curl -s --max-time 5 http://127.0.0.1:8888/health >/dev/null 2>&1; then break; fi
  if ! pgrep -f "start.sh serve" >/dev/null; then echo "BOOT-EXITED"; break; fi
  sleep 20
done
echo "=== $LABEL | EXTRA_ARGS=$(grep ^EXTRA_ARGS .env | cut -c1-120) ==="
grep "max_total_num_tokens" boot_ab_${LABEL}.log | tail -1 | cut -c1-100
docker exec qwen38-flash-next-head python -m sglang.bench_serving --backend sglang --model RadixArk/Qwen3.8-Flash-Next-NVFP4 --base-url http://127.0.0.1:8888 --dataset-name random-ids --random-input-len 256 --random-output-len 256 --num-prompts 24 --request-rate 8 2>&1 | grep -E "Output token throughput|Total token throughput|Mean E2E"
s=$(date +%s.%N)
curl -s --max-time 300 http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Write a complete Python function to merge two sorted linked lists."}],"max_tokens":600}' >/dev/null
e=$(date +%s.%N)
echo "single-stream 600-tok wall: $(echo "$e - $s" | bc)s"
