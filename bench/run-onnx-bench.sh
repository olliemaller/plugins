#!/bin/bash
# Runs onnx_arena_bench.py inside the camera.ui container (CT 101) for each CUDA config,
# each in a fresh process, and reads its GPU memory with nvidia-smi on the host.
set -u
CT=101
SECS=${1:-60}
C=$(pct exec $CT -- docker ps --format '{{.Names}}' | grep -i camera | head -1)
[ -n "$C" ] || { echo "camera.ui container not found"; exit 1; }
# camera.ui keeps its data on Docker volumes, so search the container's mount points (find -xdev skips them)
MOUNTS=$(pct exec $CT -- docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$C")
echo "mounts: $MOUNTS"
PY=$(pct exec $CT -- docker exec "$C" sh -c "find $MOUNTS -path '*camera-ui-onnx-legacy/python-3.11*/bin/python3' 2>/dev/null | head -1")
MD=$(pct exec $CT -- docker exec "$C" sh -c "find $MOUNTS -type d -name models -path '*camera-ui-onnx-legacy*' 2>/dev/null | head -1")
echo "container=$C python=$PY models=$MD"
[ -n "$PY" ] && [ -n "$MD" ] || { echo "python or models not found"; exit 1; }
pct push $CT "$(dirname "$0")/onnx_arena_bench.py" /tmp/onnx_arena_bench.py
pct exec $CT -- docker cp /tmp/onnx_arena_bench.py "$C":/tmp/onnx_arena_bench.py

mem_of_bench() {
  # GPU memory of the benchmark process: python processes on the GPU, minus the camera.ui plugin itself
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits \
    | grep -i 'python' | grep -vi 'camera.ui' | awk -F', ' '{s+=$2} END {print s+0}'
}

wait_for() { # file, marker, timeout
  for _ in $(seq 1 "$3"); do grep -q "$2" "$1" 2>/dev/null && return 0; grep -q ABORT "$1" 2>/dev/null && return 1; sleep 1; done; return 1
}

echo
printf "%-10s %14s %18s %14s\n" config after_load "after ${SECS}s load" throughput
for cfg in today arena workspace both; do
  LOG=/tmp/onnx-bench-$cfg.log; : > "$LOG"
  pct exec $CT -- docker exec "$C" "$PY" /tmp/onnx_arena_bench.py "$MD" $cfg "$SECS" > "$LOG" 2>&1 &
  if ! wait_for "$LOG" LOADED 300; then echo "$cfg: failed, see $LOG"; tail -5 "$LOG"; wait; continue; fi
  sleep 5; M1=$(mem_of_bench)
  wait_for "$LOG" RUN_DONE $((SECS + 120)); sleep 2; M2=$(mem_of_bench)
  TP=$(grep THROUGHPUT "$LOG" | awk '{print $2}')
  printf "%-10s %10s MiB %14s MiB %10s/s\n" $cfg "$M1" "$M2" "$TP"
  wait
done
echo
echo "Per-model latency (ms, mean):"
for cfg in today arena workspace both; do echo "== $cfg"; grep '^LAT' /tmp/onnx-bench-$cfg.log | awk '{print "  "$2, $3}'; done
