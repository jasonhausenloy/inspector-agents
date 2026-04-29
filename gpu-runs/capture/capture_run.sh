#!/usr/bin/env bash
# capture_run.sh <phase_dir> <command...>
# Starts collectors, runs the wrapped command, stops collectors on EXIT,
# returns the wrapped command's exit code.

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <phase_dir> <command...>" >&2
    exit 2
fi

PHASE_DIR="$1"; shift
mkdir -p "$PHASE_DIR"

NVSMI_PID=""
DCGM_PID=""
NETDEV_PID=""

cleanup() {
    [[ -n "$NVSMI_PID"  ]] && kill "$NVSMI_PID"  2>/dev/null || true
    [[ -n "$DCGM_PID"   ]] && kill "$DCGM_PID"   2>/dev/null || true
    [[ -n "$NETDEV_PID" ]] && kill "$NETDEV_PID" 2>/dev/null || true
    # Brief wait for collectors to flush stdout
    sleep 1 2>/dev/null || true
}
trap cleanup EXIT

# --- nvidia-smi @ 1 Hz -----------------------------------------------------
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu \
    --format=csv,nounits -lms 1000 > "$PHASE_DIR/nvsmi.csv" 2>"$PHASE_DIR/nvsmi.err" &
NVSMI_PID=$!

# --- DCGM @ 10 Hz, only if available + privileged --------------------------
# `dcgmi dmon -e 1001 -c 1` is a stricter test than `discovery -l`: it confirms
# the profile metrics are actually queryable, not just that the host engine is up.
if command -v dcgmi >/dev/null 2>&1 \
   && dcgmi dmon -e 1001 -c 1 >/dev/null 2>&1; then
    # Field IDs:
    #   1001 GR_ENGINE_ACTIVE  1002 SM_ACTIVE  1003 SM_OCCUPANCY
    #   1004 PIPE_TENSOR_ACTIVE  1005 DRAM_ACTIVE
    #   1009 PCIE_TX_BYTES  1010 PCIE_RX_BYTES
    #   1011 NVLINK_TX_BYTES  1012 NVLINK_RX_BYTES
    dcgmi dmon -e 1001,1002,1003,1004,1005,1009,1010,1011,1012 -d 100 \
        > "$PHASE_DIR/dcgm.csv" 2>"$PHASE_DIR/dcgm.err" &
    DCGM_PID=$!
else
    echo "dcgmi unavailable or unprivileged; skipping DCGM capture" \
        > "$PHASE_DIR/dcgm.skipped"
fi

# --- /proc/net/dev @ 1 Hz --------------------------------------------------
(
    while true; do
        printf '%s\n' "--- $(date -u +%s.%N) ---"
        cat /proc/net/dev
        sleep 1
    done
) > "$PHASE_DIR/netdev.log" 2>/dev/null &
NETDEV_PID=$!

# --- NCCL env: per-rank logs to avoid collision ----------------------------
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,INIT
export NCCL_DEBUG_FILE="$PHASE_DIR/nccl_%h_%p.log"

# --- Run the wrapped command ----------------------------------------------
"$@" > "$PHASE_DIR/stdout.log" 2> "$PHASE_DIR/stderr.log"
RC=$?

exit "$RC"
