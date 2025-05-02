#!/bin/bash
DEVICE="/dev/sda1" # Set device

# BENCHMARK_CMD="./build/final_decode"
BENCHMARK_CMD="./build/baseline0_no_compression decode bf16-int8"

SPEED_VALUES=(20 50 100 200 300 400 500 600 700 800 900 1000 1200 1500) # Desired speeds in MiB/s
# SPEED_VALUES=(20)

# === Loop through speed values ===
for SPEED_MIB in "${SPEED_VALUES[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting benchmark for ${SPEED_MIB} MiB/s..."
    echo "--------------------------------------------------"

    LIMIT_BPS=$(($SPEED_MIB * 1024 * 1024))

    echo "Clearing cache..."
    sudo bash /home/raunaks/drop_cache.sh
    sleep 2

    # === Run with calculated limit ===
    echo "Running benchmark at ${SPEED_MIB} MiB/s (Limit: ${LIMIT_BPS} Bps) on ${DEVICE}..."
    sudo systemd-run --unit=benchmark-${SPEED_MIB}mibs --slice=throttled --scope \
      -p IOReadBandwidthMax="${DEVICE} ${LIMIT_BPS}" \
      $BENCHMARK_CMD

    echo "Benchmark finished for ${SPEED_MIB} MiB/s."
    echo ""
    sleep 5
done