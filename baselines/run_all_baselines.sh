#!/bin/bash

# Script to run all baseline compression methods
# Usage: ./run_all_baselines.sh [model_name]
# If no model_name is provided, runs all paper models

# Define all models used in the paper
PAPER_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "qwen/qwen2.5-7b-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "qwen/qwen2.5-vl-32B-instruct"
    "qwen/qwen2-audio-7b-instruct"
    "deepseek-ai/deepseek-coder-33b-instruct"
    "google/gemma-3-27b-it"
)

# Check arguments and set models to run
if [ $# -eq 0 ]; then
    echo "No model specified - running ALL paper models"
    MODELS_TO_RUN=("${PAPER_MODELS[@]}")
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [model_name]"
    echo ""
    echo "Options:"
    echo "  <model_name>  Run baselines for a specific model"
    echo "  (no args)     Run baselines for ALL paper models"
    echo ""
    echo "Models used in the paper:"
    for model in "${PAPER_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "This script will run all baseline methods:"
    echo "  - Baseline 0: No Compression"
    echo "  - Baseline 1: LZ4 Compression"
    echo "  - Baseline 2: ZSTD Compression"
    echo "  - Baseline 3: SafeTensors"
    echo "  - Baseline 4: ZipNN"
    exit 0
else
    MODELS_TO_RUN=("$1")
fi

echo "========================================="
echo "Running ALL baselines for ${#MODELS_TO_RUN[@]} model(s)"
for model in "${MODELS_TO_RUN[@]}"; do
    echo "  - $model"
done
echo "========================================="

# Log file for summary  
if [ ${#MODELS_TO_RUN[@]} -eq 1 ]; then
    LOG_FILE="baseline_results_$(echo "${MODELS_TO_RUN[0]}" | tr '/' '_')_$(date +%Y%m%d_%H%M%S).log"
else
    LOG_FILE="baseline_results_all_models_$(date +%Y%m%d_%H%M%S).log"
fi
echo "Results will be logged to: $LOG_FILE"

# Function to log and display
log_and_echo() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to run a baseline and handle errors
run_baseline() {
    local baseline_name="$1"
    local script_name="$2"
    
    log_and_echo ""
    log_and_echo "========================================="
    log_and_echo "STARTING: $baseline_name"
    log_and_echo "========================================="
    log_and_echo "Time: $(date)"
    
    start_time=$(date +%s)
    
    if [ -f "$script_name" ]; then
        log_and_echo "Running: ./$script_name \"$MODEL_NAME\""
        
        if ./"$script_name" "$MODEL_NAME" 2>&1 | tee -a "$LOG_FILE"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            log_and_echo "✅ COMPLETED: $baseline_name (Duration: ${duration}s)"
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            log_and_echo "❌ FAILED: $baseline_name (Duration: ${duration}s)"
            log_and_echo "ERROR: $baseline_name failed. Check the output above for details."
            
            # Ask user if they want to continue
            echo ""
            read -p "Do you want to continue with the next baseline? (y/n): " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_and_echo "Stopping execution at user request."
                exit 1
            fi
        fi
    else
        log_and_echo "❌ SKIPPED: $baseline_name - Script $script_name not found"
    fi
}

# Start overall timing
overall_start=$(date +%s)

log_and_echo "Starting baseline comparison for ${#MODELS_TO_RUN[@]} model(s)"
log_and_echo "Start time: $(date)"

# Outer loop: iterate over all models
for MODEL_NAME in "${MODELS_TO_RUN[@]}"; do
    model_start=$(date +%s)
    
    log_and_echo ""
    log_and_echo "######################################"
    log_and_echo "STARTING MODEL: $MODEL_NAME"
    log_and_echo "######################################"
    log_and_echo "Model start time: $(date)"
    
    # Run all baselines for this model
    run_baseline "Baseline 0: No Compression" "baseline0_no_compression.sh"
    run_baseline "Baseline 1: LZ4 Compression" "baseline1_lz4.sh" 
    run_baseline "Baseline 2: ZSTD Compression" "baseline2_zstd.sh"
    run_baseline "Baseline 3: SafeTensors" "baseline3_safetensors.sh"
    run_baseline "Baseline 4: ZipNN" "baseline4_zipnn.sh"
    
    model_end=$(date +%s)
    model_duration=$((model_end - model_start))
    
    log_and_echo ""
    log_and_echo "✅ COMPLETED MODEL: $MODEL_NAME"
    log_and_echo "Model duration: ${model_duration}s ($(($model_duration / 3600))h $(($model_duration % 3600 / 60))m $(($model_duration % 60))s)"
    log_and_echo "######################################"
    
done

# Final summary
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

log_and_echo ""
log_and_echo "========================================="
log_and_echo "ALL MODELS AND BASELINES COMPLETED"
log_and_echo "========================================="
log_and_echo "Models processed: ${#MODELS_TO_RUN[@]}"
for model in "${MODELS_TO_RUN[@]}"; do
    log_and_echo "  ✅ $model"
done
log_and_echo "Total duration: ${overall_duration}s ($(($overall_duration / 3600))h $(($overall_duration % 3600 / 60))m $(($overall_duration % 60))s)"
log_and_echo "End time: $(date)"
log_and_echo "Results logged to: $LOG_FILE"

echo ""
echo "🎉 All models and baselines have been executed!"
echo "📊 Check the log file for detailed results: $LOG_FILE"
