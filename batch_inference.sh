#!/bin/bash

# Batch inference script for LatentSync
# This script runs the batch inference with default parameters

echo "Starting batch inference for LatentSync..."
echo "=========================================="

# Default parameters
VIDEO_DIR="data/Video"
AUDIO_DIR="data/Audio/spk_ganyu"
OUTPUT_DIR="output/lipsync"
CONFIG_PATH="configs/unet/stage2_512.yaml"
CKPT_PATH="checkpoints/latentsync_unet.pt"
INFERENCE_STEPS=20
GUIDANCE_SCALE=1.5
MAX_COMBINATIONS=""
RANDOM_SEED=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --ckpt_path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --inference_steps)
            INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --max_combinations)
            MAX_COMBINATIONS="--max_combinations $2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="--random_seed $2"
            shift 2
            ;;
        --allow_repeat_videos)
            ALLOW_REPEAT_VIDEOS="--allow_repeat_videos"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --video_dir DIR          Directory containing video templates (default: data/Video)"
            echo "  --audio_dir DIR          Directory containing audio files (default: data/Audio/spk_ganyu)"
            echo "  --output_dir DIR         Output directory for generated videos (default: output/lipsync)"
            echo "  --config_path PATH       Path to UNet config file (default: configs/unet/stage2_512.yaml)"
            echo "  --ckpt_path PATH         Path to checkpoint file (default: checkpoints/latentsync_unet.pt)"
            echo "  --inference_steps N      Number of inference steps (default: 20)"
            echo "  --guidance_scale FLOAT   Guidance scale (default: 1.5)"
            echo "  --max_combinations N     Maximum number of combinations to process"
            echo "  --random_seed N          Random seed for reproducibility"
            echo "  --allow_repeat_videos    Allow same video to be selected multiple times"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0  # Run with default parameters"
            echo "  $0 --max_combinations 10  # Process only 10 combinations"
            echo "  $0 --inference_steps 30 --guidance_scale 2.0  # Higher quality settings"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python batch_inference.py"
CMD="$CMD --video_dir $VIDEO_DIR"
CMD="$CMD --audio_dir $AUDIO_DIR"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --config_path $CONFIG_PATH"
CMD="$CMD --ckpt_path $CKPT_PATH"
CMD="$CMD --inference_steps $INFERENCE_STEPS"
CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
CMD="$CMD --enable_deepcache"

if [ ! -z "$MAX_COMBINATIONS" ]; then
    CMD="$CMD $MAX_COMBINATIONS"
fi

if [ ! -z "$RANDOM_SEED" ]; then
    CMD="$CMD $RANDOM_SEED"
fi

if [ ! -z "$ALLOW_REPEAT_VIDEOS" ]; then
    CMD="$CMD $ALLOW_REPEAT_VIDEOS"
fi

echo "Command: $CMD"
echo ""

# Run the batch inference
$CMD 