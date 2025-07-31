#!/usr/bin/env python3
"""
Batch inference script for LatentSync
Iterates through all video templates and audio files to generate lip sync videos
"""

import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import random
from omegaconf import OmegaConf
import torch

def get_video_files(video_dir):
    """Get all video files from the video directory"""
    video_path = Path(video_dir)
    video_files = []
    
    if video_path.exists():
        for file in video_path.glob("*.mp4"):
            video_files.append(str(file))
    
    return sorted(video_files)

def get_audio_files(audio_dir):
    """Get all audio files from the audio directory"""
    audio_path = Path(audio_dir)
    audio_files = []
    
    if audio_path.exists():
        for file in audio_path.glob("*.wav"):
            audio_files.append(str(file))
    
    return sorted(audio_files)

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def run_inference(video_path, audio_path, output_path, config_path, ckpt_path, 
                  inference_steps=20, guidance_scale=1.5, seed=None, enable_deepcache=True):
    """Run inference for a single video-audio pair"""
    
    if seed is None:
        seed = random.randint(1, 999999)
    
    # Create command
    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", config_path,
        "--inference_ckpt_path", ckpt_path,
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed)
    ]
    
    if enable_deepcache:
        cmd.append("--enable_deepcache")
    
    print(f"Running inference for:")
    print(f"  Video: {Path(video_path).name}")
    print(f"  Audio: {Path(audio_path).name}")
    print(f"  Output: {Path(output_path).name}")
    print(f"  Seed: {seed}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully generated: {Path(output_path).name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating {Path(output_path).name}:")
        print(f"  Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch inference for LatentSync")
    parser.add_argument("--video_dir", type=str, default="data/Video", 
                       help="Directory containing video templates")
    parser.add_argument("--audio_dir", type=str, default="data/Audio/spk_ganyu", 
                       help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="output/lipsync", 
                       help="Output directory for generated videos")
    parser.add_argument("--config_path", type=str, default="configs/unet/stage2_512.yaml",
                       help="Path to UNet config file")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/latentsync_unet.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--inference_steps", type=int, default=20,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.5,
                       help="Guidance scale")
    parser.add_argument("--enable_deepcache", action="store_true", default=True,
                       help="Enable DeepCache for faster inference")
    parser.add_argument("--max_combinations", type=int, default=None,
                       help="Maximum number of video-audio combinations to process")
    parser.add_argument("--random_seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--allow_repeat_videos", action="store_true", default=True,
                       help="Allow the same video to be selected multiple times")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    # Get video and audio files
    print("Scanning for video and audio files...")
    video_files = get_video_files(args.video_dir)
    audio_files = get_audio_files(args.audio_dir)
    
    print(f"Found {len(video_files)} video files:")
    for video in video_files:
        print(f"  - {Path(video).name}")
    
    print(f"Found {len(audio_files)} audio files:")
    for audio in audio_files:
        print(f"  - {Path(audio).name}")
    
    if not video_files:
        print("No video files found!")
        return
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Calculate total combinations (one per audio file)
    total_combinations = len(audio_files)
    if args.max_combinations:
        total_combinations = min(total_combinations, args.max_combinations)
    
    print(f"Total combinations to process: {total_combinations}")
    print(f"Processing strategy: Randomly sample one video per audio file")
    
    # Process combinations - randomly sample video for each audio
    successful = 0
    failed = 0
    start_time = time.time()
    
    combination_count = 0
    
    # Create a list of audio files to process
    audio_files_to_process = audio_files.copy()
    
    # If max_combinations is specified, limit the number of audio files to process
    if args.max_combinations and args.max_combinations < len(audio_files):
        audio_files_to_process = random.sample(audio_files, args.max_combinations)
    
    total_combinations = len(audio_files_to_process)
    
    # Track used videos if we don't want repeats
    used_videos = set()
    
    for audio_file in audio_files_to_process:
        # Randomly sample a video file for this audio
        if not args.allow_repeat_videos and len(used_videos) >= len(video_files):
            print("Warning: All videos have been used. Reusing videos.")
            used_videos.clear()
        
        if args.allow_repeat_videos:
            sampled_video = random.choice(video_files)
        else:
            available_videos = [v for v in video_files if v not in used_videos]
            if not available_videos:
                print("Warning: No more unique videos available. Reusing videos.")
                used_videos.clear()
                available_videos = video_files
            sampled_video = random.choice(available_videos)
            used_videos.add(sampled_video)
        
        combination_count += 1
        
        # Create output filename
        video_name = Path(sampled_video).stem
        audio_name = Path(audio_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_name}_{audio_name}_{timestamp}.mp4"
        output_path = output_dir / output_filename
        
        print(f"\n[{combination_count}/{total_combinations}] Processing combination...")
        print(f"  Audio: {Path(audio_file).name}")
        print(f"  Randomly selected video: {Path(sampled_video).name}")
        
        # Run inference
        success = run_inference(
            video_path=sampled_video,
            audio_path=audio_file,
            output_path=str(output_path),
            config_path=args.config_path,
            ckpt_path=args.ckpt_path,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.random_seed,
            enable_deepcache=args.enable_deepcache
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_combination = elapsed_time / combination_count
        remaining_combinations = total_combinations - combination_count
        estimated_remaining_time = remaining_combinations * avg_time_per_combination
        
        print(f"Progress: {combination_count}/{total_combinations} "
              f"({combination_count/total_combinations*100:.1f}%)")
        print(f"Successful: {successful}, Failed: {failed}")
        print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
        print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"BATCH INFERENCE COMPLETED")
    print(f"{'='*50}")
    print(f"Total combinations processed: {combination_count}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/combination_count*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per combination: {total_time/combination_count:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 