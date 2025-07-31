# Batch Inference for LatentSync

This directory contains scripts for batch processing video templates and audio files to generate lip sync videos using LatentSync.

## Files

- `batch_inference.py` - Main Python script for batch inference
- `batch_inference.sh` - Shell script wrapper for easy execution
- `BATCH_INFERENCE_README.md` - This documentation file

## Quick Start

### 1. Basic Usage

Run batch inference with default parameters:

```bash
./batch_inference.sh
```

This will:
- Process all audio files in `data/Audio/spk_ganyu/`
- For each audio file, randomly sample one video from `data/Video/`
- Generate lip sync videos in `output/lipsync/`
- Use default inference parameters (20 steps, guidance scale 1.5)

### 2. Limited Processing

Process only a limited number of combinations:

```bash
./batch_inference.sh --max_combinations 10
```

### 3. Higher Quality Settings

Use higher quality settings for better results:

```bash
./batch_inference.sh --inference_steps 30 --guidance_scale 2.0
```

### 4. Custom Directories

Use custom input/output directories:

```bash
./batch_inference.sh --video_dir "my_videos" --audio_dir "my_audio" --output_dir "my_output"
```

## Detailed Usage

### Python Script Direct Usage

You can also run the Python script directly with more control:

```bash
python batch_inference.py [OPTIONS]
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video_dir` | Directory containing video templates | `data/Video` |
| `--audio_dir` | Directory containing audio files | `data/Audio/spk_ganyu` |
| `--output_dir` | Output directory for generated videos | `output/lipsync` |
| `--config_path` | Path to UNet config file | `configs/unet/stage2_512.yaml` |
| `--ckpt_path` | Path to checkpoint file | `checkpoints/latentsync_unet.pt` |
| `--inference_steps` | Number of inference steps (20-50) | `20` |
| `--guidance_scale` | Guidance scale (1.0-3.0) | `1.5` |
| `--max_combinations` | Maximum number of combinations to process | `None` (all) |
| `--random_seed` | Random seed for reproducibility | `None` (random) |

### Shell Script Options

The shell script supports the same options:

```bash
./batch_inference.sh --help
```

## Examples

### Example 1: Test Run
Process only 5 combinations to test the setup:

```bash
./batch_inference.sh --max_combinations 5 --seed 0
```

### Example 2: High Quality Generation
Use higher quality settings for important videos:

```bash
./batch_inference.sh --inference_steps 40 --guidance_scale 2.5
```

### Example 3: Reproducible Results
Use a fixed seed for reproducible results:

```bash
./batch_inference.sh --random_seed 42
```

### Example 4: Custom Configuration
Use custom model configuration:

```bash
./batch_inference.sh --config_path "configs/unet/stage2.yaml" --ckpt_path "checkpoints/my_model.pt"
```

## Output

### File Naming Convention

Generated videos follow this naming pattern:
```
{video_name}_{audio_name}_{timestamp}.mp4
```

Example: `w2_00173_359_20241201_143022.mp4`

### Output Directory Structure

```
output/lipsync/
├── w2_00173_359_20241201_143022.mp4  # Random video + audio 359
├── w2_00171_358_20241201_143045.mp4  # Random video + audio 358
├── w2_00167_357_20241201_143108.mp4  # Random video + audio 357
└── ...
```

## Performance Considerations

### Memory Requirements

- **Minimum VRAM**: 8GB for LatentSync 1.5, 18GB for LatentSync 1.6
- **Recommended**: 24GB+ for batch processing

### Processing Time

- **Per combination**: 30-120 seconds (depending on settings)
- **Total time**: Depends on number of audio files
- **Example**: 100 audio files = 100 combinations ≈ 1-3 hours

### Optimization Tips

1. **Use DeepCache**: Enabled by default for faster inference
2. **Lower inference steps**: Use 20 steps for faster processing
3. **Limit combinations**: Use `--max_combinations` for testing
4. **GPU memory**: Monitor GPU memory usage and adjust batch size if needed

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `inference_steps` to 20
   - Lower `guidance_scale` to 1.0
   - Process fewer combinations at once

2. **Missing Files**
   - Check that video and audio directories exist
   - Verify file formats (MP4 for videos, WAV for audio)
   - Ensure checkpoint files are downloaded

3. **CUDA Errors**
   - Check GPU compatibility
   - Update CUDA drivers
   - Try running with CPU fallback (not recommended for batch processing)

### Logs and Monitoring

The script provides detailed progress information:
- Current combination being processed
- Success/failure counts
- Time estimates
- Error messages for failed generations

## Advanced Usage

### Processing Strategy

The batch inference uses a random sampling strategy:
- **One video per audio**: Each audio file gets paired with a randomly selected video
- **Efficient processing**: Reduces total combinations from N×M to N (where N=audio files, M=video files)
- **Good coverage**: Still provides diverse video-audio combinations
- **Reproducible**: Use `--random_seed` for consistent random sampling

### Custom Scripts

You can create custom batch processing scripts by modifying `batch_inference.py`:

```python
# Example: Process only specific audio files
audio_files = ["data/Audio/spk_ganyu/359.wav", "data/Audio/spk_ganyu/358.wav"]
```

### Integration with Other Tools

The generated videos can be further processed:
- Video editing software
- Quality assessment tools
- Automated testing frameworks

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- All LatentSync dependencies (see `requirements.txt`)
- Checkpoint files downloaded (see setup instructions)

## Support

For issues with the batch inference script:
1. Check the error messages in the console output
2. Verify all dependencies are installed
3. Ensure checkpoint files are properly downloaded
4. Check GPU memory availability 
