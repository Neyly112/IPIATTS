"""
Debug script to diagnose noise output issues in Matcha-TTS
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import json

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint for issues"""
    print("="*70)
    print("ANALYZING CHECKPOINT")
    print("="*70)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check state dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"\n✓ Found state_dict with {len(state_dict)} keys")
        
        # Check for NaN/Inf in weights
        nan_keys = []
        inf_keys = []
        zero_keys = []
        
        for key, value in state_dict.items():
            if torch.isnan(value).any():
                nan_keys.append(key)
            if torch.isinf(value).any():
                inf_keys.append(key)
            if torch.all(value == 0):
                zero_keys.append(key)
        
        if nan_keys:
            print(f"\n❌ FOUND NaN IN WEIGHTS:")
            for key in nan_keys[:5]:
                print(f"   - {key}")
        
        if inf_keys:
            print(f"\n❌ FOUND Inf IN WEIGHTS:")
            for key in inf_keys[:5]:
                print(f"   - {key}")
        
        if zero_keys:
            print(f"\n⚠️  FOUND ALL-ZERO WEIGHTS ({len(zero_keys)} tensors):")
            for key in zero_keys[:5]:
                print(f"   - {key}")
        
        if not (nan_keys or inf_keys):
            print("\n✓ No NaN/Inf detected in weights")
    
    # Check training metrics
    if 'epoch' in ckpt:
        print(f"\n📊 Training Progress:")
        print(f"   - Epoch: {ckpt['epoch']}")
        print(f"   - Global step: {ckpt.get('global_step', 'N/A')}")
    
    if 'callbacks' in ckpt:
        callbacks = ckpt['callbacks']
        if 'ModelCheckpoint' in callbacks:
            monitor_info = callbacks['ModelCheckpoint']
            print(f"\n📈 Best Metrics:")
            for key, value in monitor_info.items():
                if isinstance(value, (int, float)):
                    print(f"   - {key}: {value}")
    
    return {
        'has_nan': len(nan_keys) > 0,
        'has_inf': len(inf_keys) > 0,
        'has_zeros': len(zero_keys) > 0,
        'nan_keys': nan_keys,
        'inf_keys': inf_keys,
        'zero_keys': zero_keys
    }


def analyze_audio(audio_path):
    """Analyze generated audio for noise patterns"""
    print("\n" + "="*70)
    print("ANALYZING AUDIO FILE")
    print("="*70)
    
    audio, sr = sf.read(audio_path)
    
    print(f"\n📊 Audio Properties:")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Duration: {len(audio)/sr:.2f} seconds")
    print(f"   - Samples: {len(audio):,}")
    print(f"   - Channels: {'Stereo' if audio.ndim > 1 else 'Mono'}")
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis:")
    print(f"   - Min value: {np.min(audio):.6f}")
    print(f"   - Max value: {np.max(audio):.6f}")
    print(f"   - Mean: {np.mean(audio):.6f}")
    print(f"   - Std dev: {np.std(audio):.6f}")
    print(f"   - RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Check for issues
    issues = []
    
    # Check if audio is all zeros
    if np.all(audio == 0):
        issues.append("❌ Audio is completely silent (all zeros)")
    
    # Check if audio is clipping
    if np.max(np.abs(audio)) >= 0.99:
        issues.append("⚠️  Audio is clipping (values near ±1.0)")
    
    # Check if audio is very quiet
    if np.max(np.abs(audio)) < 0.01:
        issues.append("⚠️  Audio is very quiet (max < 0.01)")
    
    # Check for NaN/Inf
    if np.isnan(audio).any():
        issues.append("❌ Audio contains NaN values")
    if np.isinf(audio).any():
        issues.append("❌ Audio contains Inf values")
    
    # Check if it's just noise (high variance, no structure)
    # Real speech has some temporal correlation
    autocorr = np.correlate(audio[:min(1000, len(audio))], 
                            audio[:min(1000, len(audio))], mode='same')
    if np.max(autocorr[1:]) / autocorr[len(autocorr)//2] < 0.1:
        issues.append("⚠️  Audio looks like random noise (low autocorrelation)")
    
    print(f"\n🔍 Detected Issues:")
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✓ No obvious issues detected")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Waveform
    time = np.arange(len(audio)) / sr
    axes[0].plot(time, audio, linewidth=0.5)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    from scipy import signal
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
    axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_title('Spectrogram')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylim([0, 8000])
    
    # Histogram
    axes[2].hist(audio, bins=100, alpha=0.7, edgecolor='black')
    axes[2].set_title('Amplitude Distribution')
    axes[2].set_xlabel('Amplitude')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = str(Path(audio_path).parent / f"{Path(audio_path).stem}_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()
    
    return issues


def check_training_logs(log_dir="outputs/matcha_prosody"):
    """Check training logs for issues"""
    print("\n" + "="*70)
    print("CHECKING TRAINING LOGS")
    print("="*70)
    
    log_dir = Path(log_dir)
    
    # Check for tensorboard events
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    if event_files:
        print(f"\n✓ Found {len(event_files)} TensorBoard event files")
        print("   Run to view training curves:")
        print(f"   tensorboard --logdir {log_dir}")
    else:
        print("\n⚠️  No TensorBoard logs found")
    
    # Check for csv logs
    csv_files = list(log_dir.glob("**/*.csv"))
    if csv_files:
        print(f"\n✓ Found {len(csv_files)} CSV log files:")
        for csv_file in csv_files[:3]:
            print(f"   - {csv_file}")
    
    # Check checkpoints directory
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob("*.ckpt"))
        print(f"\n✓ Found {len(checkpoints)} checkpoints")
        if checkpoints:
            print("   Latest checkpoints:")
            for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-3:]:
                size_mb = ckpt.stat().st_size / (1024**2)
                print(f"   - {ckpt.name} ({size_mb:.1f} MB)")


def diagnose_common_issues():
    """Print common issues and solutions"""
    print("\n" + "="*70)
    print("COMMON ISSUES & SOLUTIONS")
    print("="*70)
    
    issues = {
        "1. Model not trained enough": [
            "- Check if training actually started (not just sanity check)",
            "- Look at loss curves in TensorBoard",
            "- Typical training needs 10k+ steps to produce intelligible speech",
            "Solution: Train longer (at least 20-30 epochs)"
        ],
        "2. Data preprocessing issues": [
            "- Check if audio files are properly normalized",
            "- Verify phonemes are correctly generated",
            "- Check data_statistics.json exists and is used",
            "Solution: Regenerate data with scripts/cleaner.py"
        ],
        "3. Model diverged (NaN/Inf in weights)": [
            "- Loss suddenly jumped to NaN",
            "- Gradient explosion",
            "Solution: Reduce learning rate, check data normalization"
        ],
        "4. Wrong checkpoint loaded": [
            "- Loading 'last.ckpt' instead of best checkpoint",
            "- Checkpoint from early training (random weights)",
            "Solution: Use checkpoint with lowest val_loss"
        ],
        "5. Vocoder (HiFi-GAN) issues": [
            "- Mel-spectrogram is correct but vocoder produces noise",
            "- Vocoder checkpoint mismatch",
            "Solution: Re-download HiFi-GAN checkpoint"
        ],
        "6. Inference parameters wrong": [
            "- Temperature too high (random sampling)",
            "- Wrong n_timesteps for diffusion",
            "Solution: Use temperature=0.667, n_timesteps=10"
        ]
    }
    
    for issue, solutions in issues.items():
        print(f"\n{issue}")
        for solution in solutions:
            print(f"  {solution}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug Matcha-TTS noise output")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--audio", type=str, help="Path to generated audio file")
    parser.add_argument("--log-dir", type=str, default="outputs/matcha_prosody", 
                        help="Path to training logs")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MATCHA-TTS OUTPUT NOISE DEBUGGER")
    print("="*70)
    
    # Analyze checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint_issues = analyze_checkpoint(args.checkpoint)
    elif Path("outputs/matcha_prosody/checkpoints/last.ckpt").exists():
        print("\n[Auto-detected checkpoint: last.ckpt]")
        checkpoint_issues = analyze_checkpoint("outputs/matcha_prosody/checkpoints/last.ckpt")
    else:
        print("\n⚠️  No checkpoint specified or found")
        checkpoint_issues = None
    
    # Analyze audio
    if args.audio and Path(args.audio).exists():
        audio_issues = analyze_audio(args.audio)
    elif Path("outputs/test_samples").exists():
        samples = list(Path("outputs/test_samples").glob("*.wav"))
        if samples:
            print(f"\n[Auto-detected audio: {samples[0]}]")
            audio_issues = analyze_audio(str(samples[0]))
        else:
            print("\n⚠️  No audio files found in outputs/test_samples")
            audio_issues = None
    else:
        print("\n⚠️  No audio file specified or found")
        audio_issues = None
    
    # Check logs
    check_training_logs(args.log_dir)
    
    # Show common issues
    diagnose_common_issues()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if checkpoint_issues:
        if checkpoint_issues['has_nan'] or checkpoint_issues['has_inf']:
            print("\n❌ CRITICAL: Checkpoint has NaN/Inf weights - model diverged!")
            print("   ACTION: Restart training with lower learning rate")
        elif checkpoint_issues['has_zeros']:
            print("\n⚠️  WARNING: Some weights are all zeros")
            print("   ACTION: Check if training actually ran")
    
    if audio_issues:
        print("\n🔍 Audio Issues Detected:")
        for issue in audio_issues:
            print(f"   {issue}")
    
    print("\n" + "="*70)
    print("\nNEXT STEPS:")
    print("1. Check training curves: tensorboard --logdir outputs/matcha_prosody")
    print("2. Verify training ran: check epoch number in checkpoint")
    print("3. Try loading best checkpoint (lowest val_loss) instead of last.ckpt")
    print("4. If trained < 10k steps: continue training")
    print("5. Check data: python scripts/check_data.py --filelist data/99-audio-text-file-list/audio_text_train.txt.cleaned")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
