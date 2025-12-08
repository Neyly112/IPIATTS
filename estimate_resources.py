"""
Estimate resource requirements for Matcha-TTS training
Helps users prepare hardware before running the pipeline
"""
import os
import json
from pathlib import Path


def format_bytes(bytes_num):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"


def estimate_audio_size(audio_dir="data/raw"):
    """Estimate total audio data size"""
    total_size = 0
    file_count = 0
    
    if os.path.exists(audio_dir):
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac')):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
    
    return total_size, file_count


def estimate_processing_requirements(audio_size_bytes, file_count):
    """Estimate disk and RAM for data processing"""
    # VAD: WAV files ~10x larger than MP3
    vad_size = audio_size_bytes * 10
    
    # Transcription: Creates subtitle files + cut segments
    # Assume average 5 segments per file, each ~5s, 16-bit WAV
    avg_segment_size = 5 * 22050 * 2  # 5s * 22.05kHz * 2 bytes
    segments_count = file_count * 5
    subs_size = segments_count * avg_segment_size
    
    # Text files are negligible
    text_size = segments_count * 200  # ~200 bytes per line
    
    # Total disk space needed
    total_disk = audio_size_bytes + vad_size + subs_size + text_size
    
    # Peak RAM during processing (Whisper model + audio buffers)
    whisper_ram = 3 * 1024**3  # ~3GB for Whisper large-v3
    audio_buffer_ram = max(vad_size * 0.1, 500 * 1024**2)  # 10% or min 500MB
    peak_processing_ram = whisper_ram + audio_buffer_ram
    
    return {
        'disk': {
            'raw_audio': audio_size_bytes,
            'vad_output': vad_size,
            'segments': subs_size,
            'text_files': text_size,
            'total': total_disk,
            'recommended': total_disk * 1.5  # 50% safety margin
        },
        'ram': {
            'whisper_model': whisper_ram,
            'audio_buffers': audio_buffer_ram,
            'peak': peak_processing_ram,
            'recommended': peak_processing_ram * 1.2  # 20% safety margin
        },
        'segments_estimate': segments_count
    }


def estimate_training_requirements(batch_size=1, accumulate_grad=4):
    """Estimate GPU VRAM and RAM for training"""
    
    # Model parameters
    text_encoder_params = 11.5e6  # 11.5M params
    phobert_params = 134e6  # 134M frozen params (vinai/phobert-base)
    prosody_fusion_params = 1.7e6  # ~1.7M params
    decoder_params = 50e6  # ~50M params
    total_params = text_encoder_params + phobert_params + prosody_fusion_params + decoder_params
    
    # Memory per parameter (FP16 mixed precision)
    bytes_per_param = 2  # FP16 = 2 bytes
    
    # Model weights
    model_memory = total_params * bytes_per_param
    
    # Optimizer state (Adam: 2x model params for momentum + variance)
    trainable_params = total_params - phobert_params  # PhoBERT is frozen
    optimizer_memory = trainable_params * bytes_per_param * 2
    
    # Gradients (same size as trainable params)
    gradient_memory = trainable_params * bytes_per_param
    
    # Activation memory (depends on batch size and sequence length)
    # Rough estimate: ~200MB per sample in batch
    activation_per_sample = 200 * 1024**2  # 200MB
    activation_memory = activation_per_sample * batch_size
    
    # Effective batch size (with gradient accumulation)
    effective_batch = batch_size * accumulate_grad
    
    # Total VRAM
    total_vram = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    # CUDA overhead (~500MB)
    cuda_overhead = 500 * 1024**2
    total_vram += cuda_overhead
    
    # RAM requirements
    # DataLoader: Keep 2-3 batches in RAM
    dataloader_ram = activation_per_sample * effective_batch * 3
    
    # System overhead
    system_overhead = 2 * 1024**3  # 2GB for Python, Lightning, etc.
    
    total_ram = dataloader_ram + system_overhead
    
    return {
        'vram': {
            'model_weights': model_memory,
            'optimizer_state': optimizer_memory,
            'gradients': gradient_memory,
            'activations': activation_memory,
            'cuda_overhead': cuda_overhead,
            'total': total_vram,
            'minimum_gpu': '4GB (GTX 1050, RTX 3050)',
            'recommended_gpu': '6GB+ (RTX 3060, RTX 4060)'
        },
        'ram': {
            'dataloader': dataloader_ram,
            'system': system_overhead,
            'total': total_ram,
            'recommended': total_ram * 1.3  # 30% safety margin
        },
        'model_info': {
            'total_params': f"{total_params/1e6:.1f}M",
            'trainable_params': f"{trainable_params/1e6:.1f}M",
            'frozen_params': f"{phobert_params/1e6:.1f}M (PhoBERT)",
            'batch_size': batch_size,
            'accumulate_grad_batches': accumulate_grad,
            'effective_batch_size': effective_batch
        }
    }


def estimate_checkpoint_size(num_epochs=100):
    """Estimate checkpoint storage requirements"""
    # Each checkpoint: model state + optimizer state + metadata
    checkpoint_size = 300 * 1024**2  # ~300MB per checkpoint
    
    # Lightning saves: last.ckpt + epoch checkpoints
    # Assume saving every 10 epochs + last
    num_checkpoints = (num_epochs // 10) + 1
    
    total_checkpoint_storage = checkpoint_size * num_checkpoints
    
    # TensorBoard logs
    tensorboard_size = 100 * 1024**2 * (num_epochs / 10)  # ~10MB per 10 epochs
    
    return {
        'checkpoint_size': checkpoint_size,
        'num_checkpoints': num_checkpoints,
        'total_checkpoints': total_checkpoint_storage,
        'tensorboard_logs': tensorboard_size,
        'total': total_checkpoint_storage + tensorboard_size
    }


def print_report():
    """Print comprehensive resource estimation report"""
    print("=" * 70)
    print("MATCHA-TTS RESOURCE ESTIMATION REPORT")
    print("=" * 70)
    print()
    
    # 1. Audio data analysis
    print("📁 AUDIO DATA ANALYSIS")
    print("-" * 70)
    audio_size, file_count = estimate_audio_size()
    print(f"Raw audio files: {file_count} files, {format_bytes(audio_size)}")
    
    if file_count == 0:
        print("⚠️  No audio files found in data/raw/")
        print("   Place your MP3/WAV files there before running the pipeline")
    print()
    
    # 2. Data processing requirements
    print("🔄 DATA PROCESSING REQUIREMENTS")
    print("-" * 70)
    processing = estimate_processing_requirements(audio_size, max(file_count, 10))
    
    print(f"Estimated output segments: ~{processing['segments_estimate']:,} audio clips")
    print()
    print("Disk Space Needed:")
    print(f"  • Raw audio:        {format_bytes(processing['disk']['raw_audio'])}")
    print(f"  • VAD output:       {format_bytes(processing['disk']['vad_output'])}")
    print(f"  • Cut segments:     {format_bytes(processing['disk']['segments'])}")
    print(f"  • Text files:       {format_bytes(processing['disk']['text_files'])}")
    print(f"  ─────────────────────────────────")
    print(f"  • Total needed:     {format_bytes(processing['disk']['total'])}")
    print(f"  • RECOMMENDED:      {format_bytes(processing['disk']['recommended'])} (with 50% margin)")
    print()
    print("RAM During Processing:")
    print(f"  • Whisper model:    {format_bytes(processing['ram']['whisper_model'])}")
    print(f"  • Audio buffers:    {format_bytes(processing['ram']['audio_buffer_ram'])}")
    print(f"  ─────────────────────────────────")
    print(f"  • Peak usage:       {format_bytes(processing['ram']['peak'])}")
    print(f"  • RECOMMENDED:      {format_bytes(processing['ram']['recommended'])} RAM minimum")
    print()
    
    # 3. Training requirements
    print("🚀 TRAINING REQUIREMENTS")
    print("-" * 70)
    
    # Read config from train script
    batch_size = 1
    accumulate_grad = 4
    num_epochs = 100
    
    if os.path.exists("train_matcha_prosody.py"):
        try:
            with open("train_matcha_prosody.py", 'r', encoding='utf-8') as f:
                content = f.read()
                if '"batch_size":' in content:
                    for line in content.split('\n'):
                        if '"batch_size":' in line and '//' not in line and '#' not in line:
                            batch_size = int(line.split(':')[1].strip().rstrip(','))
                        if '"accumulate_grad_batches":' in line:
                            accumulate_grad = int(line.split(':')[1].strip().rstrip(','))
                        if '"max_epochs":' in line:
                            num_epochs = int(line.split(':')[1].strip().rstrip(','))
        except Exception:
            pass
    
    training = estimate_training_requirements(batch_size, accumulate_grad)
    
    print(f"Model Configuration:")
    print(f"  • Total parameters:     {training['model_info']['total_params']}")
    print(f"  • Trainable params:     {training['model_info']['trainable_params']}")
    print(f"  • Frozen params:        {training['model_info']['frozen_params']}")
    print(f"  • Batch size:           {training['model_info']['batch_size']}")
    print(f"  • Gradient accumulation: {training['model_info']['accumulate_grad_batches']}")
    print(f"  • Effective batch size: {training['model_info']['effective_batch_size']}")
    print()
    print("GPU VRAM Needed:")
    print(f"  • Model weights:    {format_bytes(training['vram']['model_weights'])}")
    print(f"  • Optimizer state:  {format_bytes(training['vram']['optimizer_state'])}")
    print(f"  • Gradients:        {format_bytes(training['vram']['gradients'])}")
    print(f"  • Activations:      {format_bytes(training['vram']['activations'])}")
    print(f"  • CUDA overhead:    {format_bytes(training['vram']['cuda_overhead'])}")
    print(f"  ─────────────────────────────────")
    print(f"  • TOTAL VRAM:       {format_bytes(training['vram']['total'])}")
    print(f"  • Minimum GPU:      {training['vram']['minimum_gpu']}")
    print(f"  • Recommended GPU:  {training['vram']['recommended_gpu']}")
    print()
    print("System RAM During Training:")
    print(f"  • DataLoader:       {format_bytes(training['ram']['dataloader'])}")
    print(f"  • System overhead:  {format_bytes(training['ram']['system'])}")
    print(f"  ─────────────────────────────────")
    print(f"  • Total needed:     {format_bytes(training['ram']['total'])}")
    print(f"  • RECOMMENDED:      {format_bytes(training['ram']['recommended'])} RAM minimum")
    print()
    
    # 4. Checkpoint storage
    print("💾 CHECKPOINT STORAGE")
    print("-" * 70)
    checkpoints = estimate_checkpoint_size(num_epochs)
    print(f"Training for {num_epochs} epochs:")
    print(f"  • Per checkpoint:   {format_bytes(checkpoints['checkpoint_size'])}")
    print(f"  • Num checkpoints:  ~{checkpoints['num_checkpoints']}")
    print(f"  • Checkpoints total: {format_bytes(checkpoints['total_checkpoints'])}")
    print(f"  • TensorBoard logs: {format_bytes(checkpoints['tensorboard_logs'])}")
    print(f"  ─────────────────────────────────")
    print(f"  • TOTAL STORAGE:    {format_bytes(checkpoints['total'])}")
    print()
    
    # 5. Summary
    print("=" * 70)
    print("📊 HARDWARE RECOMMENDATIONS SUMMARY")
    print("=" * 70)
    
    total_disk_needed = processing['disk']['recommended'] + checkpoints['total']
    
    print()
    print("MINIMUM REQUIREMENTS:")
    print(f"  ✓ GPU:  4GB VRAM (GTX 1050, RTX 3050)")
    print(f"  ✓ RAM:  {format_bytes(max(processing['ram']['recommended'], training['ram']['recommended']))}")
    print(f"  ✓ Disk: {format_bytes(total_disk_needed)}")
    print()
    print("RECOMMENDED SETUP:")
    print(f"  ✓ GPU:  6GB+ VRAM (RTX 3060, RTX 4060)")
    print(f"  ✓ RAM:  16GB DDR4")
    print(f"  ✓ Disk: {format_bytes(total_disk_needed * 1.2)} SSD (20% extra margin)")
    print()
    print("⏱️  ESTIMATED TIME:")
    print(f"  • Data processing:  1-3 hours (depends on audio length)")
    print(f"  • Training (100 epochs): 12-48 hours (depends on GPU)")
    print()
    print("=" * 70)
    print()
    print("TIP: To reduce VRAM usage, modify train_matcha_prosody.py:")
    print('  - Decrease batch_size (currently: {})'.format(batch_size))
    print('  - Increase accumulate_grad_batches (currently: {})'.format(accumulate_grad))
    print('  - Disable FP16: precision="32" (slower but less VRAM)')
    print()
    
    # Save to JSON
    report = {
        'audio_data': {
            'file_count': file_count,
            'total_size_bytes': audio_size,
            'total_size_readable': format_bytes(audio_size)
        },
        'processing': processing,
        'training': training,
        'checkpoints': checkpoints,
        'summary': {
            'minimum_gpu_vram': format_bytes(training['vram']['total']),
            'minimum_ram': format_bytes(max(processing['ram']['recommended'], training['ram']['recommended'])),
            'minimum_disk': format_bytes(total_disk_needed),
            'recommended_gpu': training['vram']['recommended_gpu'],
            'recommended_ram': '16GB',
            'recommended_disk': format_bytes(total_disk_needed * 1.2)
        }
    }
    
    with open('resource_estimation.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📝 Full report saved to: resource_estimation.json")
    print()


if __name__ == "__main__":
    print_report()
