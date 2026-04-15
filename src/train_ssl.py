"""
PA-SSL: Self-Supervised Pretraining Script

Trains an encoder using the combined contrastive objective:
  L = alpha * L_augmentation + beta * L_temporal

Supports both ResNet1D and WavKAN encoders, physiology-aware and naive
augmentation pipelines, with WandB/TensorBoard logging.

Usage:
    python -m src.train_ssl --encoder resnet1d --augmentation physio --epochs 100
    python -m src.train_ssl --encoder wavkan --augmentation naive --epochs 100
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import argparse
import os
import sys
import time
import json
from tqdm import tqdm

from src.data.ecg_dataset import ECGBeatDataset, SSLECGDataset
from src.models.encoder import build_encoder
from src.models.mae import HybridMAE
from src.losses import CombinedContrastiveLoss
from src.augmentations.augmentation_pipeline import PhysioAugPipeline
from src.augmentations.naive_augmentations import NaiveAugPipeline
from src.augmentations.gpu_augmentations import get_gpu_augmentations


def ssl_collate_fn(batch):
    """Custom collate for SSLECGDataset that handles raw views for GPU-accel."""
    view1 = torch.stack([item['view1'] for item in batch])
    r_peak = torch.stack([item['r_peak'] for item in batch])
    
    result = {'view1': view1, 'r_peak': r_peak}
    
    if 'metadata' in batch[0]:
        result['metadata'] = torch.stack([item['metadata'] for item in batch])
    
    if 'temporal_view1' in batch[0]:
        result['temporal_view1'] = torch.stack([item['temporal_view1'] for item in batch])
        result['has_temporal'] = torch.tensor([item['has_temporal'] for item in batch])
    
    return result


def train_ssl(args):
    """Main SSL training loop."""
    # Set random seed for reproducibility
    if hasattr(args, 'seed') and args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(f"  Random seed: {args.seed}")
    
    # Performance: Enable TF32 for Ampere+ GPUs (like A4000)
    if torch.cuda.is_available() and sys.platform != "win32":
        print("  Enabling TF32 (TensorFloat-32) for hardware-accelerated matrix math")
        torch.set_float32_matmul_precision('high')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PA-SSL Pretraining")
    print(f"  Device: {device}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Temporal positives: {args.use_temporal}")
    if args.use_temporal:
        print(f"  Temporal scales: {args.temporal_scales}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # ─── Data ─────────────────────────────────────────────────────────────
    base_dataset = ECGBeatDataset(args.data_file)
    
    # Augmentation pipeline
    if args.augmentation == 'physio':
        aug_pipeline = PhysioAugPipeline.default(
            strength=args.aug_strength,
            exclude=args.exclude_aug,
            only=args.only_aug,
            qrs_protect=not args.no_qrs_protect
        )
        qrs_str = '(NO QRS protection — ablation)' if args.no_qrs_protect else f'({args.aug_strength})'
        print(f"Using Physiology-Aware Augmentations {qrs_str}")
        print(aug_pipeline)
    elif args.augmentation == 'naive':
        aug_pipeline = NaiveAugPipeline(p=0.5)
        print("Using Naive Augmentations (baseline)")
    else:
        aug_pipeline = None
        print("No augmentations (identity baseline)")
      
    from src.data.ecg_dataset import ECGBeatDataset, SSLECGDataset, patient_aware_split
    
    # ─── Defensive Pretraining Split ───
    # Guarantee that test patients are NEVER seen during pretraining.
    if "ptbxl" in args.data_file.lower() and "combined" not in args.data_file.lower():
        print("  [Security] Detected raw PTB-XL pretraining. Splitting to hold out test patients...")
        train_df, val_df, _ = patient_aware_split(args.data_file, seed=42)
        safe_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        base_dataset = ECGBeatDataset(safe_df)
    else:
        # combined_pretrain.csv already has test patients cleanly excluded
        base_dataset = ECGBeatDataset(args.data_file)
        
    full_dataset = SSLECGDataset(
        base_dataset, 
        augmentation_pipeline=aug_pipeline,
        use_temporal_positives=args.use_temporal,
        temporal_scales=args.temporal_scales
    )
    
    # 11.7 Pretraining Scale: Use a fraction of the unlabeled data if requested
    if args.data_fraction < 1.0:
        n_scaled = int(len(full_dataset) * args.data_fraction)
        print(f"  Scaling: Using {args.data_fraction*100:.1f}% of data ({n_scaled}/{len(full_dataset)})")
        indices = np.random.choice(len(full_dataset), n_scaled, replace=False)
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    
    # PyTorch DataLoader optimization for Windows: Keep workers alive across epochs
    # to prevent the massive delay when recreating processes CPU-side.
    # Extreme Data Flow: 2x prefetch baseline for high-bandwidth training
    prefetch = max(4, 32 // max(1, args.num_workers)) if args.num_workers > 0 else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=prefetch,
        collate_fn=ssl_collate_fn,
    )
    
    # ─── Model ────────────────────────────────────────────────────────────
    metadata_dim = 4 if args.use_metadata else 0
    encoder = build_encoder(args.encoder, proj_dim=args.proj_dim, metadata_dim=metadata_dim).to(device)
    
    if args.ssl_mode in ['mae', 'hybrid']:
        model = HybridMAE(encoder, mask_ratio=args.mask_ratio, qrs_avoidance_prob=args.qrs_avoid_prob).to(device)
        model_params = model.parameters()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Hybrid MAE Model parameters: {n_params:,}")
    else:
        model = encoder
        model_params = encoder.parameters()
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"Encoder parameters: {n_params:,}")
    
    # GPU optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Optimization: torch.compile for Linux
    if sys.platform != "win32" and hasattr(torch, "compile"):
        print("  Enabling torch.compile (reduce-overhead) for high-throughput GPU utilization (Linux)")
        try:
            # "reduce-overhead" is optimal for repetitive batch shapes like ECG beats
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"  [WARN] torch.compile failed: {e}. Falling back to eager mode.")
    else:
        print("  Running in eager mode (torch.compile disabled for Windows/unsupported)")
    
    # ─── Optimizer ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True  # Reduce memory-bandwidth bottleneck during parameter updates
    )
    
    # Cosine LR schedule with warmup
    warmup_epochs = min(10, args.epochs // 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ─── Loss ─────────────────────────────────────────────────────────────
    criterion = CombinedContrastiveLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta if args.use_temporal else 0.0,
        loss_type=args.loss_type,
    )
    
    # ─── Mixed Precision ──────────────────────────────────────────────────
    scaler = GradScaler('cuda') if args.amp else None
    
    # ─── Output Directory ─────────────────────────────────────────────────
    exp_name = f"ssl_{args.encoder}_{args.augmentation}"
    if args.use_temporal:
        exp_name += "_temporal"
        
    if args.output_dir == 'experiments':
        exp_dir = os.path.join(args.output_dir, exp_name)
    else:
        exp_dir = args.output_dir
        
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['n_params'] = n_params
    config['n_samples'] = len(dataset) # Use 'dataset' which might be a Subset
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # ─── Training Loop ────────────────────────────────────────────────────
    best_loss = float('inf')
    history = []
    batch_history = []
    is_interactive = sys.stdout.isatty()  # Detect if running in a terminal vs redirected to file
    
    start_epoch = 0
    
    # Auto-resume logic
    if os.path.exists(exp_dir):
        # Find highest epoch checkpoint
        ckpt_files = [f for f in os.listdir(exp_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
        latest_ckpt = None
        if ckpt_files:
            epochs_saved = [int(f.replace('checkpoint_epoch', '').replace('.pth', '')) for f in ckpt_files]
            latest_epoch = max(epochs_saved)
            latest_ckpt = f'checkpoint_epoch{latest_epoch}.pth'
        elif os.path.exists(os.path.join(exp_dir, 'best_checkpoint.pth')):
            latest_ckpt = 'best_checkpoint.pth'
            
        if latest_ckpt is not None:
            ckpt_path = os.path.join(exp_dir, latest_ckpt)
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if args.ssl_mode in ['mae', 'hybrid']:
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                else:
                    # Fallback to load encoder only
                    model.encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt['encoder_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            
            # Fast-forward learning rate scheduler
            for _ in range(start_epoch):
                scheduler.step()
                
            if 'loss' in ckpt:
                best_loss = float(ckpt['loss'])
                
            # Attempt to load existing history
            history_path = os.path.join(exp_dir, 'history.json')
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        saved_history = json.load(f)
                        if isinstance(saved_history, dict):
                            history = saved_history.get('epochs', [])
                            batch_history = saved_history.get('batches', [])
                        else:
                            history = saved_history
                except Exception as e:
                    print(f"  Warning: Could not load history.json ({e})")
    
    # Compute Efficiency Logger (11.8)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Efficiency Summary:")
    print(f"  Total Parameters:     {total_params/1e6:.3f}M")
    print(f"  Trainable Parameters: {trainable_params/1e6:.3f}M")
    print(f"  Device:               {device}")
    
    # 2.4 Resumption/Stability Logging
    
    # ─── GPU Augmentation Setup ──────────────────────────────────────────
    gpu_aug = None
    if args.augmentation == 'physio':
        print(f"  Enabling GPU-Accelerated Physio-Augmentations (strength={args.aug_strength})")
        gpu_aug = get_gpu_augmentations(strength=args.aug_strength, device=device)
        
    print(f"\nStarting Pretraining ({args.epochs} epochs)...")
    start_time_total = time.time()
    
    is_interactive = sys.stdout.isatty()
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_loss_aug = 0.0
        running_loss_temp = 0.0
        running_loss_mae = 0.0
        n_total = len(loader)
        
        # MAE loss warmup: ramp λ from 0 → mae_weight over mae_warmup_epochs
        if args.ssl_mode == 'hybrid' and hasattr(args, 'mae_warmup_epochs') and args.mae_warmup_epochs > 0:
            mae_lambda = args.mae_weight * min(1.0, (epoch + 1) / args.mae_warmup_epochs)
        else:
            mae_lambda = args.mae_weight
        
        # Adaptive tqdm: 0.1s updates for live tmux, 10s for clean nohup logs
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=False,
                    mininterval=(0.1 if is_interactive else 10.0),
                    ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            # Async memory transfers of RAW data
            x_raw = batch['view1'].to(device, non_blocking=True)
            r_peaks = batch['r_peak'].to(device, non_blocking=True)
            
            # --- ON-DEVICE AUGMENTATION ---
            # Instead of the CPU doing this one-by-one, the GPU does the whole batch in parallel!
            if gpu_aug is not None:
                view1 = gpu_aug.apply_batch(x_raw.clone())
                view2 = gpu_aug.apply_batch(x_raw.clone())
            else:
                view1 = x_raw.clone()
                view2 = x_raw.clone()
                
            temporal_view = None
            has_temporal = None
            if 'temporal_view1' in batch:
                x_raw_temp = batch['temporal_view1'].to(device, non_blocking=True)
                if gpu_aug is not None:
                    temporal_view = gpu_aug.apply_batch(x_raw_temp)
                else:
                    temporal_view = x_raw_temp
                has_temporal = batch['has_temporal'].to(device, non_blocking=True)
            
            metadata = None
            if 'metadata' in batch and args.use_metadata:
                metadata = batch['metadata'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            def compute_batch_loss():
                loss, loss_aug, loss_temp, loss_mae = 0.0, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                
                # 1. Contrastive Path (Unified Forward Pass)
                if args.ssl_mode in ['contrastive', 'hybrid']:
                    enc_model = model.encoder if args.ssl_mode == 'hybrid' else model
                    
                    # Concat views for single forward pass (Industrial throughput trick)
                    views_to_cat = [view1, view2]
                    if temporal_view is not None:
                        views_to_cat.append(temporal_view)
                    
                    x_combined = torch.cat(views_to_cat, dim=0)
                    
                    # Align metadata if used
                    meta_combined = None
                    if metadata is not None:
                        meta_to_cat = [metadata] * len(views_to_cat)
                        meta_combined = torch.cat(meta_to_cat, dim=0)
                    
                    # Single forward call
                    z_all = enc_model(x_combined, return_projection=True, metadata=meta_combined)
                    
                    # Split back
                    z_chunks = torch.chunk(z_all, len(views_to_cat), dim=0)
                    z1, z2 = z_chunks[0], z_chunks[1]
                    z_temp = z_chunks[2] if len(z_chunks) > 2 else None
                    
                    loss_c, loss_aug, loss_temp = criterion(z1, z2, z_temp, has_temporal)
                    loss += loss_c
                
                # 2. MAE Path
                if args.ssl_mode in ['mae', 'hybrid']:
                    recon_x, masks, masked_x = model.forward_mae(view1)
                    # MSE only on masked patches
                    masks_expanded = masks.unsqueeze(1).expand_as(view1)
                    if masks_expanded.sum() > 0:
                        loss_mae = torch.nn.functional.mse_loss(
                            recon_x[masks_expanded], view1[masks_expanded]
                        )
                    else:
                        loss_mae = torch.tensor(0.0, device=device)
                    
                    if args.ssl_mode == 'mae':
                        loss += loss_mae
                    else:
                        loss += mae_lambda * loss_mae
                        
                # Just in case loss is 0 (float), convert it assuming it has valid grads
                if isinstance(loss, float):
                    loss = torch.tensor(loss, device=device, requires_grad=True)
                    
                return loss, loss_aug, loss_temp, loss_mae
            
            if args.amp:
                with autocast('cuda'):
                    loss, loss_aug, loss_temp, loss_mae = compute_batch_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_aug, loss_temp, loss_mae = compute_batch_loss()
                loss.backward()
                optimizer.step()
            # Log batch for stability curves (every 10 batches to save memory)
            if batch_idx % 10 == 0:
                batch_history.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'loss_aug': loss_aug.item(),
                    'loss_mae': loss_mae.item()
                })
                
            running_loss += loss.item()
            running_loss_aug += loss_aug.item()
            running_loss_temp += loss_temp.item()
            running_loss_mae += loss_mae.item()
            
            if is_interactive:
                pbar_dict = {'loss': f'{loss.item():.4f}'}
                if args.ssl_mode in ['contrastive', 'hybrid']:
                    pbar_dict['aug'] = f'{loss_aug.item():.4f}'
                    if args.use_temporal: pbar_dict['tmp'] = f'{loss_temp.item():.4f}'
                if args.ssl_mode in ['mae', 'hybrid']:
                    pbar_dict['mae'] = f'{loss_mae.item():.4f}'
                pbar.set_postfix(pbar_dict)
        
        scheduler.step()
        
        # Epoch stats
        n_batches = len(loader)
        epoch_loss = running_loss / n_batches
        epoch_loss_aug = running_loss_aug / n_batches
        epoch_loss_temp = running_loss_temp / n_batches
        epoch_loss_mae = running_loss_mae / n_batches
        epoch_duration = time.time() - epoch_start_time
        
        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'loss_aug': epoch_loss_aug,
            'loss_temp': epoch_loss_temp,
            'loss_mae': epoch_loss_mae,
            'loss_temporal': epoch_loss_temp,
            'lr': scheduler.get_last_lr()[0],
            'epoch_duration': epoch_duration,
        })
        
        # Log basic stats
        print(f"Epoch {epoch+1} Complete | Loss: {epoch_loss:.4f} | Time: {epoch_duration:.1f}s "
              f"(aug={epoch_loss_aug:.4f}, temp={epoch_loss_temp:.4f})")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_data = {
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict() if args.ssl_mode in ['mae', 'hybrid'] else model.state_dict(),
                'model_state_dict': model.state_dict() if args.ssl_mode in ['mae', 'hybrid'] else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'args': vars(args)
            }
            torch.save(checkpoint_data, os.path.join(exp_dir, 'best_checkpoint.pth'))
            print(f"  > New best model saved (loss={best_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(exp_dir, f'checkpoint_epoch{epoch+1}.pth'))
            
        # Save training history iteratively
        with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
            json.dump({'epochs': history, 'batches': batch_history}, f, indent=2)

    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoint saved to: {exp_dir}")
    print(f"{'='*60}")
    
    return encoder, exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA-SSL Pretraining")
    
    # Data
    parser.add_argument('--data_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--output_dir', type=str, default='experiments')
    
    # Model
    parser.add_argument('--encoder', type=str, default='resnet1d',
                        choices=['resnet1d', 'wavkan', 'transformer', 'mamba'])
    parser.add_argument('--proj_dim', type=int, default=128)
    
    # Augmentation
    parser.add_argument('--augmentation', type=str, default='physio',
                        choices=['physio', 'naive', 'none'])
    parser.add_argument('--aug_strength', type=str, default='medium',
                        choices=['light', 'medium', 'strong'])
    parser.add_argument('--exclude_aug', nargs='+', type=str, default=None,
                        help='Specific augmentations to exclude (e.g., constrained_time_warp)')
    parser.add_argument('--only_aug', nargs='+', type=str, default=None,
                        help='Specific augmentations to exclusively include')
    parser.add_argument('--use_temporal', action='store_true', default=True)
    parser.add_argument('--no_temporal', dest='use_temporal', action='store_false')
    parser.add_argument('--temporal_scales', nargs='+', type=int, default=[1],
                        help='Number of beats away to sample temporal positives (e.g., 1 2 3)')
    parser.add_argument('--use_metadata', action='store_true', default=False,
                        help='Condition projection head on patient demography (Phase 9)')
    parser.add_argument('--no_qrs_protect', action='store_true', default=False,
                        help='Disable QRS-region protection in amplitude_perturbation (for ablation)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='ntxent',
                        choices=['ntxent', 'vicreg', 'barlow'],
                        help='Contrastive objective to use')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for augmentation contrastive loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for temporal contrastive loss')
    
    # Performance
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--compile', action='store_true', default=os.name != 'nt',
                        help='Use torch.compile for speed (Linux only)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # MAE & Hybrid Settings
    parser.add_argument('--ssl_mode', type=str, default='contrastive',
                        choices=['contrastive', 'mae', 'hybrid'],
                        help='Pretraining objective: Contrastive only, MAE only, or Hybrid')
    parser.add_argument('--mask_ratio', type=float, default=0.60,
                        help='Ratio of sequence to mask in MAE mode (0.60 recommended)')
    parser.add_argument('--qrs_avoid_prob', type=float, default=0.8,
                        help='Probability of strictly avoiding the QRS region during masking')
    parser.add_argument('--mae_weight', type=float, default=1.0,
                        help='Peak weight for MAE loss in Hybrid mode')
    parser.add_argument('--mae_warmup_epochs', type=int, default=10,
                        help='Number of epochs to linearly ramp MAE weight from 0 to mae_weight')
    
    # Execution Safety
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save a periodic checkpoint every N epochs')
    
    # Pretraining Scale
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of unlabeled data to use (0.0 to 1.0) for scaling experiments')
    
    # Quick test
    parser.add_argument('--quick_test', action='store_true',
                        help='Run 2 epochs for smoke testing')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches per epoch (for smoke test)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 2
        args.batch_size = 32
        if args.max_batches is None:
            args.max_batches = 10
    
    train_ssl(args)
