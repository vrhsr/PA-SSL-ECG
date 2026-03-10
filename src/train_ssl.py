"""
PA-SSL: Self-Supervised Pretraining Script

Trains an encoder using the combined contrastive objective:
  L = α · L_augmentation + β · L_temporal

Supports both ResNet1D and WavKAN encoders, physiology-aware and naive
augmentation pipelines, with WandB/TensorBoard logging.

Usage:
    python -m src.train_ssl --encoder resnet1d --augmentation physio --epochs 100
    python -m src.train_ssl --encoder wavkan --augmentation naive --epochs 100
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import argparse
import os
import time
import json
from tqdm import tqdm

from src.data.ecg_dataset import ECGBeatDataset, SSLECGDataset
from src.models.encoder import build_encoder
from src.losses import CombinedContrastiveLoss
from src.augmentations.augmentation_pipeline import PhysioAugPipeline
from src.augmentations.naive_augmentations import NaiveAugPipeline


def ssl_collate_fn(batch):
    """Custom collate for SSLECGDataset that handles temporal views."""
    view1 = torch.stack([item['view1'] for item in batch])
    view2 = torch.stack([item['view2'] for item in batch])
    
    result = {'view1': view1, 'view2': view2}
    
    if 'metadata' in batch[0]:
        result['metadata'] = torch.stack([item['metadata'] for item in batch])
    
    if 'temporal_view' in batch[0]:
        result['temporal_view'] = torch.stack([item['temporal_view'] for item in batch])
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PA-SSL Pretraining")
    print(f"  Device: {device}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Temporal positives: {args.use_temporal}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # ─── Data ─────────────────────────────────────────────────────────────
    base_dataset = ECGBeatDataset(args.data_file)
    
    # Augmentation pipeline
    if args.augmentation == 'physio':
        aug_pipeline = PhysioAugPipeline.default(strength=args.aug_strength)
        print(f"Using Physiology-Aware Augmentations ({args.aug_strength})")
        print(aug_pipeline)
    elif args.augmentation == 'naive':
        aug_pipeline = NaiveAugPipeline(p=0.5)
        print("Using Naive Augmentations (baseline)")
    else:
        aug_pipeline = None
        print("No augmentations (identity baseline)")
    
    ssl_dataset = SSLECGDataset(
        base_dataset,
        augmentation_pipeline=aug_pipeline,
        use_temporal_positives=args.use_temporal,
    )
    
    # PyTorch DataLoader optimization for Windows: Keep workers alive across epochs
    # to prevent the massive delay when recreating processes CPU-side.
    prefetch = max(2, 16 // max(1, args.num_workers)) if args.num_workers > 0 else None
    
    loader = DataLoader(
        ssl_dataset,
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
    
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {n_params:,}")
    
    # ─── Optimizer ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    )
    
    # ─── Mixed Precision ──────────────────────────────────────────────────
    scaler = GradScaler() if args.amp else None
    
    # ─── Output Directory ─────────────────────────────────────────────────
    exp_name = f"ssl_{args.encoder}_{args.augmentation}"
    if args.use_temporal:
        exp_name += "_temporal"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['n_params'] = n_params
    config['n_samples'] = len(ssl_dataset)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # ─── Training Loop ────────────────────────────────────────────────────
    best_loss = float('inf')
    history = []
    
    for epoch in range(args.epochs):
        encoder.train()
        running_loss = 0.0
        running_loss_aug = 0.0
        running_loss_temp = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)
            
            temporal_view = None
            has_temporal = None
            metadata = None
            
            if 'metadata' in batch and args.use_metadata:
                metadata = batch['metadata'].to(device)
                
            if 'temporal_view' in batch:
                temporal_view = batch['temporal_view'].to(device)
                has_temporal = batch['has_temporal'].to(device)
            
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    z1 = encoder(view1, return_projection=True, metadata=metadata)
                    z2 = encoder(view2, return_projection=True, metadata=metadata)
                    z_temp = encoder(temporal_view, return_projection=True, metadata=metadata) if temporal_view is not None else None
                    
                    loss, loss_aug, loss_temp = criterion(z1, z2, z_temp, has_temporal)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = encoder(view1, return_projection=True, metadata=metadata)
                z2 = encoder(view2, return_projection=True, metadata=metadata)
                z_temp = encoder(temporal_view, return_projection=True, metadata=metadata) if temporal_view is not None else None
                
                loss, loss_aug, loss_temp = criterion(z1, z2, z_temp, has_temporal)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            running_loss_aug += loss_aug.item()
            running_loss_temp += loss_temp.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'aug': f'{loss_aug.item():.4f}',
                'temp': f'{loss_temp.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            })
        
        scheduler.step()
        
        # Epoch stats
        n_batches = len(loader)
        epoch_loss = running_loss / n_batches
        epoch_loss_aug = running_loss_aug / n_batches
        epoch_loss_temp = running_loss_temp / n_batches
        
        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'loss_aug': epoch_loss_aug,
            'loss_temporal': epoch_loss_temp,
            'lr': scheduler.get_last_lr()[0],
        })
        
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f} "
              f"(aug={epoch_loss_aug:.4f}, temp={epoch_loss_temp:.4f})")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'config': config,
            }, os.path.join(exp_dir, 'best_checkpoint.pth'))
            print(f"  > New best model saved (loss={best_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(exp_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    # Save training history
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
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
                        choices=['resnet1d', 'wavkan'])
    parser.add_argument('--proj_dim', type=int, default=128)
    
    # Augmentation
    parser.add_argument('--augmentation', type=str, default='physio',
                        choices=['physio', 'naive', 'none'])
    parser.add_argument('--aug_strength', type=str, default='medium',
                        choices=['light', 'medium', 'strong'])
    parser.add_argument('--use_temporal', action='store_true', default=True)
    parser.add_argument('--no_temporal', dest='use_temporal', action='store_false')
    parser.add_argument('--use_metadata', action='store_true', default=False,
                        help='Condition projection head on patient demography (Phase 9)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for augmentation contrastive loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for temporal contrastive loss')
    
    # Performance
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Quick test
    parser.add_argument('--quick_test', action='store_true',
                        help='Run 2 epochs for smoke testing')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 2
        args.batch_size = 32
    
    train_ssl(args)
