#!/usr/bin/env python3
"""
PyTorch training script for GLA Transformer on jet data
Based on aaron_train_transformer.py but implemented in PyTorch
"""

import os
import random
import sys
import time
import argparse
import logging
from typing import Tuple, Dict, Any
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Import our GLA transformer
from gla_transformer import GLATransformer, create_gla_model


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for the model (simplified calculation)
    This is an approximation - for exact FLOPs, use tools like ptflops or fvcore
    """
    try:
        from ptflops import get_model_complexity_info
        
        def input_constructor(input_shape):
            return torch.randn(input_shape)
            
        macs, params = get_model_complexity_info(
            model, input_shape[1:], 
            input_constructor=input_constructor,
            print_per_layer_stat=False,
            verbose=False
        )
        # Convert MACs to FLOPs (approximately 2 * MACs)
        flops = 2 * macs
        return flops, params
    except ImportError:
        print("Warning: ptflops not available. Install with 'pip install ptflops'")
        # Rough estimation based on model parameters
        num_params = sum(p.numel() for p in model.parameters())
        # Very rough approximation: assume ~2 ops per parameter per forward pass
        estimated_flops = num_params * 2
        return estimated_flops, num_params


def profile_gpu_memory_and_timing(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 5
) -> Tuple[float, float, float]:
    """
    Profile GPU memory usage and inference timing
    """
    model.eval()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    times = []
    
    with torch.no_grad():
        for i, (batch_x, batch_mask, _) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            
            # Timing
            start_time = time.perf_counter()
            _ = model(batch_x, batch_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            if i > 0:  # Skip first iteration (warmup)
                times.append((end_time - start_time) / batch_x.size(0))
    
    avg_time_per_sample = np.mean(times) * 1e9  # Convert to nanoseconds
    
    # GPU memory stats
    if device.type == 'cuda':
        current_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    else:
        current_memory = peak_memory = 0.0
    
    return current_memory, peak_memory, avg_time_per_sample


def apply_sorting(x: np.ndarray, sort_by: str) -> np.ndarray:
    """Apply sorting to particle features"""
    if sort_by == "pt":
        key = x[:, :, 0]
    elif sort_by == "eta":
        key = x[:, :, 1]  
    elif sort_by == "phi":
        key = x[:, :, 2]
    elif sort_by == "delta_R":
        key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    elif sort_by == "kt":
        key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    else:
        return x
    
    # Sort in descending order
    idx = np.argsort(key, axis=1)[:, ::-1]
    return np.take_along_axis(x, idx[:, :, None], axis=1)


def create_attention_mask(x: np.ndarray, padding_value: float = 0.0) -> np.ndarray:
    """
    Create attention mask based on non-zero entries (assuming padding is zero)
    """
    # Check if all features in a position are zero (padding)
    mask = ~np.all(np.isclose(x, padding_value), axis=-1)
    return mask.astype(np.float32)


def prepare_data_loaders(
    args,
    dataset: str,
    data_dir: str,
    num_particles: int,
    sort_by: str = "pt",
    val_split: float = 0.2,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare data loaders for training, validation, and testing
    """
    logging.info(f"Loading data for {dataset} dataset...")
    
    # Load data based on dataset type
    if dataset == "hls4ml":
        # Load HLS4ML data
        x_train = np.load(os.path.join(data_dir, f"x_train_robust_{num_particles}const_ptetaphi.npy"))
        y_train = np.load(os.path.join(data_dir, f"y_train_robust_{num_particles}const_ptetaphi.npy"))
        x_test = np.load(os.path.join(data_dir, f"x_val_robust_{num_particles}const_ptetaphi.npy"))
        y_test = np.load(os.path.join(data_dir, f"y_val_robust_{num_particles}const_ptetaphi.npy"))
        
        # Split training into train/val
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_split, random_state=42
        )
        
    else:  # jetclass, top, or QG
        x_train = np.load(os.path.join(data_dir, "train/features.npy"))
        y_train = np.load(os.path.join(data_dir, "train/labels.npy"))
        x_val = np.load(os.path.join(data_dir, "val/features.npy"))
        y_val = np.load(os.path.join(data_dir, "val/labels.npy"))
        x_test = np.load(os.path.join(data_dir, "test/features.npy"))
        y_test = np.load(os.path.join(data_dir, "test/labels.npy"))
        
        if dataset == "jetclass":
            # Transpose jetclass data to match expected format
            x_train = x_train.transpose(0, 2, 1)
            x_val = x_val.transpose(0, 2, 1)
            x_test = x_test.transpose(0, 2, 1)
    
    logging.info(f"Data shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    
    # Apply sorting
    if sort_by != "none":
        x_train = apply_sorting(x_train, sort_by)
        x_val = apply_sorting(x_val, sort_by)
        x_test = apply_sorting(x_test, sort_by)
        logging.info(f"Applied '{sort_by}' sorting to data")
    
    # Create attention masks
    train_mask = create_attention_mask(x_train)
    val_mask = create_attention_mask(x_val)
    test_mask = create_attention_mask(x_test)
    
    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    train_mask = torch.BoolTensor(train_mask)
    
    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    val_mask = torch.BoolTensor(val_mask)
    
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    test_mask = torch.BoolTensor(test_mask)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(x_train, train_mask, y_train)
    val_dataset = TensorDataset(x_val, val_mask, y_val)
    test_dataset = TensorDataset(x_test, test_mask, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float = 1.0
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (batch_x, batch_mask, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch_x, batch_mask)
        
        # Compute loss
        loss = criterion(logits, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        if batch_y.dim() == 1 or batch_y.shape[1] == 1:  # Binary classification
            predictions = torch.sigmoid(logits) > 0.5
            correct_predictions += (predictions.squeeze() == batch_y.squeeze()).sum().item()
        else:  # Multi-class classification
            predictions = torch.argmax(logits, dim=1)
            targets = torch.argmax(batch_y, dim=1)
            correct_predictions += (predictions == targets).sum().item()
        
        total_samples += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(batch_x, batch_mask)
            
            # Compute loss
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            
            # Calculate accuracy
            if batch_y.dim() == 1 or batch_y.shape[1] == 1:  # Binary classification
                predictions = torch.sigmoid(logits) > 0.5
                correct_predictions += (predictions.squeeze() == batch_y.squeeze()).sum().item()
            else:  # Multi-class classification
                predictions = torch.argmax(logits, dim=1)
                targets = torch.argmax(batch_y, dim=1)
                correct_predictions += (predictions == targets).sum().item()
            
            total_samples += batch_y.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    dataset: str,
    save_dir: str
) -> Dict[str, Any]:
    """Test the model and compute metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(batch_x, batch_mask)
            
            if batch_y.dim() == 1 or batch_y.shape[1] == 1:  # Binary classification
                predictions = torch.sigmoid(logits).cpu().numpy()
            else:  # Multi-class classification
                predictions = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(batch_y.numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    if dataset in ["top", "QG"]:
        # Binary classification
        accuracy = accuracy_score(targets, (predictions.ravel() > 0.5).astype(int))
        roc_auc = roc_auc_score(targets, predictions.ravel())
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(targets, predictions.ravel())
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'targets': targets
        }
        
    else:
        # Multi-class classification
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1) if targets.ndim > 1 else targets
        
        accuracy = accuracy_score(true_classes, pred_classes)
        roc_auc = roc_auc_score(targets, predictions, average='macro', multi_class='ovo')
        
        # Plot ROC curves for each class
        if dataset == "hls4ml":
            class_names = ["q", "g", "W", "Z", "t"]
        else:  # jetclass
            class_names = [
                "QCD", "Hbb", "Hcc", "Hgg", "H4q",
                "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"
            ]
        
        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(class_names):
            if i < targets.shape[1]:
                fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
                roc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
        plt.close()
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'targets': targets
        }
    
    return results


def create_learning_rate_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    **kwargs
):
    """Create learning rate scheduler"""
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 30), gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=kwargs.get('patience', 10), factor=kwargs.get('factor', 0.5)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train GLA Transformer on jet data")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True, help="Directory containing data files")
    parser.add_argument("--dataset", choices=["hls4ml", "top", "QG", "jetclass"], default="hls4ml")
    parser.add_argument("--save_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--num_particles", type=int, default=128, help="Number of particles")
    parser.add_argument("--sort_by", choices=["pt", "eta", "phi", "delta_R", "kt", "none"], default="pt")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--expand_ratio", type=float, default=4.0, help="FFN expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--pooling_type", choices=["mean", "max", "attention", "adaptive"], default="attention")
    parser.add_argument("--use_short_conv", action="store_true", help="Use short convolutions")
    parser.add_argument("--conv_size", type=int, default=4, help="Convolution kernel size")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Training schedule
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--scheduler_type", choices=["cosine", "step", "plateau"], default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    
    # Regularization and optimization
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto", help="Device to use")
    parser.add_argument("--compile_model", action="store_true", help="Compile model with torch.compile")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set dataset-specific parameters
    if args.dataset == "jetclass":
        args.num_particles = 150
        input_dim = 17
    elif args.dataset == "top":
        args.num_particles = 200  
        input_dim = 3
    elif args.dataset == "QG":
        args.num_particles = 150
        input_dim = 3
    else:  # hls4ml
        input_dim = 3
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, str(args.num_particles), args.sort_by)
    trial = 0
    while True:
        candidate_dir = os.path.join(save_dir, f"trial-{trial}")
        if not os.path.exists(candidate_dir):
            save_dir = candidate_dir
            break
        trial += 1
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Arguments: {args}")
    logging.info(f"Save directory: {save_dir}")
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    train_loader, val_loader, test_loader = prepare_data_loaders(
        args, args.dataset, args.data_dir, args.num_particles,
        args.sort_by, args.val_split, args.batch_size
    )
    
    # Create model
    model = create_gla_model(
        dataset=args.dataset,
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expand_ratio=args.expand_ratio,
        dropout=args.dropout,
        pooling_type=args.pooling_type,
        use_short_conv=args.use_short_conv,
        conv_size=args.conv_size
    )
    
    model = model.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        logging.info("Compiling model with torch.compile")
        model = torch.compile(model)
    
    # Define loss function
    if args.dataset in ["top", "QG"]:
        criterion = nn.BCEWithLogitsLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = create_learning_rate_scheduler(
            optimizer, args.scheduler_type, args.num_epochs
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    logging.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.gradient_clip_norm
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        
        logging.info(
            f"Epoch {epoch+1}/{args.num_epochs} ({epoch_time:.2f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logging.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Save training history
    np.save(os.path.join(save_dir, 'train_loss.npy'), history['train_loss'])
    np.save(os.path.join(save_dir, 'val_loss.npy'), history['val_loss'])
    np.save(os.path.join(save_dir, 'train_accuracy.npy'), history['train_acc'])
    np.save(os.path.join(save_dir, 'val_accuracy.npy'), history['val_acc'])
    
    # Plot training curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    # Profile model
    logging.info("Profiling model performance...")
    try:
        flops, params = get_model_flops(model, (args.batch_size, args.num_particles, input_dim))
        logging.info(f"Model FLOPs: {flops:,}")
        logging.info(f"Model Parameters: {params:,}")
    except Exception as e:
        logging.warning(f"Could not profile FLOPs: {e}")
        logging.info(f"Model Parameters: {model.get_num_trainable_params():,}")
    
    # Profile memory and timing
    curr_mem, peak_mem, avg_time = profile_gpu_memory_and_timing(
        model, test_loader, device
    )
    logging.info(f"GPU Memory - Current: {curr_mem:.1f} MB, Peak: {peak_mem:.1f} MB")
    logging.info(f"Average inference time per sample: {avg_time:.2f} ns")
    
    # Test model
    logging.info("Testing model...")
    test_results = test_model(model, test_loader, device, args.dataset, save_dir)
    
    logging.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logging.info(f"Test ROC AUC: {test_results['roc_auc']:.4f}")
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        results_to_save = {k: v for k, v in test_results.items() 
                          if k not in ['predictions', 'targets']}
        json.dump(results_to_save, f, indent=2)
    
    # Save predictions and targets
    np.save(os.path.join(save_dir, 'test_predictions.npy'), test_results['predictions'])
    np.save(os.path.join(save_dir, 'test_targets.npy'), test_results['targets'])
    
    logging.info(f"Training completed successfully! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()