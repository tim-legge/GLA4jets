#!/usr/bin/env python3
"""
Example usage of the GLA Transformer for jet classification
This script demonstrates how to use the model with synthetic data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gla_transformer import create_gla_model

def create_synthetic_jet_data(
    num_samples: int = 1000,
    num_particles: int = 128,
    input_dim: int = 17,
    num_classes: int = 5
):
    """
    Create synthetic jet data for testing
    """
    # Generate random particle features
    # Features typically include: pt, eta, phi, mass, charge, pdgId, etc.
    X = np.random.randn(num_samples, num_particles, input_dim)
    
    # Make pt values positive and sort by pt in descending order
    X[:, :, 0] = np.abs(X[:, :, 0]) * 100  # pt
    X[:, :, 1] = X[:, :, 1] * 2.5  # eta (typical range -2.5 to 2.5)
    X[:, :, 2] = X[:, :, 2] * np.pi  # phi (range -π to π)
    
    # Sort by pt (descending)
    pt_indices = np.argsort(X[:, :, 0], axis=1)[:, ::-1]
    X = np.take_along_axis(X, pt_indices[:, :, None], axis=1)
    
    # Create some realistic padding (some particles have zero pt)
    for i in range(num_samples):
        # Randomly choose number of real particles (50-128)
        num_real = np.random.randint(50, num_particles + 1)
        X[i, num_real:, :] = 0  # Zero out padded particles
    
    # Generate random labels
    y = np.eye(num_classes)[np.random.choice(num_classes, num_samples)]
    
    # Create attention masks (1 for real particles, 0 for padding)
    masks = (X[:, :, 0] > 0).astype(np.float32)  # pt > 0 indicates real particle
    
    return X, y, masks

def test_gla_model():
    """Test the GLA model with synthetic data"""
    
    print("Creating synthetic jet data...")
    X, y, masks = create_synthetic_jet_data(
        num_samples=100,
        num_particles=128,
        input_dim=17,
        num_classes=5
    )
    
    print(f"Data shapes: X={X.shape}, y={y.shape}, masks={masks.shape}")
    print(f"Average particles per jet: {masks.sum(axis=1).mean():.1f}")
    
    # Create model
    print("\nCreating GLA model...")
    model = create_gla_model(
        dataset="hls4ml",
        input_dim=17,
        hidden_size=64,  # Smaller for testing
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        pooling_type="attention"
    )
    
    print(f"Model created with {model.get_num_trainable_params():,} parameters")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    mask_tensor = torch.BoolTensor(masks)
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor, mask_tensor)
        probabilities = torch.softmax(logits, dim=1)
        
    print(f"Input shape: {X_tensor.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {probabilities[:3].numpy()}")
    
    # Test with different batch sizes
    print("\nTesting with different batch sizes...")
    batch_sizes = [1, 5, 10]
    
    for bs in batch_sizes:
        with torch.no_grad():
            batch_logits = model(X_tensor[:bs], mask_tensor[:bs])
            print(f"Batch size {bs}: output shape {batch_logits.shape}")
    
    # Test training mode
    print("\nTesting training mode...")
    model.train()
    
    # Simulate training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    logits = model(X_tensor[:10], mask_tensor[:10])
    loss = criterion(logits, y_tensor[:10])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item():.4f}")
    
    return model, X_tensor, y_tensor, mask_tensor

def visualize_attention_patterns(model, X, masks, save_path="attention_viz.png"):
    """
    Visualize attention patterns (if the model supports it)
    This is a simplified visualization - full attention analysis would require
    modifying the model to return attention weights
    """
    model.eval()
    
    # Take a single sample
    sample_x = X[:1]  # [1, seq_len, input_dim]
    sample_mask = masks[:1]  # [1, seq_len]
    
    with torch.no_grad():
        # Get features before pooling
        logits, features = model(sample_x, sample_mask, return_features=True)
    
    print(f"Sample jet features shape: {features.shape}")
    
    # Simple visualization: show particle pt vs predicted features
    sample_data = sample_x[0].numpy()  # [seq_len, input_dim]
    sample_mask_np = sample_mask[0].numpy()  # [seq_len]
    
    # Get valid (non-padded) particles
    valid_particles = sample_mask_np.astype(bool)
    pt_values = sample_data[valid_particles, 0]  # pt values
    
    # Plot pt distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(pt_values)), pt_values)
    plt.xlabel('Particle Index')
    plt.ylabel('Transverse Momentum (pt)')
    plt.title('Particle pt Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(pt_values, bins=20, alpha=0.7)
    plt.xlabel('Transverse Momentum (pt)')
    plt.ylabel('Count')
    plt.title('pt Histogram')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {save_path}")

def benchmark_model_speed():
    """Benchmark model inference speed"""
    print("\nBenchmarking model speed...")
    
    # Create model and data
    model = create_gla_model(
        dataset="hls4ml",
        hidden_size=128,
        num_layers=6,
        num_heads=8
    )
    
    X, _, masks = create_synthetic_jet_data(
        num_samples=100,
        num_particles=128,
        input_dim=17
    )
    
    X_tensor = torch.FloatTensor(X)
    mask_tensor = torch.BoolTensor(masks)
    
    # GPU benchmark if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_tensor = X_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    print(f"Using device: {device}")
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_tensor[:10], mask_tensor[:10])
    
    # Benchmark
    import time
    times = []
    
    with torch.no_grad():
        for i in range(20):
            start_time = time.perf_counter()
            _ = model(X_tensor, mask_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times[5:])  # Skip first few for warmup
    throughput = len(X) / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.1f} jets/sec")
    print(f"Time per jet: {avg_time/len(X)*1e6:.2f} μs")

if __name__ == "__main__":
    print("GLA Transformer Example")
    print("=" * 50)
    
    # Test basic functionality
    model, X, y, masks = test_gla_model()
    
    # Visualize sample
    try:
        visualize_attention_patterns(model, X, masks)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Benchmark speed
    try:
        benchmark_model_speed()
    except Exception as e:
        print(f"Benchmarking failed: {e}")
    
    print("\nExample completed successfully!")
    print("\nTo train on real data, use:")
    print("python train_gla_pytorch.py --data_dir /path/to/data --dataset hls4ml --save_dir ./results")