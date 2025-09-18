import torch
import time
import psutil
import GPUtil
from collections import Counter
import numpy as np

def debug_system_resources():
    """Debug system resources and hardware utilization"""
    print("=== SYSTEM RESOURCES DEBUG ===")
    
    # CPU Info
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"RAM Usage: {psutil.virtual_memory().percent}%")
    
    # GPU Info
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                print(f"  Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                print(f"  GPU Load: {gpu.load*100:.1f}%")
        else:
            print("No GPUs detected")
    except:
        print("GPU monitoring not available")
    
    # CUDA Info
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA Available: False")
    
    print()

def debug_model_config(model):
    """Debug model configuration and device placement"""
    print("=== MODEL CONFIGURATION DEBUG ===")
    
    # Model device
    model_device = next(model.parameters()).device
    print(f"Model Device: {model_device}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024**2)
    print(f"Estimated Model Size: {model_size_mb:.2f} MB")
    
    print()

def debug_data_loader(dataloader, name="DataLoader"):
    """Debug data loader configuration and sample data"""
    print(f"=== {name.upper()} DEBUG ===")
    
    print(f"Batch Size: {dataloader.batch_size}")
    print(f"Dataset Size: {len(dataloader.dataset)}")
    print(f"Number of Batches: {len(dataloader)}")
    print(f"Number of Workers: {dataloader.num_workers}")
    
    # Sample a batch
    try:
        start_time = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start_time
        
        if isinstance(batch, (list, tuple)):
            data, labels = batch
            print(f"Data Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")
            print(f"Data Device: {data.device}")
            print(f"Labels Shape: {labels.shape}")
            print(f"Labels Type: {labels.dtype}")
            print(f"Labels Device: {labels.device}")
            
            # Check label distribution
            unique_labels = torch.unique(labels)
            print(f"Unique Labels: {unique_labels.tolist()}")
            print(f"Label Range: {labels.min().item()} - {labels.max().item()}")
            
            # Count label distribution
            label_counts = Counter(labels.cpu().numpy())
            print(f"Label Distribution: {dict(label_counts)}")
            
        print(f"Batch Load Time: {load_time:.4f} seconds")
        
    except Exception as e:
        print(f"Error loading batch: {e}")
    
    print()

def debug_training_step(model, dataloader, criterion, optimizer, device):
    """Debug a single training step for performance bottlenecks"""
    print("=== TRAINING STEP DEBUG ===")
    
    model.train()
    
    # Time different parts of training step
    times = {}
    
    # Data loading
    start_time = time.time()
    batch = next(iter(dataloader))
    data, labels = batch
    times['data_loading'] = time.time() - start_time
    
    # Data transfer to device
    start_time = time.time()
    data = data.to(device)
    labels = labels.to(device)
    times['data_transfer'] = time.time() - start_time
    
    # Forward pass
    start_time = time.time()
    outputs = model(data)
    times['forward_pass'] = time.time() - start_time
    
    # Loss computation
    start_time = time.time()
    loss = criterion(outputs, labels)
    times['loss_computation'] = time.time() - start_time
    
    # Backward pass
    start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    times['backward_pass'] = time.time() - start_time
    
    # Optimizer step
    start_time = time.time()
    optimizer.step()
    times['optimizer_step'] = time.time() - start_time
    
    total_time = sum(times.values())
    
    print("Time Breakdown:")
    for step, duration in times.items():
        percentage = (duration / total_time) * 100
        print(f"  {step}: {duration:.4f}s ({percentage:.1f}%)")
    print(f"  Total: {total_time:.4f}s")
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Output Shape: {outputs.shape}")
    print()

def debug_class_distribution(dataset, num_classes):
    """Debug class distribution in dataset"""
    print("=== CLASS DISTRIBUTION DEBUG ===")
    
    print(f"Expected Classes: {num_classes}")
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    else:
        # Manually extract labels
        all_labels = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000
            try:
                _, label = dataset[i]
                all_labels.append(label)
            except Exception as e:
                print(f"Error accessing sample {i}: {e}")
                break
    
    # Convert to numpy array first
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.cpu().numpy()
    elif isinstance(all_labels, list):
        all_labels = np.array(all_labels)
    
    # Handle different label formats
    processed_labels = []
    for label in all_labels:
        if isinstance(label, np.ndarray):
            if label.size == 1:
                processed_labels.append(int(label.item()))
            else:
                # Multi-dimensional label, take argmax or first element
                processed_labels.append(int(np.argmax(label) if label.ndim > 0 else label))
        elif isinstance(label, (int, np.integer)):
            processed_labels.append(int(label))
        elif isinstance(label, torch.Tensor):
            if label.numel() == 1:
                processed_labels.append(int(label.item()))
            else:
                processed_labels.append(int(torch.argmax(label).item()))
        else:
            try:
                processed_labels.append(int(label))
            except:
                print(f"Warning: Could not process label of type {type(label)}: {label}")
    
    if not processed_labels:
        print("⚠️  No valid labels found!")
        return
    
    label_counts = Counter(processed_labels)
    print(f"Actual Label Distribution: {dict(sorted(label_counts.items()))}")
    print(f"Number of Unique Labels: {len(label_counts)}")
    print(f"Total Samples: {sum(label_counts.values())}")
    
    # Check for missing classes
    expected_classes = set(range(num_classes))
    actual_classes = set(label_counts.keys())
    missing_classes = expected_classes - actual_classes
    extra_classes = actual_classes - expected_classes
    
    if missing_classes:
        print(f"Missing Classes: {sorted(missing_classes)}")
    if extra_classes:
        print(f"Extra Classes: {sorted(extra_classes)}")
    
    print()

def debug_federated_setup(dict_users, compromised_idxs, args):
    """Debug federated learning specific setup"""
    print("=== FEDERATED LEARNING SETUP DEBUG ===")
    
    print(f"Total Clients: {args.num_clients}")
    print(f"Compromised Fraction: {args.c_frac}")
    print(f"Compromised Clients: {len(compromised_idxs)}")
    print(f"Compromised Client IDs: {compromised_idxs}")
    
    # Check client data distribution
    client_data_sizes = []
    for idx in range(args.num_clients):
        if idx in dict_users:
            client_data_sizes.append(len(dict_users[idx]))
        else:
            client_data_sizes.append(0)
    
    print(f"Client Data Sizes: {client_data_sizes}")
    print(f"Min Client Data: {min(client_data_sizes)}")
    print(f"Max Client Data: {max(client_data_sizes)}")
    print(f"Avg Client Data: {np.mean(client_data_sizes):.2f}")
    print(f"Total Data Points: {sum(client_data_sizes)}")
    
    # Check for empty clients
    empty_clients = [idx for idx, size in enumerate(client_data_sizes) if size == 0]
    if empty_clients:
        print(f"⚠️  Empty Clients: {empty_clients}")
    
    print()

def debug_client_data_quality(dataset, dict_users, client_idx, num_classes):
    """Debug data quality for a specific client"""
    print(f"=== CLIENT {client_idx} DATA QUALITY DEBUG ===")
    
    if client_idx not in dict_users:
        print(f"⚠️  Client {client_idx} not found in dict_users!")
        return
    
    client_indices = dict_users[client_idx]
    print(f"Client {client_idx} has {len(client_indices)} samples")
    
    if len(client_indices) == 0:
        print(f"⚠️  Client {client_idx} has no data!")
        return
    
    # Get labels for this client
    client_labels = []
    for idx in client_indices:
        try:
            _, label = dataset[idx]
            # Handle different label formats
            if isinstance(label, np.ndarray):
                if label.size == 1:
                    client_labels.append(int(label.item()))
                else:
                    client_labels.append(int(np.argmax(label)))
            elif isinstance(label, torch.Tensor):
                if label.numel() == 1:
                    client_labels.append(int(label.item()))
                else:
                    client_labels.append(int(torch.argmax(label).item()))
            else:
                client_labels.append(int(label))
        except Exception as e:
            print(f"Error accessing sample {idx}: {e}")
    
    if client_labels:
        label_counts = Counter(client_labels)
        print(f"Client {client_idx} Label Distribution: {dict(sorted(label_counts.items()))}")
        print(f"Client {client_idx} Unique Classes: {len(label_counts)}")
        
        # Check class coverage
        missing_classes = set(range(num_classes)) - set(label_counts.keys())
        if missing_classes:
            print(f"Client {client_idx} Missing Classes: {sorted(missing_classes)}")
    
    print()

# Add this to your main.py to run debugging
def run_debug_suite(model, dataset_train, dataset_test, dict_users, compromised_idxs, args):
    """Run complete debug suite for federated learning"""
    print("Starting Federated Learning Debug Suite...")
    print("=" * 50)
    
    debug_system_resources()
    debug_model_config(model)
    
    # Create sample data loaders for debugging
    from torch.utils.data import DataLoader, Subset
    
    # Debug overall dataset
    print("=== DATASET DEBUG ===")
    print(f"Train Dataset Size: {len(dataset_train)}")
    print(f"Test Dataset Size: {len(dataset_test)}")
    print()
    
    debug_class_distribution(dataset_train, args.num_classes)
    debug_federated_setup(dict_users, compromised_idxs, args)
    
    # Debug a few clients
    for client_idx in range(min(3, args.num_clients)):
        debug_client_data_quality(dataset_train, dict_users, client_idx, args.num_classes)
    
    # Create a sample data loader for training step debug
    if 0 in dict_users and len(dict_users[0]) > 0:
        # Convert set to list for indexing
        client_indices = list(dict_users[0])
        sample_indices = client_indices[:min(32, len(client_indices))]
        sample_subset = Subset(dataset_train, sample_indices)
        sample_loader = DataLoader(sample_subset, batch_size=min(8, len(sample_subset)), shuffle=True)
        
        debug_data_loader(sample_loader, "Sample Client DataLoader")
        
        # Debug training step if we have the necessary components
        try:
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            debug_training_step(model, sample_loader, criterion, optimizer, args.device)
        except Exception as e:
            print(f"Could not debug training step: {e}")
    
    print("Debug suite completed!")
    print("=" * 50)

# Performance optimization suggestions
def suggest_optimizations(current_time_per_iteration):
    """Suggest optimizations based on current performance"""
    print("=== OPTIMIZATION SUGGESTIONS ===")
    
    if current_time_per_iteration > 60:  # More than 1 minute per iteration
        print("⚠️  Very slow training detected!")
        print("Suggestions:")
        print("1. Check if GPU is being used properly")
        print("2. Reduce batch size if running out of memory")
        print("3. Increase DataLoader num_workers")
        print("4. Use mixed precision training (torch.cuda.amp)")
        print("5. Profile your model for bottlenecks")
        print("6. Check for data loading bottlenecks")
        print("7. Verify client data distribution is not causing issues")
    
    if current_time_per_iteration > 10:  # More than 10 seconds
        print("🔍 Consider these optimizations:")
        print("1. Use DataLoader with pin_memory=True")
        print("2. Optimize data preprocessing")
        print("3. Use torch.compile() for model optimization")
        print("4. Check for unnecessary CPU-GPU transfers")
        print("5. Reduce number of clients per round if possible")
        print("6. Use gradient compression techniques")
    
    print("🔧 Federated Learning Specific Optimizations:")
    print("1. Check if client data is properly distributed")
    print("2. Verify no clients have empty datasets")
    print("3. Consider using client sampling instead of all clients")
    print("4. Check if Byzantine detection is causing overhead")
    print("5. Profile aggregation methods for efficiency")
    
    print()

def debug_args_config(args):
    """Debug command line arguments and configuration"""
    print("=== CONFIGURATION DEBUG ===")
    
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: ResNet18")
    print(f"Device: {args.device}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Global Epochs: {args.global_ep}")
    print(f"Local Epochs: {args.local_ep}")
    print(f"Batch Size: {args.local_bs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Number of Clients: {args.num_clients}")
    print(f"Compromised Fraction: {args.c_frac}")
    print(f"Alpha (Non-IID): {args.alpha}")
    print(f"Plugin Enabled: {args.plugin}")
    print(f"Attack Type: {args.p}")
    print(f"Debug Mode: {args.debug}")
    
    # Check for potential issues
    if args.c_frac > 0.5:
        print("⚠️  High compromised fraction detected!")
    
    if args.local_bs > 128:
        print("⚠️  Large batch size may cause memory issues")
    
    if args.lr > 0.1:
        print("⚠️  High learning rate may cause instability")
    
    print()