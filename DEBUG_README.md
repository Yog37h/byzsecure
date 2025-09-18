# Federated Learning Debug Suite

This debug suite helps identify performance bottlenecks and issues in your federated learning system.

## Installation

First, install the debugging dependencies:

```bash
python install_debug_deps.py
```

Or manually install:
```bash
pip install psutil>=5.8.0 GPUtil>=1.4.0
```

## Usage

Run your federated learning with debug mode enabled:

```bash
python main.py --debug
```

## What the Debug Suite Checks

### 1. System Resources
- **CPU Usage & Cores**: Monitor CPU utilization
- **RAM Usage**: Check available memory
- **GPU Information**: GPU memory usage and load
- **CUDA Availability**: Verify CUDA setup

### 2. Model Configuration
- **Device Placement**: Ensure model is on correct device
- **Parameter Count**: Total and trainable parameters
- **Model Size**: Memory footprint estimation

### 3. Dataset Analysis
- **Class Distribution**: Check for class imbalance
- **Missing Classes**: Identify missing or extra classes
- **Dataset Sizes**: Train/test/validation split sizes

### 4. Federated Learning Setup
- **Client Distribution**: Data distribution across clients
- **Empty Clients**: Identify clients with no data
- **Compromised Clients**: Verify Byzantine client setup
- **Data Quality**: Per-client data analysis

### 5. Training Performance
- **Time Breakdown**: Identify bottlenecks in training steps
- **Data Loading**: Check data transfer efficiency
- **Forward/Backward Pass**: Profile model computation
- **Memory Usage**: Monitor GPU memory consumption

## Common Issues and Solutions

### Issue: Very Slow Training (>60s per iteration)

**Possible Causes:**
- GPU not being used properly
- Data loading bottlenecks
- Model too large for available memory
- Inefficient client data distribution

**Solutions:**
1. Verify GPU usage: Check if model and data are on GPU
2. Increase DataLoader `num_workers`
3. Use `pin_memory=True` in DataLoader
4. Reduce batch size if memory constrained
5. Use mixed precision training (`torch.cuda.amp`)

### Issue: Low Accuracy (Random Performance)

**Possible Causes:**
- Incorrect label mapping
- Class imbalance
- Learning rate too high/low
- Data preprocessing issues

**Solutions:**
1. Check class distribution in debug output
2. Verify label ranges match expected classes
3. Adjust learning rate
4. Check data normalization

### Issue: Empty Client Lists `[]`

**Possible Causes:**
- Data sampling issues
- Incorrect client data distribution
- Path configuration problems

**Solutions:**
1. Check `dict_users` in debug output
2. Verify dataset loading
3. Check non-IID sampling parameters
4. Ensure dataset paths are correct

### Issue: High Memory Usage

**Possible Causes:**
- Large batch sizes
- Model too large
- Memory leaks in training loop

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Clear cache with `torch.cuda.empty_cache()`
4. Use smaller model architecture

## Debug Output Interpretation

### System Resources
```
=== SYSTEM RESOURCES DEBUG ===
CPU Usage: 25.3%
CPU Cores: 8
Available RAM: 12.45 GB
RAM Usage: 45.2%
GPU 0: NVIDIA RTX 3080
  Memory Used: 2048MB / 10240MB (20.0%)
  GPU Load: 85.5%
CUDA Available: True
```

**Good Signs:**
- GPU load > 70%
- CUDA available
- Sufficient RAM available

**Warning Signs:**
- GPU load < 30% (underutilized)
- RAM usage > 90%
- CUDA not available when expected

### Client Data Distribution
```
=== FEDERATED LEARNING SETUP DEBUG ===
Total Clients: 10
Compromised Clients: 2
Client Data Sizes: [500, 520, 480, 510, 0, 490, 505, 515, 495, 485]
Empty Clients: [4]
```

**Issues to Address:**
- Empty clients (size 0)
- Highly uneven distribution
- Too many compromised clients

### Training Step Timing
```
=== TRAINING STEP DEBUG ===
Time Breakdown:
  data_loading: 0.0234s (15.2%)
  data_transfer: 0.0045s (2.9%)
  forward_pass: 0.0890s (57.8%)
  backward_pass: 0.0298s (19.4%)
  optimizer_step: 0.0072s (4.7%)
  Total: 0.1539s
```

**Optimization Targets:**
- If data_loading > 30%: Increase num_workers
- If data_transfer > 10%: Use pin_memory
- If forward_pass > 70%: Consider model optimization

## Advanced Debugging

### Profile Specific Clients
The debug suite automatically checks the first 3 clients. To debug specific clients, modify the debug script:

```python
# Debug specific clients
for client_idx in [0, 5, 9]:  # Specify client IDs
    debug_client_data_quality(dataset_train, dict_users, client_idx, args.num_classes)
```

### Custom Performance Monitoring
Add timing to your training loop:

```python
import time

start_time = time.time()
# Your training code here
iteration_time = time.time() - start_time
print(f"Iteration time: {iteration_time:.2f}s")
```

## Troubleshooting

### Import Errors
If you get import errors for `psutil` or `GPUtil`:
```bash
pip install psutil GPUtil
```

### CUDA Issues
If CUDA is not detected but you have a GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
If you run out of memory during debugging:
1. Reduce batch size in debug functions
2. Use smaller sample sizes for client debugging
3. Run debug suite on CPU first

## Performance Benchmarks

**Good Performance Targets:**
- Training iteration: < 10 seconds
- GPU utilization: > 70%
- Data loading: < 20% of total time
- Memory usage: < 80% of available

**Optimization Priority:**
1. Fix empty clients first
2. Ensure GPU utilization
3. Optimize data loading
4. Tune hyperparameters

## Contact

If you encounter issues with the debug suite, check:
1. All dependencies are installed
2. Your dataset is properly loaded
3. Model architecture matches your data
4. CUDA is properly configured (if using GPU)