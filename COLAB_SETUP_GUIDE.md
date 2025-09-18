# Google Colab Setup Guide

## Why Use Google Colab?

✅ **Free GPU Access** - T4 GPU (16GB VRAM)  
✅ **Faster Training** - 10-100x faster than CPU  
✅ **No Local Setup** - Everything runs in the cloud  
✅ **Easy Sharing** - Share notebooks with collaborators  

## Step-by-Step Setup

### 1. Upload Project to Google Drive

1. **Compress your project folder:**
   ```bash
   # On Windows (PowerShell)
   Compress-Archive -Path "C:\Users\ADMIN\Desktop\Fl\gaudi-byzantine" -DestinationPath "gaudi-byzantine.zip"
   ```

2. **Upload to Google Drive:**
   - Go to [Google Drive](https://drive.google.com)
   - Create a new folder called `gaudi-byzantine`
   - Upload all your project files to this folder

### 2. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click "Upload" and upload the `Federated_Learning_Colab.ipynb` file
3. Or create a new notebook and copy the cells from the template

### 3. Enable GPU

1. In Colab: **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### 4. Run the Setup

Execute the first cell to:
- Install required packages
- Mount Google Drive
- Check GPU availability

## Quick Start Commands

### Debug Mode (Recommended First)
```python
!python main.py --debug --dataset bloodmnist --method fedavg --global_ep 5
```

### Fast Training Test
```python
!python main.py --dataset bloodmnist --method fedavg --global_ep 10 --local_ep 2 --lr 0.01
```

### Full Training with GPU
```python
!python main.py --dataset bloodmnist --method fedavg --global_ep 50 --local_ep 5 --lr 0.01 --local_bs 64
```

## Performance Comparison

| Environment | Time per Epoch | Total Time (50 epochs) |
|-------------|----------------|-------------------------|
| Your CPU    | ~123 seconds   | ~1.7 hours             |
| Colab T4 GPU| ~5-10 seconds  | ~4-8 minutes           |
| **Speedup** | **12-25x**     | **12-25x faster**      |

## Optimized Parameters for Colab

```python
# Recommended settings for fast training
--dataset bloodmnist
--method fedavg
--global_ep 50          # Reduced from 200
--local_ep 5            # Keep as is
--lr 0.01               # Increased from 0.001
--local_bs 64           # Reduced from 128 (GPU memory)
--num_clients 10        # Keep as is
--alpha 10.0            # Keep as is
```

## Troubleshooting

### Issue: "No module named 'medmnist'"
**Solution:**
```python
!pip install medmnist
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```python
--local_bs 32  # or even 16
```

### Issue: "Files not found"
**Solution:** Check your Google Drive path
```python
import os
print(os.listdir('/content/drive/MyDrive/'))  # List your Drive contents
```

### Issue: "Session timeout"
**Solution:** Colab sessions last ~12 hours. For longer training:
1. Save checkpoints regularly
2. Use Colab Pro for longer sessions
3. Break training into smaller chunks

## Advanced Features

### 1. TensorBoard Monitoring
```python
%load_ext tensorboard
%tensorboard --logdir runs/
```

### 2. Real-time GPU Monitoring
```python
!nvidia-smi
```

### 3. Download Results
```python
!zip -r results.zip runs/ *.log
from google.colab import files
files.download('results.zip')
```

### 4. Multiple Experiments
```python
methods = ['fedavg', 'krum', 'trimmed_mean']
for method in methods:
    !python main.py --dataset bloodmnist --method {method} --global_ep 20
```

## Expected Performance Improvements

With GPU acceleration, you should see:

1. **Training Speed:** 123s → 5-10s per iteration
2. **Total Time:** 6+ hours → 10-20 minutes
3. **Accuracy:** Should improve due to faster convergence
4. **Resource Usage:** GPU utilization 80-90%

## Cost Considerations

- **Free Tier:** 12 hours/day, T4 GPU
- **Colab Pro:** $10/month, longer sessions, better GPUs
- **Colab Pro+:** $50/month, even better resources

## Next Steps

1. Upload your project to Google Drive
2. Open the Colab notebook
3. Run the debug suite first
4. Start with small experiments
5. Scale up to full training

The GPU acceleration will make your federated learning experiments much more practical and allow you to iterate quickly on different configurations!