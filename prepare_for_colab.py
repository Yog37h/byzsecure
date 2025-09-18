#!/usr/bin/env python3
"""
Prepare the federated learning project for Google Colab
This script creates a clean version of your project ready for upload
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_colab_package():
    """Create a clean package for Colab upload"""
    
    # Define source and destination
    source_dir = Path(".")
    colab_dir = Path("colab_package")
    
    # Create clean directory
    if colab_dir.exists():
        shutil.rmtree(colab_dir)
    colab_dir.mkdir()
    
    # Files and directories to include
    include_patterns = [
        "*.py",
        "*.md", 
        "*.txt",
        "*.ipynb",
        "src/",
        "utils/",
        "data/",
        "execute/"
    ]
    
    # Files to exclude
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".git",
        "runs/",
        "*.log",
        "*.zip"
    ]
    
    print("📦 Creating Colab package...")
    
    # Copy essential files
    essential_files = [
        "main.py",
        "debug_main.py", 
        "colab_setup.py",
        "Federated_Learning_Colab.ipynb",
        "COLAB_SETUP_GUIDE.md",
        "DEBUG_README.md",
        "README.md"
    ]
    
    for file in essential_files:
        if Path(file).exists():
            shutil.copy2(file, colab_dir / file)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️  Missing {file}")
    
    # Copy directories
    dirs_to_copy = ["src", "utils"]
    for dir_name in dirs_to_copy:
        if Path(dir_name).exists():
            shutil.copytree(dir_name, colab_dir / dir_name, 
                          ignore=shutil.ignore_patterns(*exclude_patterns))
            print(f"✅ Copied {dir_name}/")
        else:
            print(f"⚠️  Missing {dir_name}/")
    
    # Create requirements.txt for Colab
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "medmnist>=2.0.0",
        "tensorboard>=2.7.0",
        "psutil>=5.8.0",
        "GPUtil>=1.4.0",
        "tqdm>=4.62.0",
        "numpy>=1.21.0"
    ]
    
    with open(colab_dir / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    print("✅ Created requirements.txt")
    
    # Create a quick start script
    quick_start = '''#!/usr/bin/env python3
"""
Quick start script for Google Colab
Run this after uploading to Colab
"""

# Install packages
import subprocess
import sys

packages = ["medmnist", "psutil", "GPUtil"]
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Run debug
import os
os.system("python main.py --debug --global_ep 2")
'''
    
    with open(colab_dir / "quick_start.py", "w") as f:
        f.write(quick_start)
    print("✅ Created quick_start.py")
    
    # Create zip file
    zip_path = "gaudi_byzantine_colab.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(colab_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(colab_dir)
                zipf.write(file_path, arc_path)
    
    print(f"✅ Created {zip_path}")
    
    # Cleanup
    shutil.rmtree(colab_dir)
    
    print("\n🎉 Colab package ready!")
    print(f"📁 Upload {zip_path} to Google Drive")
    print("📝 Follow the COLAB_SETUP_GUIDE.md for next steps")
    
    return zip_path

def print_colab_instructions():
    """Print quick instructions for Colab setup"""
    print("\n" + "="*60)
    print("🚀 GOOGLE COLAB SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Upload gaudi_byzantine_colab.zip to Google Drive")
    print("2. Extract the files in Drive")
    print("3. Open Google Colab (colab.research.google.com)")
    print("4. Upload the .ipynb notebook file")
    print("5. Enable GPU: Runtime → Change runtime type → GPU")
    print("6. Run the first cell to setup environment")
    print("7. Start with debug mode: !python main.py --debug")
    print("\n💡 Expected speedup: 10-25x faster with GPU!")
    print("💡 Training time: 6+ hours → 10-20 minutes")
    print("="*60)

if __name__ == "__main__":
    zip_file = create_colab_package()
    print_colab_instructions()
    
    # Show file size
    size_mb = os.path.getsize(zip_file) / (1024 * 1024)
    print(f"\n📊 Package size: {size_mb:.1f} MB")
    
    if size_mb > 25:
        print("⚠️  Large file - consider removing unnecessary files")
    else:
        print("✅ Good size for Google Drive upload")