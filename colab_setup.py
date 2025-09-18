"""
Google Colab Setup Script for Federated Learning Project
Run this in the first cell of your Colab notebook
"""

# Install required packages
import subprocess
import sys
import os

def install_packages():
    """Install required packages for the federated learning project"""
    packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'tensorboard',
        'medmnist',
        'psutil',
        'GPUtil',
        'tqdm',
        'numpy',
        'matplotlib'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_colab_environment():
    """Setup the Colab environment"""
    print("Setting up Colab environment...")
    
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU available - training will be slow")
    
    # Set working directory (adjust path as needed)
    project_path = "/content/drive/MyDrive/gaudi-byzantine"
    if os.path.exists(project_path):
        os.chdir(project_path)
        print(f"✅ Changed to project directory: {project_path}")
    else:
        print(f"⚠️ Project directory not found: {project_path}")
        print("Please upload your project to Google Drive first")
    
    return project_path

def check_project_files():
    """Check if all required project files are present"""
    required_files = [
        'main.py',
        'debug_main.py',
        'utils/options.py',
        'utils/dataset.py',
        'utils/sampling.py',
        'utils/test.py',
        'src/update.py',
        'src/aggregation.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files found")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Federated Learning project in Google Colab...")
    print("=" * 60)
    
    # Install packages
    install_packages()
    print()
    
    # Setup environment
    project_path = setup_colab_environment()
    print()
    
    # Check files
    if check_project_files():
        print("🎉 Setup complete! You can now run your federated learning experiments.")
        print("\nTo start training with GPU:")
        print("!python main.py --debug --device cuda")
    else:
        print("❌ Setup incomplete. Please upload missing files to Google Drive.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()