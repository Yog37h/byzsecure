#!/usr/bin/env python3
"""
Install debugging dependencies for the federated learning debug suite.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required debugging packages"""
    print("Installing debugging dependencies...")
    
    packages = [
        "psutil>=5.8.0",
        "GPUtil>=1.4.0"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All debugging dependencies installed! You can now run the debug suite.")
    else:
        print("⚠️  Some packages failed to install. The debug suite may not work properly.")

if __name__ == "__main__":
    main()