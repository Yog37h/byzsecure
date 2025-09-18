from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os

try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_data(args):
    # Try to use MedMNIST first
    if MEDMNIST_AVAILABLE and args.dataset in INFO:
        try:
            info = INFO[args.dataset]
            DataClass = getattr(medmnist, info['python_class'])
            trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

            dataset_train = DataClass(split='train', transform=trans, download=True, as_rgb=True)
            dataset_test = DataClass(split='test', transform=trans, download=True, as_rgb=True)
            dataset_val = DataClass(split='val', transform=trans, download=True, as_rgb=True)
            
            return dataset_train, dataset_test, dataset_val
        except Exception as e:
            print(f"Failed to load {args.dataset}: {e}")
            print("Falling back to CIFAR-10...")
            return load_cifar10_fallback()
    else:
        print(f"MedMNIST not available or dataset {args.dataset} not found. Using CIFAR-10...")
        return load_cifar10_fallback()

def load_cifar10_fallback():
    """Fallback to CIFAR-10 dataset if MedMNIST fails"""
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    
    # For CIFAR-10, we'll use a subset of training data as validation
    import torch
    from torch.utils.data import Subset
    
    # Use last 5000 samples of training set as validation
    val_indices = list(range(45000, 50000))
    train_indices = list(range(0, 45000))
    
    dataset_val = Subset(dataset_train, val_indices)
    dataset_train = Subset(dataset_train, train_indices)
    
    return dataset_train, dataset_test, dataset_val
