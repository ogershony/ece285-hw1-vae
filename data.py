'''
Oren Gershony

Loads PIL data into Pytorch DataLoader and inspects data shape
'''

from shared import TRAIN_PATH, TEST_PATH, VAL_PATH

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data() -> dict[str, DataLoader]:
    # Image transformation
    # 1. Greyscale instead of RGB because images are already grey
    # 2. Resize to 128x128 from 1858x2090 because of compute
    # 3. Move to tensor instead of PIL format
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),                  
        transforms.ToTensor(),                        
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform = transform)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    test_dataset = datasets.ImageFolder(root=TEST_PATH, transform = transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

    val_dataset = datasets.ImageFolder(root=VAL_PATH, transform = transform)
    val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
    
    return {
        "train" : train_dataloader,
        "test" : test_dataloader,
        "val" : val_dataloader
    }

def inspect_data(dataset: dict[str, DataLoader]) -> None:
    ''' Inspect shape of dataset and other statistics '''

    # Get shape of dataset item
    feature, _ = next(iter(dataset["train"]))

    print(f"Image shape: {feature.shape}")
