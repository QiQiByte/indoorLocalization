import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class CSIJointDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        class_id = int(row['class_id']) - 1  # 转换为0-based索引
        x = float(row['x'])*0.5  # 假设1单位=50cm，转换为米
        y = float(row['y'])*0.5  # 假设1单位=50cm，转换为米
        coord = torch.tensor([x, y], dtype=torch.float32)

        return image, class_id, coord

def get_dataloaders(csv_path, image_dir, batch_size=64, val_ratio=0.2, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整为更适合的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    full_dataset = CSIJointDataset(
        csv_file=csv_path,
        image_dir=image_dir,
        transform=transform
    )

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader