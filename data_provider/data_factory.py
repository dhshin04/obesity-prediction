from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, random_split

from .data_loader import ObesityTorchDataset


def data_provider(args, data_path: str, doctor: bool, relevant_features: List[str] = None) -> Tuple:
    # Split Data
    obesity_dataset = ObesityTorchDataset(data_path, relevant_features, doctor=doctor)

    train_size = int(args.train_ratio * len(obesity_dataset))
    val_size = int(args.val_ratio * len(obesity_dataset))
    test_size = len(obesity_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(obesity_dataset, [train_size, val_size, test_size])

    num_workers = args.num_workers if torch.cuda.is_available() else 1
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        num_workers=num_workers,
        drop_last=bool(args.drop_last)
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
