from dataclasses import dataclass

import torch


@dataclass
class Config:
    RS: int = 42  # RandomSeed
    KNN: int = 11
    treshold: float = 0.0
    window: int = 15
    path_train: str = r"D:\Projects\innopolis2022\data\train_dataset_train.csv"
    path_test: str = r"D:\Projects\innopolis2022\data\test_dataset_test.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs: int = 80
    StepLR: int = 60
    batch_size: int = 256
    num_workers: int = 0
    val: bool = True
