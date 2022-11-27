from dataclasses import dataclass


@dataclass
class Config:
    RS: int = 42
    device = "cpu"
    num_epochs: int = 40
    batch_size: int = 256
    num_workers: int = 0

    path_train: str = r"D:\Projects\innopolis2022\data\train_dataset_train.csv"
    path_test: str = r"D:\Projects\innopolis2022\data\test_dataset_test.csv"
    path_subm: str = r"D:\Projects\innopolis2022\data\sample_solution.csv"
