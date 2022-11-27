from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
from config import Config
from model_0 import TorchModel_v0
import torch
import torch.nn as nn


def train_model(dataloader_train):

    model = TorchModel_v0(Config.window)
    model = model.to(Config.device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-2,
    )  # weight_decay=1e-4)

    epochs = tqdm(range(Config.num_epochs), leave=False)
    epochs.set_description("Epoch")
    epochs_loss = []
    scheduler = StepLR(optimizer, step_size=Config.StepLR, gamma=0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 60], gamma=0.1)
    for epoch in epochs:
        running_loss = 0.0
        for data in dataloader_train:
            X = data["X"].to(Config.device)
            y = data["y"].to(Config.device)
            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss(preds, y)
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item() / y.shape[0]
        scheduler.step()
        epochs.set_postfix(
            epoch=epoch, loss=running_loss, lr=scheduler.get_last_lr()[0]
        )
        epochs_loss.append(loss_value.item())

    return model, epochs_loss
