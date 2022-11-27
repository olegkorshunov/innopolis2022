from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from cnn1d.config import Config
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR
from torchmetrics import Recall
from tqdm import tqdm
from typing import List
from cnn1d.model import TorchModel


def get_recall(model, dataloader):
    model.eval()
    y_pred = torch.tensor([])
    y_true = torch.tensor([])
    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(Config.device)
            y = model(X)
            y_pred = torch.concat((y_pred, y.to("cpu")))
            y_true = torch.concat((y_true, batch["y"]))

        recall = Recall()
        return recall(
            torch.argmax(y_pred, dim=1),
            y_true.to(torch.int),
        ), torch.argmax(y_pred, dim=1)  # ,y_true


def early_stopping(loss: List[float], delta: float, window: int):
    min_loss = np.min(loss)
    curr_loss = loss[-1]
    first_min_loss_idx = np.where(loss == min_loss)[0][0]
    if (len(loss)-first_min_loss_idx > window) and abs(min_loss-curr_loss) <= delta:
        return True
    return False


def train_model(drpt, dataloader_train, dataloader_val=None):

    model = TorchModel(drpt)
    model = model.to(Config.device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-2, weight_decay=1e-4)

    epochs = tqdm(range(Config.num_epochs), leave=False)
    epochs.set_description("Epoch")
    epochs_loss = []
    #scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    #scheduler = MultiStepLR(optimizer, milestones=[20,50], gamma=0.6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-6)
    for epoch in epochs:
        model.train()
        running_loss = 0.0
        for data in dataloader_train:
            X = data["X"].to(Config.device)
            y = data["y"].to(Config.device)
            optimizer.zero_grad()
            preds = model(X)
            loss_value = loss(preds, y)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=2.0, norm_type=2, error_if_nonfinite=True)
            optimizer.step()
            running_loss += loss_value.item() / y.shape[0]
        scheduler.step(running_loss)
        if dataloader_val:
            epoch_recall, _ = get_recall(model, dataloader_val)
        epochs.set_postfix(
            # scheduler.get_last_lr()[0]
            epoch=epoch, epoch_recall=epoch_recall, loss=running_loss, lr=optimizer.param_groups[
                0]['lr'],
        )
        epochs_loss.append(running_loss)
        # print(epochs_loss)
#         if early_stopping(epochs_loss,0.001,10):
#             break

    return model, epochs_loss
