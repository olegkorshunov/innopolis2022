import logging
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics import R2Score
from tqdm import tqdm

from config import Config
from dataset import CustomDataset
from train_model import train_model


def get_r2score(model, dataloader, X, y_i):
    res = torch.tensor([])
    with torch.no_grad():
        for batch in dataloader:
            y_pred = model(batch["X"].to(Config.device))
            res = torch.concat((res, y_pred.to("cpu")))

        r2score_torch = R2Score()
        return r2score_torch(
            res,
            torch.tensor(X.values[:, [y_i]].ravel(), dtype=torch.float32),
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="py_log.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    df_train = pd.read_csv(Config.path_train_nan).drop(["id"], axis=1)
    df_test = pd.read_csv(Config.path_test_nan).drop(["id"], axis=1)
    df_train = df_train.sort_index(axis=1)
    df_test = df_test.sort_index(axis=1)
    y = df_train[["crop"]]
    # df_train_geo_area = df_train[[".geo", "area"]]
    # df_test_geo_area = df_train[[".geo", "area"]]
    df_train.drop([".geo", "area", "crop"], axis=1, inplace=True)
    df_test.drop([".geo", "area"], axis=1, inplace=True)

    df_concat = pd.concat((df_train, df_test), ignore_index=True)

    len_df = df_concat.shape[1]
    cols = tqdm(range(len_df))  # CURR
    #cols = tqdm([7, 25])
    cols.set_description("Cols ")

    df_train_rf = df_train.copy()
    df_test_rf = df_test.copy()

    for i in cols:
        df_train_i = df_concat.copy()

        half_window = Config.window // 2
        if i - half_window < 0:
            left, right = 0, Config.window
            y_i = i
        elif i + half_window >= len_df:
            left, right = len_df - Config.window, len_df
            y_i += 1
        else:
            left, right = i - half_window, i + half_window + 1
            y_i = half_window

        df_train_i = df_concat.iloc[:, range(left, right)]
        df_train_i = df_train_i.loc[
            df_train_i[df_train_i.columns[y_i]] > Config.treshold
        ].reset_index(drop=True)

        if Config.val:
            x_trn, x_tst = train_test_split(df_train_i, test_size=0.2)
            dataset_train = CustomDataset(x_trn, x_trn, y_i, Config.KNN, mode="Train")
            dataset_val = CustomDataset(x_trn, x_tst, y_i, Config.KNN, mode="Val")
            dataloader_val = DataLoader(
                dataset_val,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

        else:
            dataset_train = CustomDataset(
                df_train_i, df_train_i, y_i, Config.KNN, mode="Train"
            )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        model, epochs_loss = train_model(dataloader_train)

        model = model.eval()

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        if Config.val:
            r2score_train = get_r2score(model, dataloader_train, x_trn, y_i)
            r2score_val = get_r2score(model, dataloader_val, x_tst, y_i)
            logging.info(
                f"R2score_val: {r2score_val} R2score_train: {r2score_train} "
                f"left: {left} right: {right} y_i: {y_i} {df_train_i.columns[y_i]}"
            )
        else:
            r2score_train = get_r2score(model, dataloader_train, df_train_i, y_i)
            logging.info(
                f"R2score_train: {r2score_train} left: {left} right: {right} y_i: {y_i} {df_train_i.columns[y_i]}"
            )

            # ---
            for key, df in (("train", df_train_rf), ("test", df_test_rf)):
                df = df.iloc[:, range(left, right)]
                idx = df.loc[df.iloc[:, y_i] <= Config.treshold].index
                df = df.iloc[idx]
                dataset_rest = CustomDataset(
                    df_train_i, df, y_i, Config.KNN, mode="Test"
                )

                dataloader_rest = DataLoader(
                    dataset_rest,
                    batch_size=Config.batch_size,
                    shuffle=False,
                    num_workers=Config.num_workers,
                    pin_memory=True,
                )
                y_pred_i = torch.tensor([])
                with torch.no_grad():
                    for batch in dataloader_rest:
                        y_pred = model(batch["X"].to(Config.device))
                        y_pred_i = torch.concat((y_pred_i, y_pred.to("cpu")))
                idx = df.loc[df.iloc[:, y_i] <= Config.treshold].index
                if key == "train":
                    df_train_rf.iloc[idx, i] = y_pred_i.detach().cpu().numpy()
                if key == "test":
                    df_test_rf.iloc[idx, i] = y_pred_i.detach().cpu().numpy()
            # ---
            y.join(df_train_rf).to_csv(
                os.path.join(os.getcwd(), "chkps", f"df_train_rf_0{i}.csv")
            )
            df_test_rf.to_csv(
                os.path.join(os.getcwd(), "chkps", f"df_test_rf_0{i}.csv")
            )

        del model
        torch.cuda.empty_cache()
        cuda_memory = torch.cuda.memory_allocated()
        logging.info(f"cuda_memory: {cuda_memory} loss: {epochs_loss[::4]}")
