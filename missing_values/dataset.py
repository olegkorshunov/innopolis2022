import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


class CustomDataset(Dataset):    
    def __init__(self,
                 df_KNN:pd.DataFrame,
                 df_restore:pd.DataFrame,
                 y_i:int=0,
                 N:int=5,
                 mode:str='Train',
                ):        
        """
        y_i - number of restoring column
        N:int=5 - n_neighbors - odd number!
        mode:Train Train/Val/Test
        """
        self.df_KNN=df_KNN.values
        self.df_restore=df_restore.values         
        self.y_i=y_i
        self.N=N
        self.mid=self.N//2
        assert self.N%2==1, 'N must be odd!'
        self.mode=mode
        self.len=self.df_restore.shape[0]                
        self.KNN=NearestNeighbors(n_neighbors=self.N);        
        self.KNN.fit(X=self.df_KNN)   

    def __len__(self):
        return self.len

    def __getitem__(self, idx):        
        v=np.expand_dims(self.df_restore[idx],axis=0)
        _,idxs=self.KNN.kneighbors(v,return_distance=True)
        idxs=np.concatenate((idxs[0][1::2],idxs[0][::2]))
        x=self.df_KNN[idxs]        
        if self.mode=='Test':
            x[self.mid][self.y_i]=0
            return {
                'X':torch.tensor(x,dtype=torch.float32).unsqueeze(0),
            }
        y=x[self.mid][self.y_i]
        x[self.mid][self.y_i]=0
        return {
            'X':torch.tensor(x,dtype=torch.float32).unsqueeze(0),
            'y':torch.tensor(y,dtype=torch.float32),
        }
        