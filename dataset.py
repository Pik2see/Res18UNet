import os
import glob
import csv
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import Config

cfg = Config()


class FaultData(Dataset):

    def __init__(self, mode, cfg=cfg) -> None:

        super(FaultData, self).__init__()

        self.batch_size = cfg.batch_size
        self.dim = cfg.dim
        self.n_channels = cfg.n_channels
        self.shuffle = cfg.shuffle
        self.mode = mode

        if mode == 'train':
            self.seismPath = cfg.seismPathTr
            self.faultPath = cfg.faultPathTr
            self.data_IDs = self._calculate_files(self.seismPath)
            # self.data_IDs = cfg.train_id
        elif mode == 'valid':
            self.seismPath = cfg.seismPathVa
            self.faultPath = cfg.faultPathVa
            self.data_IDs = self._calculate_files(self.seismPath)
            # self.data_IDs = cfg.valid_id
        elif mode == 'pred':
            self.seismPath = cfg.seismPathPre
            self.faultPath = cfg.faultPathPre
            self.data_IDs = self._calculate_files(self.seismPath)
            # self.data_IDs = cfg.test_id
        else:
            raise ValueError('Mode error!')

        # self.data_IDs = cfg.train_id

        # self.mean, self.var = self.__mean_var()

        print('initialization successful!')
        print(f'{mode} data len is', self.data_IDs)

    def __getitem__(self, index):
        # return super().__getitem__(index)
        if self.mode == 'pred':

            gx = np.fromfile(
                self.seismPath+str(index)+'.dat', dtype=np.single)
            gx = np.reshape(gx, self.dim)

            xm = np.mean(gx)
            xs = np.std(gx)
            gx = gx-xm
            gx = gx/xs
            # 在地震处理中，地震阵列的维数通常按a[n3][n2][n1]排列，其中n1表示垂直维数。
            # 这就是为什么我们需要在python中转置数组的原因
            gx = np.transpose(gx)

            data = np.zeros((self.n_channels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.n_channels, *self.dim))

            data = torch.tensor(data)

            return data

        else:

            gx = np.fromfile(
                self.seismPath+str(index)+'.dat', dtype=np.single)
            fx = np.fromfile(
                self.faultPath+str(index)+'.dat', dtype=np.single)
            gx = np.reshape(gx, self.dim)
            fx = np.reshape(fx, self.dim)

            xm = np.mean(gx)
            xs = np.std(gx)
            gx = gx-xm
            gx = gx/xs
            # 在地震处理中，地震阵列的维数通常按a[n3][n2][n1]排列，其中n1表示垂直维数。
            # 这就是为什么我们需要在python中转置数组的原因
            gx = np.transpose(gx)
            fx = np.transpose(fx)

            data = np.zeros((self.n_channels, *self.dim), dtype=np.single)
            label = np.zeros((self.n_channels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.n_channels, *self.dim))
            label = np.reshape(fx, (self.n_channels, *self.dim))

            data = torch.tensor(data)
            label = torch.tensor(label)

            return data, label

    def __len__(self):
        return self.data_IDs

    def _calculate_files(self, file_path: str) -> int:

        file_nums = 0
        for _ in sorted(os.listdir(file_path)):

            if os.path.isdir(os.path.join(file_path, _)):
                continue

            if os.path.splitext(os.path.join(file_path, _))[1] == '.dat':
                file_nums += 1

        return file_nums

    def getmean(self, data):
        pass


# 获得单个样本
class SingleSample():

    def __init__(self, dataType, filePath, dim=cfg.dim,
                 nChannels=cfg.n_channels, fileType='.dat') -> None:

        self.filePath = filePath
        self.dataType = dataType
        self.dim = dim
        self.nChannels = nChannels
        self.fileType = fileType
        self.fileNums = self.specific_file_num()

        print(
            f'initialization successful! This folder hive {self.fileNums}{fileType}')

    # 返回单个样本
    def getitem(self, id):

        if self.dataType == 'fault':

            gx = np.fromfile(os.path.join(self.filePath, str(
                id)+self.fileType), dtype=np.single)
            gx = np.reshape(gx, self.dim)
            gx = np.transpose(gx)
            data = np.zeros((self.nChannels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.nChannels, *self.dim))

            data = torch.tensor(data)
            data = data.unsqueeze(0)

            return data

        elif self.dataType == 'seis':

            file_id = str(id)+self.fileType
            gx = np.fromfile(os.path.join(
                self.filePath, file_id), dtype=np.single)
            gx = np.reshape(gx, self.dim)

            xm = np.mean(gx)
            xs = np.std(gx)
            gx = gx-xm
            gx = gx/xs
            gx = np.transpose(gx)

            data = np.zeros((self.nChannels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.nChannels, *self.dim))

            data = torch.tensor(data)
            data = data.unsqueeze(0)

            return data

    # 计算指定文件夹下指定类型文件的数量
    def specific_file_num(self) -> int:

        file_nums = 0
        for _ in sorted(os.listdir(self.filePath)):

            if os.path.isdir(os.path.join(self.filePath, _)):
                continue

            if os.path.splitext(os.path.join(self.filePath, _))[1] == self.fileType:
                file_nums += 1

        return file_nums


# 此类未完成，不要使用
class BatchSample():

    def __init__(self, dataType, filePath, fileType, batchSize=cfg.batch_size,
                 dim=cfg.dim, n_channels=cfg.n_channels) -> None:

        self.filePath = filePath
        self.dataType = dataType
        self.fileType = fileType
        self.batchSize = batchSize
        self.dim = dim
        self.n_channels = n_channels

        print('initialization successful!')

    # 返回按batch_size样本
    def get_batchs(self):

        sumData = []

        batchNum = self.specific_file_num()

        if self.dataType == 'fault':

            gx = np.fromfile(self.path+str(self.id)+'.dat', dtype=np.single)
            gx = np.reshape(gx, self.dim)
            gx = np.transpose(gx)
            data = np.zeros((self.n_channels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.n_channels, *self.dim))

            data = torch.tensor(data)
            data = data.unsqueeze(0)

            return data

        elif self.dataType == 'seis':

            gx = np.fromfile(self.path+str(self.id)+'.dat', dtype=np.single)
            gx = np.reshape(gx, self.dim)

            xm = np.mean(gx)
            xs = np.std(gx)
            gx = gx-xm
            gx = gx/xs
            gx = np.transpose(gx)

            data = np.zeros((self.n_channels, *self.dim), dtype=np.single)
            data = np.reshape(gx, (self.n_channels, *self.dim))

            data = torch.tensor(data)
            data = data.unsqueeze(0)

            return data

    # 计算指定文件夹下指定类型文件的数量
    def specific_file_num(self) -> int:

        file_nums = 0
        for _ in sorted(os.listdir(self.filePath)):

            if os.path.isdir(os.path.join(self.filePath, _)):
                continue

            if os.path.splitext(os.path.join(self.filePath, _))[1] == self.fileType:
                file_nums += 1

        return file_nums
