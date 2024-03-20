import torch


# configuration
class Config:

    def __init__(self) -> None:

        # parameters
        self.batch_size = 8
        self.dim = (128, 128, 128)
        self.n_channels = 1
        self.shuffle = True
        self.lr = 0.0001  # 文献设置为0.0001，初始轮数，一般0.01~0.001
        self.epoch = 25  # 文献设置为25个训练轮次
        self.threshold = 0.8

        # data_path
        self.seismPathTr = "./data/train/0seis/"
        self.faultPathTr = "./data/train/0fault/"

        self.seismPathVa = "./data/validation/seis/"
        self.faultPathVa = "./data/validation/fault/"

        self.seismPathTe = ''
        self.faultPathTe = ''

        self.seismPathPre = './data/CB/seis/'
        self.faultPathPre = './data/CB/prefault/'

        # data_set_length
        self.train_id = range(200)
        self.valid_id = range(10)
        self.test_id = range(10)

        # device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
