import torch
import inspect
import re
import os
import csv
import numpy as np
# precision_recall_curve, roc_curve, auc
from torchsummary import summary
from sklearn import metrics as skleval
from config import Config

cfg = Config()


class Evaluate():

    def __init__(self) -> None:
        pass

    def environment_info(self):

        print('CUDA版本:', torch.version.cuda)
        print('cuDNN版本:', torch.backends.cudnn.version())
        print('Pytorch版本:', torch.__version__)
        print('显卡是否可用:', '可用' if (torch.cuda.is_available()) else '不可用')
        print('显卡数量:', torch.cuda.device_count())
        print('是否支持BF16数字格式:', '支持' if (
            torch.cuda.is_bf16_supported()) else '不支持')
        print('当前显卡型号:', torch.cuda.get_device_name())
        print('当前显卡的CUDA算力:', torch.cuda.get_device_capability())
        print('当前显卡的总显存:', torch.cuda.get_device_properties(
            0).total_memory/1024/1024/1024, 'GB')
        print('是否支持TensorCore:', '支持' if (
            torch.cuda.get_device_properties(0).major >= 7) else '不支持')
        print('当前显卡的显存使用率:', torch.cuda.memory_allocated(0) /
              torch.cuda.get_device_properties(0).total_memory*100, '%')

    def model_structure(self, model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|'
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(
            model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

    def model_summary(self, model, input_size=(1, 128, 128, 128)):

        summary(model, input_size=(1, 128, 128, 128))

    def fault_mask(self, data, threshold):

        return data.ge(threshold)

    def _flatten2np(self, data):

        data = data.detach().cpu().numpy().transpose().reshape(-1)
        # data = data.detach().reshape(-1).cpu().numpy()

        return data

    def accuracy(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold)
        pred = self._flatten2np(pred)
        label = self._flatten2np(label)

        return skleval.accuracy_score(label, pred)

    def precision(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold)
        pred = self._flatten2np(pred)
        label = self._flatten2np(label)

        return skleval.precision_score(label, pred)

    def recall(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold)
        pred = self._flatten2np(pred)
        label = self._flatten2np(label)

        return skleval.recall_score(label, pred)

    def pr_curve(self, logits, label):

        logits = self._flatten2np(logits)
        label = self._flatten2np(label)

        return skleval.precision_recall_curve(label, logits)

    def roc_curve(self, logits, label):

        logits = self._flatten2np(logits)
        label = self._flatten2np(label)

        return skleval.roc_curve(label, logits)

    def pr_auc(self, logits, label):

        logits = self._flatten2np(logits)
        label = self._flatten2np(label)

        return skleval.average_precision_score(label, logits)

    def roc_auc(self, logits, label):

        logits = self._flatten2np(logits)
        label = self._flatten2np(label)

        return skleval.roc_auc_score(label, logits)

    def auc(self, logits, label):
        return
        precision, recall, _ = self.pr_curve(logits, label)
        return skleval.auc(recall, precision)

    def min_max(self, data):
        # 计算数据的最小值和最大值
        min_val = np.min(data)
        max_val = np.max(data)

        # 对数据进行归一化处理
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def norm(self, data):
        # 计算数据的平均值和标准差
        mean_val = np.mean(data)
        std_dev = np.std(data)

        # 对数据进行标准化处理
        standardized_data = (data - mean_val) / std_dev
        return standardized_data

    # 准确率，真阳性和真阴性，Accuracy = (TP+TN)/(TP+TN+FP+FN)

    def accuracy_custom(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold).float()
        correct = torch.eq(pred, label).float().sum().item()
        print('相等点的数量:', correct)
        total = label.numel()
        print('总共点的数量:', total)

        return correct / total

    # 精准率，只关注真阳性，Precision = TP/(TP+FP)

    def precision_custom(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold).float()
        # True Positive
        tp = pred[(pred == 1.) & (pred == label)].float().sum().item()
        # False Positive
        fp = pred[(pred == 1.) & (pred != label)].float().sum().item()
        # print('预测到真实的点:', tp, '预测错真实的点:', fp)
        # print('精准率:', tp / (tp + fp))
        return (tp / (tp + fp))

    # Recall = TP/(TP+FN)

    def recall_custom(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold).float()
        # true positive
        tp = pred[(pred == 1.) & (pred == label)].float().sum().item()

        return (tp / label.float().sum().item())

    # TPR = TP/(TP+FN)

    def sensitivity_custom(self, logits, label, threshold):
        pass

    # False Positive Rate, FPR = 1 - TN/(FP+TN) = FP/(FP+TN)

    def fpr_custom(self, logits, label, threshold):

        pred = self.fault_mask(logits, threshold).float()
        fp = pred[(pred == 1.) & (pred != label)].float().sum().item()
        tn = pred[(pred == 0.) & (pred == label)].float().sum().item()

        return fp / (fp + tn)

    def save_data(self, ratetype, id, rate):

        with open(os.path.join(f'data/rate/{ratetype}.csv'), mode='a+', newline='') as f:

            writer = csv.writer(f)
            writer.writerow([id, rate])

    # 提取变量名为字符串

    def varname(self, p):

        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:

            m = re.search(
                r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)

        if m:

            return m.group(1)
