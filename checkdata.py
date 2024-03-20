import os
import hashlib
import csv
import numpy as np
from config import Config

cfg = Config()


class CheckData():

    def __init__(self) -> None:
        pass

    # 检查上传文件中的缺失文件
    def check_files_exist(self, filePath: str, fileType='dat') -> None:

        files = self.load_sorted_file(filePath, fileType)
        flag = True

        print('files:', len(files))

        for i in range(len(files)):
            if not os.path.exists(filePath + f'{i}.dat'):
                if flag:
                    flag = False
                print('file', i, "is not exsist!")

        if flag:
            print('No file lost!')

    # 检查上传文件中数据尺寸损坏的文件
    def check_files_shape(self, filePath: str, fileType='dat', const_shape=None) -> None:

        files = self.load_sorted_file(filePath, fileType)
        flag = True  # 提示标志
        num = 0

        if const_shape == None:
            const_shape = np.fromfile(os.path.join(
                filePath, files[0]), dtype=np.single).shape
            # print(const_shape)
            # print(files[0])

        for i in range(len(files)):

            data_shape = np.fromfile(
                os.path.join(filePath, files[i]), dtype=np.single).shape

            if data_shape != const_shape:

                if flag:
                    flag = False

                print('file', i, "is not matched!", 'shape is', data_shape)
                num += 1

        if flag:
            print('All file shape is matched!')
        else:
            print(num, 'files is not matched!')

    # 检验数据内容损坏的文件，图方便直接运行此函数
    def check_files_md5(self, filePath: str, fileName='md5Catalog.csv') -> None:

        catalog = os.path.join(str(filePath), fileName)
        flag = True
        num = 0

        if filePath == None or not os.path.exists(catalog):
            print('No \'md5Catalog.csv\' file found!')
            return

        with open(catalog, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                name, md5 = row
                if md5 != self._calculate_md5(os.path.join(filePath, name)):

                    if flag:
                        flag = False

                    print('file', name, "md5 is uncorrect!")
                    num += 1
        if flag:
            print('All files md5 is correct!')
        else:
            print(num, ' files md5 is uncorrect!')

    # 生成MD5文件校验目录
    def generate_md5_catalog(self, filePath: str,
                             fileType='dat', fileName='md5Catalog.csv') -> None:

        catalog = os.path.join(str(filePath), fileName)
        print(catalog)

        if filePath == None or not os.path.exists(catalog):

            print('\'md5Catalog.csv\' not exist! Now generating...')

            '''
            打开一个文件只用于读写。如果该文件已存在则打开文件，并从开头开始编辑，
            即原有内容会被删除。如果该文件不存在，创建新文件。
            '''
            with open(catalog, mode='w+', newline='') as f:

                writer = csv.writer(f)
                files = self.load_sorted_file(filePath, fileType)

                for name in files:
                    md5 = self._calculate_md5(os.path.join(filePath, name))
                    writer.writerow([name, md5])
                print(f'{len(files)} files have md5 and writen!')
        else:
            print('\'md5Catalog.csv\' exist!')

    # 计算单个文件的md5
    def _calculate_md5(self, filePath: str):
        with open(filePath, 'rb') as file:
            data = file.read()

        # 创建MD5对象并计算文件的MD5值
        md5 = hashlib.md5()
        md5.update(data)

        return md5.hexdigest()

    # 检查切割的数据块是否和源数据一致
    def check_slice_data(self, ori_data: np.ndarray, new_data: np.ndarray) -> bool:

        return np.equal(ori_data, new_data).sum() == ori_data.size

    # 计算一个文件夹下某种类型的文件数
    def specific_file_num(self, file_path: str, fileType='dat') -> int:

        file_nums = 0
        for _ in sorted(os.listdir(file_path)):

            if os.path.isdir(os.path.join(file_path, _)):
                continue

            if os.path.splitext(os.path.join(file_path, _))[1] == fileType:
                file_nums += 1

        return file_nums

    # 返回按规范文件名排序的指定类型文件列表
    def load_sorted_file(self, filePath: str, fileType='dat') -> list:

        files = []

        for name in os.listdir(filePath):

            path = os.path.join(filePath, name)

            if os.path.isfile(path) and name.endswith(fileType):
                files.append(name)

        files.sort(key=lambda x: int(x.split('.')[0]))

        return files
