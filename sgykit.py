import segyio
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from config import Config

cfg = Config()


class Sgykit():

    # 初始化，‘r’为只读，‘r+’为可读写
    def __init__(
            self, fileName='',
            sliceSize={'iline': 128, 'xline': 128, 'time': 128},
            mode='r', iline=189, xline=193, strict=True,
            ignore_geometry=False) -> None:

        # fileName = 'D:\Work\Project\Geomodeling\CB\CB_depth\Chengbei_2020_3D_psdm_final_depth.segy'
        # fileName = 'D:\Work\Project\Geomodeling\CB\CB_depth\sample.segy'
        self.mode = mode

        self.openPar = {'iline': iline, 'xline': xline, 'strict': strict,
                        'ignore_geometry': ignore_geometry}

        if fileName != '':

            with segyio.open(filename=fileName, mode=mode, iline=iline,
                             xline=xline, strict=strict,
                             ignore_geometry=ignore_geometry) as file:

                file.mmap()  # mmap方法可以加速数据读取

                self.iline0 = file.ilines[0]  # 起始inline
                self.xline0 = file.xlines[0]  # 其实crossline
                # 采样间隔
                self.smpInterval = file.bin[segyio.BinField.Interval] / 1000
                # 数据尺寸
                self.ixtShape = (len(file.ilines),
                                 len(file.xlines),
                                 len(file.samples))

            print('sgy shape is :', self.ixtShape)

            if sliceSize != 0:
                iSection = len(file.ilines) // sliceSize['iline']
                iRemainder = len(file.ilines) % sliceSize['iline']
                xSection = len(file.xlines) // sliceSize['xline']
                xRemainder = len(file.xlines) % sliceSize['xline']
                tSection = len(file.samples) // sliceSize['time']
                tRemainder = len(file.samples) % sliceSize['time']
                print(
                    f'Inline can be sliced {iSection} section. There are {iRemainder} remainder.')
                print(
                    f'Crossline can be sliced {xSection} section. There are {xRemainder} remainder.')
                print(
                    f'Time can be sliced {tSection} section. There are {tRemainder} remainder.')
                print('Blocks is', iSection*xSection*tSection)

    # 数据轮廓
    def sgy_outline(self, fileName: str) -> None:

        with segyio.open(fileName, self.mode, **self.openPar) as file:

            file.mmap()  # mmap方法可以加速数据读取

            print("About '{}'".format(fileName))
            print("Format type: {}".format(file.format))
            print("Offset count: {}".format(file.offsets))
            print("Samples: {}".format(file.samples))
            print("Unstructured: {}".format(file.unstructured))
            print("Ext_headers: {}".format(file.ext_headers))
            print("ilines: {}".format(", ".join(map(str, file.ilines))))
            print("xlines: {}".format(", ".join(map(str, file.xlines))))

    # 数据细节，地震数据文件头
    def sgy_detail(self, fileName: str) -> None:

        with segyio.open(fileName, strict=False) as s:

            s.mmap()  # mmap方法可以加速数据读取
            # Read the data
            data3D = np.stack([f.astype(np.float32) for f in s.trace], axis=0)
            print(data3D.shape)

            # Get the (x,y) locations.
            x = [t[segyio.TraceField.GroupX]for t in s.header]
            y = [t[segyio.TraceField.GroupY]for t in s.header]
            print('(x,y) locations:', len(x), len(y))

            # Get the troce numbers.
            cdp = np.array([t[segyio.TraceField.CDP]for t in s.header])
            print('troce numbers:', cdp.shape)

            # Get the first textuol heoder.
            header = s.text[0].decode('ascii')
            formatted = '\n'.join(chunk for chunk in self._chunks(header, 80))

            # Get data from the binary header.
            # Get the somple intervol in ms (convert from microsec)
            sample_interval = s.bin[segyio.BinField.Interval] / 1000
            print('sample_interval:', sample_interval, 'ms')

        print(formatted)

    def _chunks(self, s, n):

        for start in range(0, len(s), n):

            yield s[start: start+n]

    # 读取二维数据
    def read2D(self, fileName: str) -> np.ndarray:

        with segyio.open(fileName, self.mode, **self.openPar) as file:

            file.mmap()  # mmap方法可以加速数据读取
            # np.asarray浅拷贝，可以保证原始数据不被篡改
            data2D = np.asarray([np.copy(x) for x in file.trace[:]]
                                )  # np.asarray浅拷贝，可以保证原始数据不被篡改
        # print(data.shape)
        return data2D  # (inline*crossline, time)

    # 读取三维数据
    def read3D(self, fileName: str) -> np.ndarray:

        with segyio.open(fileName, self.mode, **self.openPar) as file:

            file.mmap()  # mmap方法可以加速数据读取
            # segy文件为2维数据，在纵轴上将数据打平，此工具还原为3维
            data3D = segyio.tools.cube(file)  # (inline, crossline, time)

        return data3D

    # 自定义读取三维数据方法
    def read3D_custom(self, fileName: str) -> np.ndarray:

        with segyio.open(fileName, self.mode, **self.openPar) as file:

            file.mmap()  # mmap方法可以加速数据读取

            x_len = len(file.xlines)
            i_len = len(file.ilines)
            data3D = []  # 存放三维数据
            data2D = []  # 存放二维数据片
            for i in range(len(file.trace)):
                data2D.append(file.trace[i])  # 地震道数据复制到列表中
                if not (i+1) % x_len:
                    data3D.append(data2D)
                    data2D = []  # 每隔x_len道将data2D清空一次
            data3D = np.array(data3D)  # (inline, crossline, time)
            # print(data3D.shape)
        return data3D

    # inline切面
    def plt_il_section(self, data3D: np.ndarray, iline: int,
                       clip=1e+2, max_figsize=(6, 15)) -> None:

        vmin, vmax = -clip, clip   # 显示范围，负值越大越明显

        trace_numbers = range(self.xline0, self.xline0 +
                              data3D.shape[1], 1)  # 道索引的范围
        time_samples = range(0, data3D.shape[2]*4, 4)  # 时间样本的范围

        data_shape = data3D[iline, :, :].shape
        # plt.figure(figsize=(300, 100))
        # 计算自适应的图像大小
        width_factor = min(1.0, max_figsize[0] / data_shape[1])
        height_factor = min(1.0, max_figsize[1] / data_shape[0])

        figsize = (width_factor * data_shape[1], height_factor * data_shape[0])

        fig, ax = plt.subplots(figsize=figsize)  # 设置图像大小

        plt.imshow(data3D[iline, :, :].transpose(),
                   cmap=plt.cm.seismic,
                   interpolation='nearest',
                   #    aspect=2.,
                   aspect='auto',  # 使用'auto'自适应数据的纵横比
                   vmin=vmin,
                   vmax=vmax,
                   extent=[min(trace_numbers), max(trace_numbers),
                           max(time_samples), min(time_samples)],
                   origin='upper')  # 设置原点为左上角

        # 反转Y轴方向，使原点位于左上方
        # ax.invert_yaxis()

        # 设置x轴的位置在上方
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # 设置y轴的位置在左侧
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')

        # 调整坐标轴位置
        ax.tick_params(axis='both', which='both', direction='in',
                       left=True, top=True, right=False, bottom=False)

        plt.xlabel('Inline number')
        plt.ylabel('Time sample')
        plt.show()

    # crossline切面
    def plt_xl_section(self, data3D: np.ndarray, xline: int,
                       clip=1e+2, max_figsize=(6, 15)) -> None:

        vmin, vmax = -clip, clip   # 显示范围，负值越大越明显

        trace_numbers = range(self.iline0, self.iline0 +
                              data3D.shape[1], 1)  # 道索引的范围
        time_samples = range(0, data3D.shape[2]*4, 4)  # 时间样本的范围

        data_shape = data3D[:, xline, :].shape

        # 计算自适应的图像大小
        width_factor = min(1.0, max_figsize[0] / data_shape[1])
        height_factor = min(1.0, max_figsize[1] / data_shape[0])

        figsize = (width_factor * data_shape[1], height_factor * data_shape[0])

        fig, ax = plt.subplots(figsize=figsize)  # 设置图像大小

        plt.imshow(data3D[:, xline, :].transpose(),
                   cmap=plt.cm.seismic,
                   interpolation='nearest',
                   #    aspect=2.,
                   aspect='auto',  # 使用'auto'自适应数据的纵横比
                   vmin=vmin,
                   vmax=vmax,
                   extent=[min(trace_numbers), max(trace_numbers),
                           max(time_samples), min(time_samples)],
                   origin='upper')  # 设置原点为左上角

        # 反转Y轴方向，使原点位于左上方
        # ax.invert_yaxis()

        # 设置x轴的位置在上方
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # 设置y轴的位置在左侧
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('left')

        # 调整坐标轴位置
        ax.tick_params(axis='both', which='both', direction='in',
                       left=True, top=True, right=False, bottom=False)

        plt.xlabel('Crossline number')
        plt.ylabel('Time sample')
        plt.show()

    # 剖面
    def plt_base_map(self, data3D: np.ndarray, time: int, clip=1e+2) -> None:

        vmin, vmax = -clip, clip   # 显示范围，负值越大越明显

        plt.imshow(data3D[:, :, time],
                   cmap=plt.cm.seismic,
                   interpolation='nearest',
                   aspect=1,
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower')
        plt.xlabel('Crossline number')
        plt.ylabel('Inline sample')

    # 只能用于简单数据
    def plt_il_section_ori(self, data3D: np.ndarray, iline: int, clip=1e+2) -> None:

        # 直接对原始数据可视化
        clip = 1e+2  # 显示范围，负值越大越明显
        vmin, vmax = -clip, clip

        # 彩色图像
        plt.imshow(data3D[:, iline, :].transpose(),
                   cmap=plt.cm.seismic,
                   interpolation='nearest',
                   aspect=1,
                   vmin=vmin,
                   vmax=vmax,
                   origin='upper')

        plt.xlabel('Trace number')
        plt.ylabel('Time sample')

    # 将输入的数据直接保存
    def save_dat(self, data: np.ndarray, path: str) -> None:

        data.tofile(path)
        print(f'The data{data.shape} has already been saved in {path}')

    # 切割数据
    def split_sgy(self, data: np.ndarray,
                  path: str,
                  isSave=False,
                  size={'i': 128, 'x': 128, 't': 128},
                  bias={'i': 0, 'x': 0, 't': 0}) -> np.ndarray:

        # 数据块大小
        isize, xsize, tsize = size['i'], size['x'], size['t']
        # 起始切割偏移量
        ibias, xbias, tbias = bias['i'], bias['x'], bias['t']
        # 数据shape
        ilen, xlen, tlen = data.shape[0], data.shape[1], data.shape[2]
        # 切割起始点
        istart, xstart, tstart, = ibias, xbias, tbias
        # 经过偏移的数据可以切割的整数块数
        istep, xstep, tstep = ((ilen - ibias) // isize,
                               (xlen - xbias) // xsize,
                               (tlen - tbias) // tsize)
        # 切下来的完整数据块
        integralSlicedData = data[istart: istart+isize*istep,
                                  xstart: xstart+xsize*xstep,
                                  tstart: tstart+tsize*tstep]
        # 将切下来的小数据块存入列表返回
        # dbList = []
        num = 0

        for t in range((tlen-tbias)//tsize):

            # tstart = t * tsize

            for x in range((xlen-xbias)//xsize):

                # xstart = x * xsize

                for i in range((ilen-ibias)//isize):

                    # istart = i * isize
                    db = data[istart:istart+isize,
                              xstart:xstart+xsize,
                              tstart:tstart+tsize]
                    # print(db.shape)
                    # dbList.append(db)

                    istart += isize
                    num += 1

                    if isSave:

                        db.tofile(path+f'{num-1}.dat')

                        with open(path+'datashape.csv', mode='w+', newline='') as f:

                            writer = csv.writer(f)
                            writer.writerow((i+1, x+1, t+1))

                istart = ibias
                xstart += xsize

            xstart = xbias
            tstart += tsize

        if isSave:

            with open(path+'datashape.txt', mode='w+', newline='') as f:

                f.write(
                    f'''Original data shape(inline, crossline, time) is {data.shape}
                        Sliced data shape(inline, crossline, time) is {integralSlicedData.shape}.
                        There are {(i+1)*(x+1)*(t+1)} data blocks, 
                        the blocks shape(inline, crossline, time) is {(i+1, x+1, t+1)}.
                    ''')

            with open(path+'datashape.csv', mode='a+', newline='') as f:

                writer = csv.writer(f)
                writer.writerow(integralSlicedData.shape)
                writer.writerow(data.shape)

        return integralSlicedData

    # 重建数据
    def reconstruct_sgy(self, path: str, size={'i': 128, 'x': 128, 't': 128},
                        fielType='dat') -> np.ndarray:

        # 读取数据块信息
        with open(path+'datashape.csv', mode='r') as f:

            reader = csv.reader(f)

            # for row in reader:
            #     for i in range(len(row)):
            #         row[i] = int(row[i])
            row = next(reader)
            for i in range(len(row)):
                row[i] = int(row[i])

        isize, xsize, tsize = size['i'], size['x'], size['t']
        iline, xline, time = row[0], row[1], row[2]
        # print(iline, xline, time)

        datalist = self.load_sorted_file(path, fielType)

        if iline * xline * time != len(datalist):
            print(iline, xline, time)
            raise ValueError(
                f'Shape not matched! {(iline*xline*time)}!={len(datalist)}')

        data1D = []
        data2D = []
        data3D = []

        for i in range(len(datalist)):

            db = np.fromfile(os.path.join(path, f'{i}.dat'), dtype=np.single)
            db = db.reshape(isize, xsize, tsize)

            if type(data1D) != np.ndarray:

                data1D = db

            else:

                data1D = np.concatenate((data1D, db), axis=0)

            if not (i+1) % iline:

                if type(data2D) != np.ndarray:
                    data2D = data1D
                    data1D = []
                else:
                    data2D = np.concatenate((data2D, data1D), axis=1)
                    data1D = []

            if not (i+1) % (xline*iline):

                if type(data3D) != np.ndarray:
                    data3D = data2D
                    data2D = []
                else:
                    data3D = np.concatenate((data3D, data2D), axis=2)
                    data2D = []

        return data3D

    def load_sorted_file(self, filePath: str, fileType='dat') -> list:

        files = []

        for name in os.listdir(filePath):

            path = os.path.join(filePath, name)

            if os.path.isfile(path) and name.endswith(fileType):
                files.append(name)

        files.sort(key=lambda x: int(x.split('.')[0]))

        return files

    # 平滑数据
    def smooth_sgy(
            self,
            data: np.ndarray,
            # fileName,
            # dataSize,
            patchSize=(128, 128, 128)):

        patchSize = np.array(patchSize)
        patchStride = patchSize // 2
        # data = np.fromfile(fileName, dtype=np.single)
        # data = data.reshape(dataSize)

        result = np.zeros(data.shape)
        normalization = np.zeros(data.shape)
        gaussian_map = self.get_gaussian(patchSize)
        for i in range(0, data.shape[0]-patchSize[0]+1, patchStride[0]):
            for j in range(0, data.shape[1]-patchSize[1]+1, patchStride[1]):
                for k in range(0, data.shape[2]-patchSize[2]+1, patchStride[2]):
                    patch = data[i:i+patchSize[0],
                                 j:j+patchSize[1],
                                 k:k+patchSize[2]].astype(np.float32)
                    patch *= gaussian_map
                    normalization[i:i+patchSize[0],
                                  j:j+patchSize[1],
                                  k:k+patchSize[2]] += gaussian_map
                    result[i:i+patchSize[0], j:j+patchSize[1],
                           k:k+patchSize[2]] += patch
        result /= normalization
        return result.astype(np.single)

    # 获取高斯滤波
    def get_gaussian(self, s=(128, 128, 128), sigma=1.0/8) -> np.ndarray:
        temp = np.zeros(s)
        # print('temp:', temp)
        # print('sigma:', sigma)
        coords = [i // 2 for i in s]
        # print('coords:', coords)
        sigmas = [i * sigma for i in s]
        # print('sigmas:', sigmas)
        temp[tuple(coords)] = 1
        # print('tuple(coords):', tuple(coords))
        # print('temp[tuple(coords)]:', temp[(1, 1)])
        gaussian_map = gaussian_filter(
            temp, sigmas, 0, mode='constant', cval=0)
        gaussian_map /= np.max(gaussian_map)
        return gaussian_map

    # 从dat（二进制文件）读取数据
    def read_dat(self, path):
        # path = 'H:/LocalDisk/project/CuanDongBei/slice/pred/resunet/leakyrelu/'
        fullpred = np.fromfile(path+'predb.datt', dtype=np.single)
        with open(path+'datashape.csv', mode='r') as f:
            reader = csv.reader(f)
            row = next(reader)
            row = next(reader)
            for i in range(len(row)):
                row[i] = int(row[i])
        datashape = tuple(row)
        fullpred = fullpred.reshape(datashape)
        # print(fullpred.shape)
        return fullpred
