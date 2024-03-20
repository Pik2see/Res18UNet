import numpy as np
import os
import glob


def main():

    enhance_data()


def enhance_data():

    root = '1/wu_reproduce_with_pytorch/data/train/'
    tsFiles = os.path.join(root, '0seis/')
    tfFiles = os.path.join(root, '0fault/')
    # seisPath = os.path.join(root, 'oriSeis/')
    # faultPath = os.path.join(root, 'oriFault/')

    seisPath = tsFiles
    faultPath = tfFiles
    # print(len(os.listdir(trainSeis)))
    # print(seisPath)
    # print(faultPath)

    # 将文件名按正常升序排列
    seisFiles = os.listdir(seisPath)
    seisFiles.sort(key=lambda x: int(x.split('.')[0]))
    faultFiles = os.listdir(faultPath)
    faultFiles.sort(key=lambda x: int(x.split('.')[0]))
    print(len(seisFiles), len(faultFiles))
    print(len(os.listdir(tsFiles)), len(os.listdir(tfFiles)))
    # print(len(os.listdir(trainFiles)))

    for i in range(len(seisFiles)):
        break
        seis = os.path.join(seisPath, f'{i}.dat')
        fault = os.path.join(faultPath, f'{i}.dat')
        print(seis)
        # break

        # 读取数据
        sx = np.fromfile(seis, dtype=np.single)
        fx = np.fromfile(fault, dtype=np.single)

        # 数据增强操作,
        # sx, fx = transpose(sx, fx)
        sx, fx = rotate(sx, fx, 1)
        print(sx.shape)
        # break

        # 写入数据
        sx.tofile(
            f'1/wu_reproduce_with_pytorch/data/train/enseis/{len(seisFiles)+i}.dat')
        fx.tofile(
            f'1/wu_reproduce_with_pytorch/data/train/enfault/{len(faultFiles)+i}.dat')
        # break


def transpose(sx, fx):

    sx = np.reshape(sx, (128, 128, 128))
    fx = np.reshape(fx, (128, 128, 128))

    sx = np.transpose(sx)
    fx = np.transpose(fx)

    sx = np.flipud(sx)
    fx = np.flipud(fx)

    sx = np.transpose(sx)
    fx = np.transpose(fx)

    sx = sx.reshape(-1)
    fx = fx.reshape(-1)

    return sx, fx


def rotate(sx, fx, times):

    sx = np.reshape(sx, (128, 128, 128))
    fx = np.reshape(fx, (128, 128, 128))

    sx = np.transpose(sx)
    fx = np.transpose(fx)

    for i in range(times):

        sx = np.rot90(sx, i, (2, 1))
        fx = np.rot90(fx, i, (2, 1))

    sx = np.transpose(sx)
    fx = np.transpose(fx)

    sx = sx.reshape(-1)
    fx = fx.reshape(-1)

    return sx, fx


def save_data(data: np.ndarray, path: str, name: str) -> None:

    data.tofile(path+name)


def save_dataT(data: np.ndarray, path: str, name: str) -> None:

    data.transpose().tofile(path+name)
    # print(f'The data{data.shape} has already been saved in {path}')


if __name__ == '__main__':
    main()
