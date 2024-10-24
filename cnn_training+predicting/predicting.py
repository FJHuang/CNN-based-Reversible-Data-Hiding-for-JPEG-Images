import numpy as np
import torch
import cv2
import os
import math
import jpegio as jio
import time
import csv
from model.cnn_predictor import CNNP
import torchvision.transforms as transforms

def PSNR(imagename):
    img1 = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)  # 打开的图片时RGB三通道的，奥处理一下
    img1 = img1.astype(np.float64)
    img2 = cv2.imread("stego.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def INC(imagename):
    img1_size = os.path.getsize(imagename)
    img2_size = os.path.getsize("stego.jpg")
    return (img2_size - img1_size)*8


def main(epoch, model_path, payload, data, csv_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cuda_available = torch.cuda.is_available()

    model = CNNP().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['network'])
    if cuda_available:
        model.cuda()

    image_path = '/home/yangx/yangx/database/ucid100/'
    image_path = '/home/yangx/yangx/database/tmp/'
    image_path = '/home/yangx/yangx/database/kodak/90/'
    image_path = '/home/yangx/yangx/database/uscsipi/80/'
    image_path = '/home/yangx/yangx/database/BOSSbase100/80/'
    # image_path = '/home/yangx/yangx/database/usc8_kodak_90/'
    # coef_path = '/home/yangx/yangx/database/usc_predictioncoef/'
    print(image_path)
    file_list = []
    path_list = os.listdir(image_path)
    for file in path_list:
        file_list.append(os.path.join(image_path, file))

    d = {}
    for z in range(len(path_list)):
        t1 = time.time()

        image_name = image_path + path_list[z]
        jpegObj = jio.read(image_name)
        jpegCoef = jpegObj.coef_arrays[0]

        m, n = jpegCoef.shape

        num_block = m * n // 64
        count0 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                tmp = jpegCoef[i::8, j::8]
                tmp = np.where(tmp, 0, 1)
                count0[i, j] = np.sum(tmp)
                if count0[i, j] == 0:
                    count0[i, j] += 1
                elif count0[i, j] == num_block:
                    count0[i, j] -= 1
        all_lambda = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pz = count0[i, j]
                tmp = -2 * np.log(1 - pz / num_block)
                all_lambda[i, j] = tmp

        def CDF(lambdaa, x):
            return 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-lambdaa * np.abs(x)))

        pc = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pzz = CDF(all_lambda[i, j], 0.5) - CDF(all_lambda[i, j], -0.5)
                pcc = (CDF(all_lambda[i, j], 1.5) - CDF(all_lambda[i, j], 0.5)) * 2
                pc[i, j] = pcc / (1 - pzz)

        prediction_coef = np.zeros((m, n))
        for i in range(8):
            for j in range(8):
                tmp = np.ones((m // 8, n // 8)) * pc[i, j]
                prediction_coef[i::8, j::8] = tmp
        zeros = np.where(jpegCoef, 1, 0)
        prediction_coef = prediction_coef * zeros

        prediction = np.zeros((m // 8, n // 8, 63))
        nonono = np.zeros((m // 8, n // 8, 63))
        c = 0
        for i in range(8):
            for j in range(8):
                if i + j != 0:
                    nonono[:, :, c] = jpegCoef[i::8, j::8]
                    prediction[:, :, c] = prediction_coef[i::8, j::8]
                    c += 1

        inputCoef = transforms.ToTensor()(prediction)
        inputCoef = inputCoef.type(torch.FloatTensor).to('cuda')
        inputCoef = inputCoef.unsqueeze(0)


        model.eval()
        torch.no_grad()
        outputCoef = model(inputCoef)
        outputCoef = outputCoef.cpu()
        outputCoef = outputCoef.squeeze(0).detach().numpy()
        outputCoef = np.transpose(outputCoef, (1, 2, 0))

        outputCoef = np.clip(outputCoef, 0, 1)
        predictionCoef = np.zeros((m, n))
        c = 0
        for i in range(8):
            for j in range(8):
                if i + j != 0:
                    predictionCoef[i::8, j::8] = outputCoef[:, :, c]
                    c += 1
        t2 = time.time()
        d[path_list[z]] = t2 - t1
        # savepath = '/home/yangx/yangx/DEEPJPEG/jpeg_abs/predicted_coef/uscsipi/80/' + path_list[z].replace('jpg', 'npy')
        # np.save(savepath, predictionCoef)
    print(d)

if __name__ == '__main__':
    payload = 10000
    np.random.seed(0)
    data = np.random.randint(0, 2, size=payload)
    # 70
    # epoch = 817
    # model_path = '/home/yangx/yangx/DEEPJPEG/jpeg_abs/runs/yx70 2023.01.09--00-51-02//saved_model/model_state_epoch_' + str(epoch) + '.pth'
    # main(epoch, model_path, payload, data, 'ucid70.csv')
    # 80
    epoch = 785
    model_path = '/home/yangx/yangx/DEEPJPEG/jpeg_abs/runs/yx80 2023.01.04--12-04-19/saved_model/model_state_epoch_' + str(epoch) + '.pth'
    main(epoch, model_path, payload, data, 'ucid80.csv')
    # 90
    # epoch = 852
    # model_path = '/home/yangx/yangx/DEEPJPEG/jpeg_abs/runs/yx 2022.12.30--14-06-29/saved_model/model_state_epoch_' + str(epoch) + '.pth'
    # main(epoch, model_path, payload, data, 'test.csv')
    # 100
    # epoch = 912
    # model_path = '/home/yangx/yangx/DEEPJPEG/jpeg_abs/runs/yx100 2023.01.10--11-54-16/saved_model/model_state_epoch_' + str(epoch) + '.pth'
    # main(epoch, model_path, payload, data, 'ucid100.csv')
