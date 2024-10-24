import os
import random

import torch
import time
import csv
import logging
import cv2
import jpegio as jio
from PIL import Image

import numpy as np
from model import dataset
from torchvision import transforms
from model.predict_model import PredictModel
from torchvision.transforms import InterpolationMode

def getDataloaders(config):
    train_datasets = dataset.ImageDataset('/home/yangx/JPEGRDH/database/tmp_5_predictioncoef', train=True)
    train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   drop_last=True)

    test_datasets = dataset.ImageDataset('/home/yangx/JPEGRDH/database/usc70_predictioncoef', train=False)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1)

    return train_dataloader, test_dataloader
    # return train_dataloader
# @profile
def preprocessing(config):
    datasets = '/home/yangx/JPEGRDH/database/imageNet_test_gray/'
    file_list = []
    path_list = os.listdir(datasets)
    for file in path_list:
        file_list.append(os.path.join(datasets, file))
    file_list = random.sample(file_list, 12800)
    num = 0
    for file_name in file_list:
        img = Image.open(file_name)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),  # 随机翻转
        ])
        img = transform_train(img)
        img = np.array(img)
        # QF = np.random.choice([70, 80, 90, 100])
        QF = 70
        save_name = '/home/yangx/JPEGRDH/database/tmp_5_QF70/' + str(num) + '.jpg'
        cv2.imwrite(save_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])

        jpegObj = jio.read(save_name)
        jpegCoef = jpegObj.coef_arrays[0]

        m, n = jpegCoef.shape
        num_block = m*n//64

        count0 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                tmp = jpegCoef[i::8, j::8]
                tmp = np.where(tmp, 0, 1)
                count0[i, j] = np.sum(tmp)
                # for ii in range(m//8):
                #     for jj in range(n//8):
                #         if jpegCoef[ii*8+i, jj*8+j] == 0:
                #             count0[i, j] += 1
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

        absjpegCoef = np.abs(jpegCoef)
        original_coef = np.where(absjpegCoef == 1, 1, 0)
        # for i in range(m):
        #     for j in range(n):
        #         if np.abs(jpegCoef[i, j]) == 1 and (i % 8) + (j % 8) != 0:
        #             original_coef[i, j] = 1
        #         if jpegCoef[i, j] != 0 and (i % 8) + (j % 8) != 0:
        #             prediction_coef[i, j] = pc[i % 8, j % 8]
        save_name1 = '/home/yangx/JPEGRDH/database/tmp_5_predictioncoef/' + str(num) + '.npy'
        save_name2 = '/home/yangx/JPEGRDH/database/tmp_5_originalcoef/' + str(num) + '.npy'
        np.save(save_name1, prediction_coef)
        np.save(save_name2, original_coef)
        num += 1

def create_folder_for_run(runs_folder, experiment_name):

    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')
    # this_run_folder = os.path.join(runs_folder,f'{experiment_name}')
    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'saved_model'))

    return this_run_folder


def print_progress(losses_accu):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        logging.info(loss_name.ljust(max_len+4) + '{:.4f}'.format(np.mean(loss_value)))


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in losses_accu.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


def save_model(folder, epoch, model: PredictModel):
    file_name = os.path.join(folder, 'model_state_epoch_{}.pth'.format(epoch))
    state = {'network': model.model.state_dict(),
             'optimizer': model.optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, file_name)


def load_model(file_name, config, model: PredictModel):
    checkpoint = torch.load(file_name)
    model.model.load_state_dict(checkpoint['network'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    config.start_epoch = checkpoint['epoch'] + 1

