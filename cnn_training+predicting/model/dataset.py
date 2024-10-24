import torch.utils.data as data
import os
import numpy as np
import jpegio as jio
import cv2
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

def get_file_list(folder):
    file_list = []
    path_list = os.listdir(folder)
    for file in path_list:
        file_list.append(os.path.join(folder, file))
    return file_list

zigzagOrder = [[0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [0, 2], [0, 3], [1, 2],
               [2, 1], [3, 0], [4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [0, 5],
               [1, 4], [2, 3], [3, 2], [4, 1], [5, 0], [6, 0], [5, 1], [4, 2],
               [3, 3], [2, 4], [1, 5], [0, 6], [0, 7], [1, 6], [2, 5], [3, 4],
               [4, 3], [5, 2], [6, 1], [7, 0], [7, 1], [6, 2], [5, 3], [4, 4],
               [3, 5], [2, 6], [1, 7], [2, 7], [3, 6], [4, 5], [5, 4], [6, 3],
               [7, 2], [7, 3], [6, 4], [5, 5], [4, 6], [3, 7], [4, 7], [5, 6],
               [6, 5], [7, 4], [7, 5], [6, 6], [5, 7], [6, 7], [7, 6], [7, 7]]

def zigzag(Coef):
    m, n = Coef.shape
    EmbCoef = np.zeros((m//8, n//8, 64))
    for i in range(m//8):
        for j in range(n//8):
            for k in range(1, 64):
                EmbCoef[i, j, k] = Coef[i * 8 + zigzagOrder[k][0], j * 8 + zigzagOrder[k][1]]
    return EmbCoef
# @profile
def image_division(file_name, train=True):
    original_filename = file_name.replace('predictioncoef', 'originalcoef')
    prediction_coef = np.load(file_name)
    original_coef = np.load(original_filename)
    m, n = prediction_coef.shape
    prediction = np.zeros((m // 8, n // 8, 63))
    original = np.zeros((m // 8, n // 8, 63))
    c = 0
    for i in range(8):
        for j in range(8):
            if i + j != 0:
                prediction[:, :, c] = prediction_coef[i::8, j::8]
                original[:, :, c] = original_coef[i::8, j::8]
                c += 1

    # prediction_coef = zigzag(prediction_coef)
    # original_coef = zigzag(original_coef)

    return prediction, original

class ImageDataset(data.Dataset):
    def __init__(self, folders, train=True):
        self.folder_roots = folders
        self.all_files = get_file_list(folders)
        self.train = train

    def __getitem__(self, index):
        from copy import deepcopy

        inputCoef, targetCoef = image_division(self.all_files[index], self.train)
        inputCoef = deepcopy(inputCoef)
        targetCoef = deepcopy(targetCoef)
        inputCoef = transforms.ToTensor()(inputCoef)
        targetCoef = transforms.ToTensor()(targetCoef)

        return inputCoef, targetCoef

    def __len__(self):
        return len(self.all_files)
