import torch
from options import Configuration
from model.cnn_predictor import CNNP
import torch.nn as nn


class PredictModel:
    def __init__(self, config: Configuration):
        super(PredictModel, self).__init__()

        self.model = CNNP().to(config.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=1e-5)
        self.myloss = nn.MSELoss()
        self.config = config

    # @profile
    def train_on_batch(self, input_image, target_image):

        self.model.train()
        with torch.enable_grad():
            predicted_image = self.model(input_image)
            self.optimizer.zero_grad()
            g_loss = self.myloss(predicted_image, target_image)
            g_loss.backward()
            self.optimizer.step()

        losses = {
            'loss           ': g_loss.item(),
        }

        return losses

    def test_on_batch(self, input_image, target_image):

        self.model.eval()
        # with torch.no_grad():
        #     predicted_image = self.model(input_image)
        #     g_loss = self.myloss(predicted_image, target_image)
        torch.no_grad()
        predicted_image = self.model(input_image)
        g_loss = self.myloss(predicted_image, target_image)
        losses = {
            'loss           ': g_loss.item(),
        }

        return losses

    def to_stirng(self):
        return '{}\n'.format(str(self.model))


