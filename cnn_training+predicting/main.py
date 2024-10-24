import torch
import argparse
import os
import sys
import logging
import utils
import time
import math
from options import Configuration
from model.predict_model import PredictModel
# from memory_profiler import profile
# import profile

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
# @profile
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda') #if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of CNN')
    parser.add_argument('--batch-size', '-b', default=64, type=int, help='The batch size.')
    parser.add_argument('--epochs', '-e', default=1000, type=int, help='Number of epochs to run the simulation.')
    parser.add_argument('--quality-factor', '-qf', default=70, type=int, help='The quality factor of JPEG image')
    parser.add_argument('--name', '-n', default="test", type=str, help='The name of the experiment.')
    parser.add_argument('--runs-folder', '-rf', default=os.path.join('.', 'runs'), type=str, help='The root folder.')
    parser.add_argument('--img-path', '-folder', default="", type=str,
                        help='The place of the image.')

    args = parser.parse_args()
    this_run_folder = utils.create_folder_for_run(args.runs_folder, args.name)
    start_epoch = 1
    config = Configuration(
        batch_size=args.batch_size,
        number_of_epochs=args.epochs,
        this_run_folder=this_run_folder,
        start_epoch=start_epoch,
        experiment_name=args.name,
        quality_factor=args.quality_factor,
        device=device,
        # img_path=os.path.join(args.img_path, str(args.quality_factor)),
        img_path=args.img_path,
        height=512,
        weight=512,
    )
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{args.name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    model = PredictModel(config)
    train(model, config)

def train(model, config):
    file_count = 12800
    steps_in_epoch = file_count // config.batch_size

    # load_path = '/home/yangx263/CNN_RDH/JPEG_CNN/code_newnet_abs/runs/newnet_90_512_noqt 2022.11.09--14-43-45/saved_model/model_state_epoch_11.pth'
    # utils.load_model(load_path, config, model)

    print_each = 50
    warmup_epochs = 40
    baselr = 0.001
    for epoch in range(config.start_epoch, config.number_of_epochs + 1):
        if epoch < warmup_epochs:
            lr = baselr * epoch / warmup_epochs
        else:
            lr = baselr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (config.number_of_epochs - warmup_epochs)))
        model.optimizer.param_groups[0]['lr'] = lr
        print('\nlr:', model.optimizer.param_groups[0]['lr'])
        logging.info('Starting epoch {}/{}'.format(epoch, config.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(config.batch_size, steps_in_epoch))
        losses_train = {}
        epoch_start = time.time()
        step = 1
        ###########################
        # Running in Training set #
        ###########################
        print('preprocessing')
        if epoch % 10 == 1:
            utils.preprocessing(config)
        print('loading and training')
        train_dataloader, test_dataloader = utils.getDataloaders(config)
        for inputCoef, targetCoef in train_dataloader:
            inputCoef = inputCoef.type(torch.FloatTensor).to(config.device)
            targetCoef = targetCoef.type(torch.FloatTensor).to(config.device)
            losses = model.train_on_batch(inputCoef, targetCoef)
            if not losses_train:
                for name in losses:
                    losses_train[name] = []
            for name, loss in losses.items():
                losses_train[name].append(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, config.number_of_epochs, step, steps_in_epoch))
                utils.print_progress(losses_train)
                logging.info('-' * 40)
            # print(step)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(config.this_run_folder, 'train.csv'), losses_train, epoch, train_duration)
        # if epoch % 10 == 0:
        if 1:
            utils.save_model(os.path.join(config.this_run_folder, 'saved_model'), epoch, model)

        #######################
        # Running in Test set #
        #######################
        logging.info('Running Test set for epoch {}/{}'.format(epoch, config.number_of_epochs))
        epoch_start = time.time()
        losses_test = {}
        for inputCoef, targetCoef in test_dataloader:
            inputCoef = inputCoef.type(torch.FloatTensor).to(config.device)
            targetCoef = targetCoef.type(torch.FloatTensor).to(config.device)
            losses = model.test_on_batch(inputCoef, targetCoef)
            if not losses_test:
                for name in losses:
                    losses_test[name] = []
            for name, loss in losses.items():
                losses_test[name].append(loss)
        utils.print_progress(losses_test)
        logging.info('-' * 40)
        utils.write_losses(os.path.join(config.this_run_folder, 'test.csv'), losses_test, epoch,
                           time.time() - epoch_start)

if __name__ == "__main__":
    main()
