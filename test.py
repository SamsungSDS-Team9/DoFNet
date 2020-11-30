from __future__ import print_function
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import random
from sklearn.metrics import average_precision_score, accuracy_score
import data
import torch.backends.cudnn as cudnn

# from dataset_load import dataset_loader
from model_load import model_loader

import argparse
from torch.backends import cudnn
import data

class test():
    def __init__(self, config, trainDataset, valDataset, testDataset):
        self.debug = config.debug
                    
        self.test_display = config.test_display
        testDataloader = []
        for test_data in testDataset:
            test_dataset = data.CustomDataset(test_data, config.resize)
            testDataloader.append(
                torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                            pin_memory=True))

        print('dataload end')
        print("test set : {}".format(test_dataset.__len__()))
        classes = ('real', 'fake')
        (self.testloader, self.classes) = (testDataloader, classes)
        # model loading...
        (net, self.feat_size) = model_loader(config.model_name)

        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        print(os.path.isfile(config.PATH))
        load_weight = torch.load(config.PATH)
        try:
            old_weight = load_weight['net']
            new_weight = self.net.state_dict()
            for key, val in old_weight.items():
                new_weight[key.split('module.')[-1]] = old_weight[key]
                
            self.net.load_state_dict(new_weight)
        except:
            new_weight = load_weight
            self.net.load_state_dict(new_weight)


        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.deterministic = True
            cudnn.benchmark = False

        # loss loading...
        self.criterion = nn.CrossEntropyLoss()

        # optimizer loading...
        self.init_optimizer = optim.SGD(net.parameters(), lr=config.init_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        self.last_optimizer = optim.SGD(net.parameters(), lr=config.last_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        self.test()

    def test(self):

        # network initialization...
        self.net.eval()
        for data_name_index, test_data in enumerate(self.testloader):
            # logging initialization...
            test_loss = 0
            correct = 0
            total = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                # batch iterations...
                for batch_idx, (inputs, targets) in enumerate(test_data):
                    # forward
                    inputs1 = inputs[:, :3]
                    inputs2 = inputs[:, 3:]
                    inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(self.device)
                    (outputs, _, _) = self.net.module.forward_dual(inputs1, inputs2)

                    # loss estimation (for logging)
                    loss = self.criterion(outputs, targets)

                    # logging...
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    prediction = torch.max(outputs.data, 1)[1]

                    y_pred.extend(prediction)
                    y_true.extend(targets.flatten().tolist())
                    # print(prediction)
                    # correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                acc = accuracy_score(y_true, y_pred > 0.5)
                ap = average_precision_score(y_true, y_pred)
                print('test display type : ', self.test_display[data_name_index], 'acc : ', acc, ', ap : ', ap, ', total : ', total, ', correct : ', correct)
            # Print Test Result
            print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def main(config):
    trainDataset, valDataset, testDataset = data.get_display_path(config)
    test(config, trainDataset, valDataset, testDataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='RESNET18', type=str, help='name of training model')
    parser.add_argument('--init_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay factor for loss')
    parser.add_argument('--ckpt_folder_root', default='./result', type=str, help='folder name for ckpt folder')
    parser.add_argument('--last_lr', default=0.01, type=float, help='last learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for momentum SGD')
    parser.add_argument('--lr_change_epoch', default=40, type=int, help='epoch when the learning rate changes')

    parser.add_argument('--resize', type=int, default=256, help='image size for resize')
    parser.add_argument('--crop_rate', type=float, default=0.99)
    parser.add_argument('--is_random_crop', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_display', type=str, nargs='+', default=['display'], choices=['display', 'display_s', 'projector']) # apple, samsung, projector
    parser.add_argument('--valid_display', type=str, nargs='+', default=['display', 'display_s', 'projector'], choices=['display', 'display_s', 'projector'])
    parser.add_argument('--test_display', type=str, nargs='+', default=['display', 'display_s', 'projector'], choices=['display', 'display_s', 'projector'])

    parser.add_argument('--root_path', type=str, default='../../new_dataset_all_split')
#     parser.add_argument('--PATH', type=str, default='./result/model_ckpt.pth', help='model_path')
#     parser.add_argument('--PATH', type=str, default='../MpF/result/adv_learn/model_ckpt_103_16_50_0.4_0.3_0.3.pth', help='model_path') # 105
    parser.add_argument('--PATH', type=str, default='../MpF/result/model_ckpt_105.pth', help='model_path') # 105
    
    
    parser.add_argument('--debug', type=bool, default=True)

    config = parser.parse_args()
    main(config)

