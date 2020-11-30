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


class train():
    def __init__(self, config, trainDataset, valDataset, testDataset):
        self.lmd1 = config.lmd1
        self.lmd2 = config.lmd2
        self.lmd3 = config.lmd3
        
        self.debug = config.debug
        self.epochs = config.epochs
        self.lr_change_epoch = config.lr_change_epoch
        print("config.rand_seed: ", config.rand_seed)
        self.rand_seed = config.rand_seed
        self.batch_size = config.batch_size
                    
        self.test_display = config.test_display
        self.valid_display = config.valid_display

        # random seed arrange
        np.random.seed(config.rand_seed)
        torch.manual_seed(config.rand_seed)
        train_dataset = data.CustomDataset(trainDataset, config.resize)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=8, pin_memory=True)
        testDataloader = []
        for test_data in testDataset:
            test_dataset = data.CustomDataset(test_data, config.resize)
            testDataloader.append(
                torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                            pin_memory=True))

        valDataloader = []
        for valid_data in valDataset:
            val_dataset = data.CustomDataset(valid_data, config.resize)
            valDataloader.append(
                torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                            pin_memory=True))
        print('dataload end')
        print("train set : {}, test set : {}".format(train_dataset.__len__(), test_dataset.__len__()))
        print("random_seed : {}, batch_size : {}, epochs : {}".format(config.rand_seed, config.batch_size, config.epochs))

        classes = ('real', 'fake')
        (self.trainloader, self.testloader, self.classes) = (dataloader, testDataloader, classes)
        # model loading...
        (net, self.feat_size) = model_loader(config.model_name)

        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
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

        self.train()
        self.test()
        self.save(config.ckpt_folder_root)

    
    def train(self):

        # Network initialization...
        self.net.train()
        optimizer = self.init_optimizer

        for i_epoch in range(self.epochs):

            # epoch initialization
            train_loss = 0
            background_loss = 0.
            adv_loss = 0.
            l2_loss = 0.
            correct = 0
            correct_single = 0
            correct_single2 = 0
            total = 0

            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs1 = inputs[:, :3]
                inputs2 = inputs[:, 3:]
                inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(self.device)

                ## dual image conventional process
                # forward
                optimizer.zero_grad()
                (outputs_dual, outputs_single1, outputs_single2) = self.net.module.forward_dual(inputs1, inputs2)

                # backword
                loss_dual = self.criterion(outputs_dual, targets)
                loss_single = self.criterion(outputs_single1, targets * 0 + 1)
                loss_single_0 = self.criterion(outputs_single2, targets)
                
                loss = self.lmd1 * loss_dual + self.lmd2 * loss_single + self.lmd3 * loss_single_0
                loss.backward()

                # optimization
                optimizer.step()

                # logging...
                train_loss += loss.item()
                _, predicted = outputs_dual.max(1)
                _, predicted_single = outputs_single1.max(1)
                _, predicted_single2 = outputs_single2.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                correct_single += predicted_single.eq(targets * 0 + 1).sum().item()
                correct_single2 += predicted_single2.eq(targets).sum().item()

                background_loss += loss_single_0
                adv_loss += loss_single
                l2_loss += loss_dual
#                 print('%02d epoch finished >> Training Loss: %.3f ' % (i_epoch, train_loss / (batch_idx + 1)))
            
            # Print Training Result          
            if self.debug:
                      print('%02d epoch finished >> Training Loss: %.3f C.E. Loss: %.3f adv Loss: %.3f background Loss: %.3f' % (i_epoch, train_loss / (batch_idx + 1), l2_loss / (batch_idx + 1), adv_loss / (batch_idx + 1), background_loss / (batch_idx + 1)))
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

    def save(self, ckpt_folder):
        # state building...
        state = {'net': self.net.state_dict()}

        # Save checkpoint...
        if not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        torch.save(state, './' + ckpt_folder + '/model_ckpt_{}_{}_{}_{}_{}_{}.pth'.format(self.rand_seed, self.batch_size, self.epochs, self.lmd1, self.lmd2, self.lmd3))
        return ckpt_folder