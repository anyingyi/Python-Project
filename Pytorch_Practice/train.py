import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np

from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from models import *
from utils.config import get_args
from utils.utils import progress_bar

dir_img = ''
dir_mask = ''
dir_checkpoint = 'checkpoints/'

# Data
logging.info(f'==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def train_net(net,
              device,
              epochs=22,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    # here is prepare Dataset
    # network optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    n_train = trainloader.__len__()
    n_val = testloader.__len__()
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')

    if len(classes) > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            global_step += 1
            if global_step %(n_train // (10 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.','/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                # val_score = eval_net(net, testloader, device)
                # scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                writer.add_images('images', inputs, global_step)
                if len(classes) == 1:
                    writer.add_images('masks/true', inputs, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(inputs) > 0.5, global_step)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if save_cp:
            if epoch % 5 ==0:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def test_net(net,
              device,
              epochs=22,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    for epoch in range(epochs):
        pass

        if save_cp:
            if epoch % 5 ==0:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # change here to adapt to your model
    # TODO
    net = VGG('VGG16')
    logging.info(f'Network:\n'
                 f'\t{net.features} input channels\n'
                 f'\t{net.classifier} output channels (classes)\n')
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net = net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)


        # test_net(net=net,
        #          epochs=22,
        #          batch_size=1,
        #          lr=0.001,
        #          device=device,
        #          img_scale=0.5,
        #          val_percent=0.1,
        #          save_cp=True)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)