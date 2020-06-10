import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as opt
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import h5py
from NumpyDataset import NumpyDataset
from classifier import classifier

def get_device():
    """
    Select GPU if available, else CPU.

    :return: device variable used for further training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def load_dataset():
    """
    Load train- and testset from subfolder 'dataset'.
    Download dataset from: https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1/data
    Run png_to_hdf5.py
    :return: trainset, testset
    """

    train_path = os.path.join('./dataset', 'hdf5_train', 'train.h5')
    test_path = os.path.join('./dataset', 'hdf5_test', 'test.h5')

    f_train = h5py.File(train_path, 'r', driver='core')
    f_test = h5py.File(test_path, 'r', driver='core')

    x_train = f_train['data'].value.reshape(-1, 3, 256, 256)
    y_train = f_train['label'].value
    x_test = f_test['data'].value.reshape(-1, 3, 256, 256)
    y_test = f_test['label'].value

    #transform = transforms.Compose([transforms.ToTensor()])

    trainset = NumpyDataset(x_train, y_train)
    testset = NumpyDataset(x_test, y_test)

    return trainset, testset


def evaluate(net, loader, device, mobilenet=None):
    """
    Evaluate accuracy of a model.

    :param net: model to evaluate
    :param loader: dataloader for dataset
    :param device: CPU or GPU
    :param mobilenet: provide pretrained mobilenet, if applicable
    :return: accuracy of the model on dataset
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)
            label = label.to(device)
            if mobilenet is not None:
                feat = mobilenet(img)
                feat = feat.view(feat.size(0), -1)
            else:
                feat = img
            out = net(feat)
            _, pred = torch.max(out.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    return 100 * correct / total


def train(net, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, device, mobilenet=None, name=''):
    """
    Train and evaluate a model with CPU or GPU.

    :param net: model to train
    :param trainloader: dataloader for trainset
    :param testloader: dataloader for testset
    :param criterion: loss function
    :param optimizer: optimization method
    :param scheduler: learning rate scheduler for adaptive learning
    :param num_epochs: number of epochs
    :param device: device to train on (CPU or GPU)
    :param mobilenet: pretrained mobilenet (optional)
    :param name: name extension to save log files and graphs
    :return: None
    """

    print("Training on:", device)

    # log
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []

    for epoch in range(num_epochs):
        # training
        start = time.time()
        net.train()

        for i, (img, label) in enumerate(trainloader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            if mobilenet is not None:
                feat = mobilenet(img)
                feat = feat.view(feat.size(0), -1)
            else:
                feat = img
            out = net(feat)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            # record
            curr_loss = torch.mean(loss).item()
            running_loss = (curr_loss if ((i==0) and (epoch==0)) else (1 - 0.01) * running_loss + 0.01 * curr_loss)

        scheduler.step()

        # evaluation
        net.eval()
        train_acc = evaluate(net, trainloader, device, mobilenet)
        test_acc = evaluate(net, testloader, device, mobilenet)
        loss_hist.append(running_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        print('epoch {} \t loss {:.5f} \t train acc {:.3f} \t test acc {:.3f} \t time {:.1f} sec'.format(
            epoch+1, running_loss, train_acc, test_acc, time.time() - start))

        # create directory
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if not os.path.exists('./logs'):
            os.mkdir('./logs')

        # save loss plot
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(loss_hist)
        plt.grid(True, which="both")
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('average loss', fontsize=14)
        plt.savefig(os.path.join('./plots', 'loss' + name + '.png'))

        # save train accuracy plot
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(train_acc_hist)
        plt.grid(True, which='both')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.savefig(os.path.join('./plots', 'train_acc' + name + '.png'))

        # save test accuracy plot
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(test_acc_hist)
        plt.grid(True, which='both')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.savefig(os.path.join('./plots', 'test_acc' + name + '.png'))

        # close all figures
        plt.close("all")

        # save model weights
        torch.save(net.state_dict(), os.path.join('./models', 'net' + name + '.pt'))

        # save logs
        file = open(os.path.join('./logs', 'log' + name + '.txt'), 'w')
        print('Final Loss:', loss_hist[-1], file=file)
        print('Final Train Accuracy:', train_acc_hist[-1], file=file)
        print('Final Test Accuracy:', test_acc_hist[-1], file=file)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", dest='mode', action='store',
                        help="Training mode: from_scratch, finetune")
    parser.add_argument("--transfer_layer", dest='layer', action='store', type=int,
                        help="Layer of mobilenet to get the features from")
    parser.add_argument("--num_epoch", dest='num_epochs', action='store', type=int,
                        help="Number of Epochs")
    parser.add_argument("--learning_rate", dest='lr', action='store', type=float,
                        help="Learning rate")
    parser.add_argument("--adaptive_learning", dest='steps_epochs', action='store', nargs='+', type=int,
                        help="Epochs at which to drop the learning rate by factor 10")
    parser.add_argument("--batch_size", dest='batch_size', action='store', type=int,
                        help="Batch size for training")
    parser.add_argument("--num_workers", dest='num_workers', action='store', type=int,
                        help="Number of workers for dataloader")

    parser.set_defaults(mode='from_scratch', layer=None, num_epochs=100, lr=1e-2, steps_epochs=[50, 80, 100],
                        batch_size=128, num_workers=0)
    args = parser.parse_args()

    # set mode to transfer learning, if layer number of mobilenet is given
    if args.layer is not None:
        args.mode = 'transfer'

    # name extension
    name = args.mode

    # check if GPU available
    device = get_device()

    # load data
    trainset, testset = load_dataset()
    print("Load data")

    # dataloader
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    testloader = data.DataLoader(testset, args.batch_size)

    # model
    print("Initialize Training Mode: {}".format(args.mode))
    if args.mode == 'from_scratch':
        mobilenet = None
        net = models.mobilenet_v2(pretrained=False).to(device)
    elif args.mode == 'finetune':
        mobilenet = None
        net = models.mobilenet_v2(pretrained=True).to(device)
    elif args.mode == 'transfer' and args.layer is not None:
        mobilenet = models.mobilenet_v2(pretrained=True).features[:args.layer]
        batch = next(iter(trainloader))
        net = classifier(batch)
        name += '{}'.format(args.layer)
    else:
        if args.mode == 'transfer' and args is None:
            print("Error: Layer of mobilenet from which to get features is not specified!")
        else:
            print("Error: Training Mode {} is not defined!".format(args.mode))

    # scheduler + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(net.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, args.steps_epochs, 0.1)

    # training
    train(net, trainloader, testloader, criterion, optimizer, scheduler, args.num_epochs, device, mobilenet, name)





