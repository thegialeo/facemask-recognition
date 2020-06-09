import os
import  argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as opt
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from HDF5Dataset import HDF5Dataset

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
    :return: trainset, testset
    """

    train_path = os.path.join('./dataset', 'train.h5')
    test_path = os.path.join('./dataset', 'test.h5')

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = HDF5Dataset(train_path, False, True, transform=transform)
    testset = HDF5Dataset(test_path, False, True, transform=transform)

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
            out = net(img)
            _, pred = torch.max(out.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    return 100 * correct / total


def train(net, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, device, mobilenet=None):
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
        print('epoch {} \t loss {:.5f} \t train acc {:.3f} \t test acc {:.3f}'.format(epoch+1, running_loss,
                                                                                      train_acc, test_acc))

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
        plt.savefig(os.path.join('./plots', 'loss.png'))

        # save train accuracy plot
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(train_acc_hist)
        plt.grid(True, which='both')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.savefig(os.path.join('./plots', 'train_acc.png'))

        # save test accuracy plot
        plt.figure(num=None, figsize=(8, 6))
        plt.plot(test_acc_hist)
        plt.grid(True, which='both')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.savefig(os.path.join('./plots', 'test_acc.png'))

        # close all figures
        plt.close("all")

        # save model weights
        torch.save(net.state_dict(), os.path.join('./models', 'net.pt'))

        # save logs
        file = open(os.path.join('./logs', 'log.txt'), 'w')
        print('Final Loss:', loss_hist[-1], file=file)
        print('Final Train Accuracy:', train_acc_hist[-1], file=file)
        print('Final Test Accuracy:', test_acc_hist[-1], file=file)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
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

    parser.set_defaults(num_epochs=100, lr=1e-2, steps_epochs=[50, 80, 100], batch_size=128, num_workers=8)
    args = parser.parse_args()

    # check if GPU available
    device = get_device()

    # load data
    trainset, testset = load_dataset()
    print("Load data")

    # dataloader
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    testloader = data.DataLoader(testset, args.batch_size)

    # model
    net = models.mobilenet_v2(pretrained=False).to(device)

    # scheduler + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(net.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, args.steps_epochs, 0.1)

    # training
    train(net, trainloader, testloader, criterion, optimizer, scheduler, args.num_epochs, device, mobilenet=None)





