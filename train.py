import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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

    train_path = os.path.join('./dataset', 'train')
    test_path = os.path.join('./dataset', 'test')

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.47333890199661255, 0.4121790826320648, 0.38239002227783203],
                                        std=[0.24937640130519867, 0.23304365575313568, 0.232471764087677])])

    trainset = datasets.ImageFolder(train_path, transform)
    testset = datasets.ImageFolder(test_path, transform)

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
        for i, (img, label) in enumerate(tqdm(loader)):
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
        print()
        print(80 * '-')
        print()
        print("Epoch: {}".format(epoch+1))
        print("Training:")
        #start = time.time()
        net.train()

        for i, (img, label) in enumerate(tqdm(trainloader)):
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
        print("Evaluating:")
        net.eval()
        # compute accuracy
        train_acc = evaluate(net, trainloader, device, mobilenet)
        test_acc = evaluate(net, testloader, device, mobilenet)
        # record
        loss_hist.append(running_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        # print loss and accuracy
        print('epoch {}, loss {:.5f}, train acc {:.3f}, test acc {:.3f}'.format(
            epoch+1, running_loss, train_acc, test_acc))

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





