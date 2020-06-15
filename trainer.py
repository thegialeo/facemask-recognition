import os
import pickle
import torch
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import time


def evaluate_acc(net, loader, device):
    """
    Evaluate accuracy of a model.

    :param net: classifier to train on top of mobilenet
    :param loader: dataloader for dataset
    :param device: CPU or GPU
    :return: accuracy of the model on dataset
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)
            label = label.to(device)
            out = net(img)
            _, pred = torch.max(out.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    return 100 * correct / total


def train(model, trainloader, testloader, criterion, optimizer, scheduler, num_epochs, device, mode, detection):
    """
    Train and evaluate a model with CPU or GPU.

    :param model: classifier to train on top of mobilenet
    :param trainloader: dataloader for trainset
    :param testloader: dataloader for testset
    :param criterion: loss function
    :param optimizer: optimization method
    :param scheduler: learning rate scheduler for adaptive learning
    :param num_epochs: number of epochs
    :param device: device to train on (CPU or GPU)
    :param mode: trainings mode
    :param detection: train object detection or classification
    :return: None
    """

    print("Training on:", device)

    if detection:
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
            scheduler.step()
            evaluate(model, testloader, device=device)
    else:
        # log
        loss_hist = []
        train_acc_hist = []
        test_acc_hist = []

        for epoch in range(num_epochs):
            # training
            start = time.time()
            model.train()

            for i, (img, label) in enumerate(trainloader):
                img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(img)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

                # record
                curr_loss = torch.mean(loss).item()
                running_loss = (curr_loss if ((i==0) and (epoch==0)) else running_loss + curr_loss)

            scheduler.step()

            # evaluation
            model.eval()
            running_loss /= len(trainloader)
            train_acc = evaluate_acc(model, trainloader, device)
            test_acc = evaluate_acc(model, testloader, device)
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

            # create name extension
            name = '_' + mode

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
            torch.save(model.state_dict(), os.path.join('./models', 'net' + name + '.pt'))

            # save logs
            file = open(os.path.join('./logs', 'log' + name + '.txt'), 'w')
            print('Final Loss:', loss_hist[-1], file=file)
            print('Final Train Accuracy:', train_acc_hist[-1], file=file)
            print('Final Test Accuracy:', test_acc_hist[-1], file=file)

            # save variables
            with open(os.path.join('./logs', 'log' + name + '.pkl'), 'wb') as f:
                pickle.dump([loss_hist, train_acc_hist, test_acc_hist], f)
