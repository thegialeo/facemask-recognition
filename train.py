import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as opt
from torchvision import models
from GPU import get_device
from dataloader import load_dataset
from trainer import train
from classifier import classifier




if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", dest='mode', action='store',
                        help="Training mode: from_scratch, finetune, small_net")
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
    parser.add_argument("--no_pin_memory", dest="pin_memory", action='store_false',
                        help="Disable pin memory")

    parser.set_defaults(mode='from_scratch', layer=None, num_epochs=100, lr=1e-3, steps_epochs=[50, 80, 100],
                        batch_size=128, num_workers=0, pin_memory=True)
    args = parser.parse_args()

    # set mode to transfer learning, if layer number of mobilenet is given
    if args.layer is not None:
        args.mode = 'transfer'

    # check if GPU available
    device = get_device()

    # load data
    trainset, testset = load_dataset()
    print("Load data")

    # dataloader
    trainloader = data.DataLoader(trainset, args.batch_size, True, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory)
    testloader = data.DataLoader(testset, args.batch_size)

    # model
    print("Initialize Training Mode: {}".format(args.mode))
    if args.mode == 'from_scratch':
        net = models.mobilenet_v2(pretrained=False).features.to(device)
        net.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                       nn.Flatten(),
                                       nn.Linear(1280*7*7, 2, bias=True)).to(device)
    elif args.mode == 'finetune':
        net = models.mobilenet_v2(pretrained=True).features.to(device)
        net.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                       nn.Flatten(),
                                       nn.Linear(1280*7*7, 2, bias=True)).to(device)
    elif args.mode == 'small_net':
        net = nn.Sequential(nn.Flatten(),
                            nn.Linear(3*224*224, 100, bias=True),
                            nn.ReLU(),
                            nn.Linear(100, 100, bias=True),
                            nn.ReLU(),
                            nn.BatchNorm1d(100),
                            nn.Dropout(0.2),
                            nn.Linear(100, 2)).to(device)
    elif args.mode == 'transfer' and args.layer is not None:
        net = models.mobilenet_v2(pretrained=True).features[:args.layer].to(device)
        clf = classifier(args.layer).to(device)
        print("transfer learning mode deprecated")
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
    train(net, trainloader, testloader, criterion, optimizer, scheduler, args.num_epochs, device, args.mode)

    # free GPU space
    del net
    torch.cuda.empty_cache()





