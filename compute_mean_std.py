import os
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

def compute_mean_std(path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

    mean = 0.
    std = 0.
    for img, label in tqdm(dataloader):
        img = img.view(img.size(0), img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    print("mean:", mean.tolist())
    print("std:", std.tolist())

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', action='store',
                        help="mode: single, detection")
    parser.set_defaults(mode='single')
    args = parser.parse_args()

    if args.mode == 'single':
        print("Compute mean and std for single person dataset:")
        path = os.path.join('.', 'dataset', 'train')
        compute_mean_std(path)
    elif args.mode == 'detection':
        print("Compute mean and std for detection dataset:")
        path = os.path.join('.', 'dataset', 'detection_mean_std')
        compute_mean_std(path)


