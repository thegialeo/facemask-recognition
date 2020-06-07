import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
trainset = datasets.ImageFolder("./dataset/train", transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, num_workers=0, shuffle=False)

mean = 0.
std = 0.
for img, label in tqdm(trainloader):
    img = img.view(img.size(0), img.size(1), -1)
    mean += img.mean(2).sum(0)
    std += img.std(2).sum(0)

mean /= len(trainloader.dataset)
std /= len(trainloader.dataset)

print("mean:", mean.tolist())
print("std:", std.tolist())
