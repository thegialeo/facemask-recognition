import os
import glob
import numpy as np
import cv2
import h5py
from random import shuffle
from tqdm import tqdm

# trainset
print("Converting trainset to hdf5")
for i, subdir in enumerate(os.listdir(os.path.join("./dataset", "train"))):
    img_path = glob.glob(os.path.join("./dataset", "train", subdir, "*"))
    labels = [i for j in range(len(img_path))]
    trainset = list(zip(img_path, labels)) if i == 0 else trainset + list(zip(img_path, labels))

shuffle(trainset)
train_img, train_label = zip(*trainset)

with h5py.File(os.path.join("./dataset", "train.h5"), 'a') as f:
    # images
    f.create_dataset("data", (len(train_img), 256*256*3), np.float32)
    for i, path in enumerate(tqdm(train_img)):
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = img.ravel()
        f["data"][i, ...] = img[None]

    # labels
    f.create_dataset("labels", (len(train_label),), np.int8)
    f["labels"][...] = train_label

print("train.h5 created!")



# testset
print("Converting testset to hdf5")
for i, subdir in enumerate(os.listdir(os.path.join("./dataset", "test"))):
    img_path = glob.glob(os.path.join("./dataset", "test", subdir, "*"))
    labels = [i for j in range(len(img_path))]
    testset = list(zip(img_path, labels)) if i == 0 else testset + list(zip(img_path, labels))

test_img, test_label = zip(*testset)

with h5py.File(os.path.join("./dataset", "test.h5"), 'a') as f:
    # images
    f.create_dataset("data", (len(test_img), 256*256*3), np.float32)
    for i, path in enumerate(tqdm(test_img)):
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = img.ravel()
        f["data"][i, ...] = img[None]

    # labels
    f.create_dataset("labels", (len(test_label),), np.int8)
    f["labels"][...] = test_label

print("test.h5 created!")