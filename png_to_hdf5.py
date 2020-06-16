import os
import argparse
import glob
import json
import numpy as np
import cv2
import h5py
from random import shuffle
from tqdm import tqdm
from dataloader import getImageNames, get_path, parse_xml



def convert_trainset():
    print("Converting trainset to hdf5")
    for i, subdir in enumerate(os.listdir(os.path.join(".", "dataset", "train"))):
        img_path = glob.glob(os.path.join(".", "dataset", "train", subdir, "*"))
        labels = [i for j in range(len(img_path))]
        trainset = list(zip(img_path, labels)) if i == 0 else trainset + list(zip(img_path, labels))

    shuffle(trainset)
    train_img, train_label = zip(*trainset)

    if not os.path.exists(os.path.join('.', 'dataset', "hdf5_train")):
        os.mkdir(os.path.join('.', 'dataset', "hdf5_train"))

    with h5py.File(os.path.join(".", "dataset", "hdf5_train", "train.h5"), 'a') as f:
        # images
        f.create_dataset("data", (len(train_img), 224*224*3), np.float32)
        for i, path in enumerate(tqdm(train_img)):
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = img.ravel()
            f["data"][i, ...] = img[None]

        # labels
        f.create_dataset("label", (len(train_label),), np.int8)
        f["label"][...] = train_label

    print("train.h5 created!")



def convert_testset():
    print("Converting testset to hdf5")
    for i, subdir in enumerate(os.listdir(os.path.join(".", "dataset", "test"))):
        img_path = glob.glob(os.path.join(".", "dataset", "test", subdir, "*"))
        labels = [i for j in range(len(img_path))]
        testset = list(zip(img_path, labels)) if i == 0 else testset + list(zip(img_path, labels))

    test_img, test_label = zip(*testset)

    if not os.path.exists(os.path.join('.', 'dataset', "hdf5_test")):
        os.mkdir(os.path.join('.', 'dataset', "hdf5_test"))

    with h5py.File(os.path.join(".", "dataset", "hdf5_test", "test.h5"), 'a') as f:
        # images
        f.create_dataset("data", (len(test_img), 224*224*3), np.float32)
        for i, path in enumerate(tqdm(test_img)):
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = img.ravel()
            f["data"][i, ...] = img[None]

        # labels
        f.create_dataset("label", (len(test_label),), np.int8)
        f["label"][...] = test_label

    print("test.h5 created!")



def convert_detecttion_set():
    print("Converting detection dataset to hdf5")
    all_img = getImageNames()

    if not os.path.exists(os.path.join('.', 'dataset', 'hdf5_detection')):
        os.mkdir(os.path.join('.', 'dataset', 'hdf5_detection'))

    with h5py.File(os.path.join(".", "dataset", "hdf5_detection", "detection.h5"), 'a') as f:
        f.create_dataset("data", (len(all_img), 224*224*3), np.float32)
        f.create_dataset("label", (len(all_img),), h5py.string_dtype())
        label = {}
        size = {}
        for i, name in enumerate(tqdm(all_img)):
            image_path, label_path = get_path(name)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            img = img.ravel()
            f["data"][i, ...] = img[None]
            f["label"][i, ...] = label_path
            label[label_path], size[label_path] = parse_xml(label_path)
        with open(os.path.join(".", "dataset", "hdf5_detection", "label.txt"), 'w') as file:
            json.dump(label, file)
        with open(os.path.join(".", "dataset", "hdf5_detection", "size.txt"), 'w') as file:
            json.dump(size, file)

    print("detection.h5 created!")
    print("label.txt created!")
    print("size.txt created!")



if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', action='store',
                        help="mode: all, train, test, detection")
    parser.set_defaults(mode='all')
    args = parser.parse_args()

    if args.mode == 'all':
        convert_trainset()
        convert_testset()
        convert_detecttion_set()
    elif args.mode == 'train':
        convert_trainset()
    elif args.mode == 'test':
        convert_testset()
    elif args.mode == 'detection':
        convert_detecttion_set()
    else:
        print("mode {} not defined!".format(args.mode))