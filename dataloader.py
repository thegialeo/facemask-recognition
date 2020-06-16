import os
import h5py
import json
from torch.utils import data
import torchvision.transforms as transforms
import xmltodict
import cv2
import matplotlib.pyplot as plt
from CustomDatasets import NumpyDataset, DetectionDataset

def getImageNames():
    image_names = []

    for dirname, _, filenames in os.walk(os.path.join('.', 'dataset', 'detection', 'images')):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            extension = fullpath[len(fullpath) - 4:]
            if extension != '.xml':
                image_names.append(filename)

    return image_names


def get_path(image_name):
    home_path = os.path.join('.', 'dataset', 'detection')
    image_path = os.path.join(home_path, 'images', image_name)

    if image_name[-4:] == 'jpeg':
        label_name = image_name[:-5] + '.xml'
    else:
        label_name = image_name[:-4] + '.xml'

    label_path = os.path.join(home_path, 'labels', label_name)

    return image_path, label_path


def parse_xml(label_path):
    x = xmltodict.parse(open(label_path, 'rb'))
    item_list = x['annotation']['object']

    # when image has only one bounding box
    if not isinstance(item_list, list):
        item_list = [item_list]

    result = []

    for item in item_list:
        name = item['name']
        bndbox = [int(item['bndbox']['xmin']), int(item['bndbox']['ymin']),
                  int(item['bndbox']['xmax']), int(item['bndbox']['ymax'])]
        result.append((name, bndbox))

    size = [int(x['annotation']['size']['width']),
            int(x['annotation']['size']['height'])]

    return result, size


def visualize_image(image_name, bndbox=True):

    image_path, label_path = get_path(image_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if bndbox:
        labels, size = parse_xml(label_path)
        thickness = int(sum(size) / 400.)

        for label in labels:
            name, bndbox = label

            if name == 'good':
                cv2.rectangle(image, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (0, 255, 0), thickness)
            elif name == 'bad':
                cv2.rectangle(image, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), (255, 0, 0), thickness)
            else:  # name == 'none'
                cv2.rectangle(image, bndbox[0], bndbox[1], (0, 0, 255), thickness)

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title(image_name)
    plt.imshow(image)
    plt.show()



def load_dataset(detection=False):
    """
    Load train- and testset from subfolder 'dataset'.
    Download dataset from: https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1/data
    Run png_to_hdf5.py
    :param detection: If true, load detection dataset instead
    :return: trainset, testset
    """
    if detection:
        path = os.path.join('.', 'dataset', 'hdf5_detection')
        f_h5py = h5py.File(os.path.join(path, 'detection.h5') , 'r', driver=None)

        with open(os.path.join(path, 'label.txt'), 'r') as file:
            label_dict = json.load(file)

        with open(os.path.join(path, 'size.txt'), 'r') as file:
            size_dict = json.load(file)

        x_data = f_h5py['data'][()].reshape(-1, 3, 224, 224)
        y_data = f_h5py['label'][()]

        # normalization
        transform = transforms.Normalize([0.4454523026943207, 0.4200827479362488, 0.4140508770942688],
                                          [0.2556115388870239, 0.24527578055858612, 0.24393153190612793])

        dataset = DetectionDataset(x_data, y_data, label_dict, size_dict, transform)
        trainset, testset = data.random_split(dataset, [542, 136])

    else:
        train_path = os.path.join('.', 'dataset', 'hdf5_train', 'train.h5')
        test_path = os.path.join('.', 'dataset', 'hdf5_test', 'test.h5')

        f_train = h5py.File(train_path, 'r', driver=None)
        f_test = h5py.File(test_path, 'r', driver=None)

        x_train = f_train['data'][()].reshape(-1, 3, 224, 224)
        y_train = f_train['label'][()]
        x_test = f_test['data'][()].reshape(-1, 3, 224, 224)
        y_test = f_test['label'][()]

        # data augmentation
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                                              transforms.RandomPerspective(),
                                              transforms.RandomResizedCrop((224, 224), (0.7, 1.0)),
                                              transforms.ToTensor(),
                                              transforms.RandomErasing(),
                                              transforms.Normalize(
                                                  [0.47329697012901306, 0.412136435508728, 0.38234803080558777],
                                                  [0.24916036427021027, 0.23281708359718323, 0.23224322497844696])])

        transform_test = transforms.Normalize([0.47329697012901306, 0.412136435508728, 0.38234803080558777],
                                              [0.24916036427021027, 0.23281708359718323, 0.23224322497844696])

        trainset = NumpyDataset(x_train, y_train, transform_train)
        testset = NumpyDataset(x_test, y_test, transform_test)

    return trainset, testset