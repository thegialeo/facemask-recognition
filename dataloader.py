import os
import h5py
import torchvision.transforms as transforms
from CustomDatasets import NumpyDataset


def load_dataset():
    """
    Load train- and testset from subfolder 'dataset'.
    Download dataset from: https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1/data
    Run png_to_hdf5.py
    :return: trainset, testset
    """

    train_path = os.path.join('./dataset', 'hdf5_train', 'train.h5')
    test_path = os.path.join('./dataset', 'hdf5_test', 'test.h5')

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