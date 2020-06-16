import torch
from torch.utils import data

class NumpyDataset(data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)



class DetectionDataset(data.Dataset):
    def __init__(self, data, target, target_dict, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = target
        self.target_dict = target_dict
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y_item_list = self.target_dict[self.target[index]]

        boxes = []
        labels = []
        for box in y_item_list:
            if box[0] == 'good':
                label = 1
            elif box[0] == 'bad':
                label = 2
            elif box[0] == 'none':
                label = 3
            else:
                print("Something is wrong with label: {}".format(box[0]))
            labels.append(label)
            boxes.append(box[1])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        y = {}
        y["boxes"] = boxes
        y["labels"] = labels
        y["image_id"] = torch.tensor([index])
        y["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        y["iscrowd"] = torch.zeros((len(y_item_list),), dtype=torch.int64) # assume no instances are crowd

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
