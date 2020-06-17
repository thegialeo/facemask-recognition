import os
import argparse
import math
import torch
import torch.utils.data as data
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import numpy as np
from CustomDatasets import NumpyDataset
import GPU

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", dest='path', action='store',
                    help="File path of image to apply facemask detection")
parser.add_argument("--force_cpu", dest='cpu', action='store_true',
                    help="Force evalution on CPU, even if GPU is available")

parser.set_defaults(path=None, cpu=False)
args = parser.parse_args()

print("Image path: {}".format(args.path))

# check if GPU available
if args.cpu:
    device = 'cpu'
else:
    device = GPU.get_device()
print("Evaluation device: {}".format(device))

# read image
img = cv2.imread(args.path)
shape = img.shape
thickness = int(sum(shape) / 800.)
if shape[0] >= shape[1]:
    long_side = shape[0]
    short_side = shape[1]
else:
    long_side = shape[1]
    short_side = shape[0]
scale_factor = max([long_side / 1280, short_side / 780])
if scale_factor > 1:
    img = cv2.resize(img, (math.floor(long_side / scale_factor), math.floor(short_side / scale_factor)), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Read image")

# load model
mtcnn = MTCNN(image_size=224, margin=0, min_face_size=4, post_process=False, keep_all=True, device=device)
classifier = InceptionResnetV1(classify=True, num_classes=3).to(device)
classifier.load_state_dict(torch.load(os.path.join(".", "models", "net_mtcnn.pt"), map_location=device))
print("Load models")

# get faces
faces = mtcnn(img, save_path="crop_" + args.path)
print("Crop faces and save")
print("Found {} faces".format(len(faces)))

# get boxes
boxes, probs = mtcnn.detect(img)
print("Detect bounding boxes")
print("Found {} bounding boxes".format(len(boxes)))

# dataset + dataloader (pytorch wants dataloader)
dummy_labels = np.zeros(len(faces))
dataset = NumpyDataset(faces.numpy(), dummy_labels)
dataloader = data.DataLoader(dataset, 1)

# plot boxes
for i, (face, _) in enumerate(dataloader):
    out = classifier(face.repeat(2, 1, 1, 1).to(device))
    _, pred = torch.max(out.data, 1)
    if pred[0] == 0:
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), thickness)
    elif pred[0] == 1:
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255, 0, 0), thickness)
    elif pred[0] == 2:
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), thickness)

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title(args.path)
plt.imshow(img)
plt.savefig("results_" + args.path)
plt.show()




