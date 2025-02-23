import struct
from array import array
from os import path
import torchvision
import os
import numpy as np
from PIL import Image

train_dataset = torchvision.datasets.MNIST(root='sample-data/', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='sample-data/', train=False, download=True)

def read(dataset):
    if dataset is 'training':
        path_img = 'sample-data/MNIST/raw/train-images-idx3-ubyte'
        path_lbl = 'sample-data/MNIST/raw/train-labels-idx1-ubyte'
    elif dataset is 'testing':
        path_img = 'sample-data/MNIST/raw/t10k-images-idx3-ubyte'
        path_lbl = 'sample-data/MNIST/raw/t10k-labels-idx1-ubyte'
    else:
        raise ValueError('dataset must be training or testing')

    with open(path_lbl, 'rb') as f_lable:
        _, size = struct.unpack(">II", f_lable.read(8))
        lbl = array("b", f_lable.read()) # signed char array

    with open(path_img, 'rb') as f_img:
        _, size, rows, cols = struct.unpack(">IIII", f_img.read(16))
        img = array("B", f_img.read()) # unsigned char array

    return lbl, img, size, rows, cols

def write_dataset(labels, data, size, rows, cols, output_dir):
    classes = {i: f"class_{i}" for i in range(10)}

    output_dirs = [
        path.join(output_dir, classes[i]) for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)
    print('writing')
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".jpg")
        with open(output_filename, 'wb') as f:
            data_i = [
                data[(i * rows * cols + j * cols) : (i * rows * cols + (j + 1) * cols)]
                for j in range(rows)
            ]
            data_array = np.asarray(data_i)

            im = Image.fromarray(data_array)
            im.save(output_filename)

output_path = 'mnist/'
for dataset in ["training", "testing"]:
    write_dataset(*read(dataset), path.join(output_path, dataset))
